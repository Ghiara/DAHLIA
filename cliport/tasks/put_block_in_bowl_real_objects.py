import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class PutBlockInMatchingBowlReal(Task):
    """Put Block in Matching Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put all the blocks in the bowls with matching colors"
        self.task_completed_desc = "done placing blocks in bowls."
        self.seed = 0
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        np.random.seed(self.seed)
        random.seed(self.seed)
        n_bowls = np.random.randint(3, 8)
        n_blocks = np.random.randint(3, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_bowls)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        # bowl_urdf = 'bowl/bowl.urdf'
        bowl_urdf = 'container/container-template.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            blocks.append((block_id, (0, None)))

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.eye(len(blocks)), bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 2

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        distractor_bowl = []
        while n_distractors < max_distractors and distractor_colors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            if is_block:
                distractor_block.append(color_name)
            else:
                distractor_bowl.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls] + distractor_bowl
        np.random.shuffle(bowl_list)
        self.scene_description = f"On the table, there are {n_blocks + len(distractor_block)} blocks. " \
                                 f"Their colors are {', '.join(block_list)}. " \
                                 f"There are {n_bowls + len(distractor_bowl)} bowls. " \
                                 f"Their colors are {', '.join(bowl_list)}."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
