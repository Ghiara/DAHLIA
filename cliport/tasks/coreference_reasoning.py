import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class StackBlockOfSameColor(Task):
    """
    Stack ALL Blocks of the same color together.
    There are at most three kinds of color and at most 3 blocks with the same color.
   """

    def __init__(self):
        super().__init__()
        self.n_block_colors = np.random.randint(1, 4)
        if self.n_block_colors == 1:
            self.n_blocks = np.random.randint(2, 4)
        elif self.n_block_colors == 2:
            self.n_blocks = np.random.randint(4, 7)
        else:  # 3
            self.n_blocks = np.random.randint(6, 10)

        self.n_distractors = np.random.randint(0, 3)
        self.max_steps = self.n_blocks + 2
        self.lang_template = "stack all the blocks of the same color together"
        self.task_completed_desc = "done stacking block."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Block colors.
        color_names = self.get_colors()
        random.shuffle(color_names)
        block_color_names = random.sample(color_names, self.n_block_colors)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        base_poses = []
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            if i < self.n_block_colors:
                base_poses.append(block_pose)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1,
                                rgbaColor=block_colors[i % self.n_block_colors] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Put later blocks over the first blocks of the same color
        place_poses = []
        for block_idx in range(self.n_block_colors, self.n_blocks):
            base_pose = base_poses[block_idx % self.n_block_colors]
            pos, rot = base_pose
            height = block_size[2]
            place_pos = (pos[0], pos[1], pos[2] + height * (block_idx // self.n_block_colors))
            place_poses.append((place_pos, rot))

        match_matrix = np.eye(self.n_blocks - self.n_block_colors)
        self.goals.append(
            (objs[self.n_block_colors:], match_matrix, place_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)

        # Add distractor blocks
        distractor_color_names = [c for c in utils.COLORS if c not in block_color_names]
        distractor_color = [utils.COLORS[c] for c in distractor_color_names]
        n_distractors = min(self.n_distractors, len(distractor_color))
        for i in range(n_distractors):
            pose = self.get_random_pose(env, block_size)
            distractor_id = env.add_object(block_urdf, pose)
            p.changeVisualShape(distractor_id, -1, rgbaColor=distractor_color[i] + [1])

        block_color_list = block_color_names + distractor_color_names[:n_distractors]
        np.random.shuffle(block_color_list)
        self.scene_description = f"On the table, there are {self.n_blocks + n_distractors} blocks, " \
                                 f"and their colors are {', '.join(block_color_list)}."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
