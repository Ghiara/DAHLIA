import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class LineConstruction(Task):
    """Manipulate the blocks on the table to form a straight line shape """

    def __init__(self):
        super().__init__()
        self.n_blocks = random.randint(2, 5)
        self.max_steps = self.n_blocks + 2

        self.lang_template = "manipulate all the blocks to form a straight line shape"
        self.task_completed_desc = "done manipulating blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        self.scene_description = f"On the table, there are {self.n_blocks} blocks. " \
                                 f"Their colors are {color_names[:self.n_blocks]}. "

        # Associate placement locations for goals, use the first block as the base location.
        # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #              (0, 0.05, 0.03), (0, -0.025, 0.08),
        #              (0, 0.025, 0.08), (0, 0, 0.13)]
        place_pos = [(0, 0, 0.03 + 0.05 * i) for i in range(self.n_blocks)]
        # place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        match_matrix = np.eye(self.n_blocks)
        # match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, targs, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
