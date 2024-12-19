import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class ColorGradientTower(Task):
    """Create a color gradient tower by stacking blocks from darkest to lightest shade."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the blocks from darkest to lightest shade to create a color gradient tower in the zone"
        self.task_completed_desc = "done stacking color gradient tower."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors, sizes, and URDFs for the blocks and pallet.
        block_colors = ['black', 'gray', 'white']
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the block
        block_urdf = 'block/block.urdf'
        pallet_size = (0.15, 0.15, 0.02)  # x, y, z dimensions for the pallet
        pallet_urdf = 'pallet/pallet.urdf'

        # Add pallet at a fixed position.
        pallet_pose = ((0.5, 0, 0.01), (0, 0, 0, 1))  # Fixed position and orientation
        env.add_object(pallet_urdf, pallet_pose, 'fixed')

        blocks = []

        # Add two blocks of each color.
        for color in block_colors:
            for _ in range(2):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[color] + [1])
                blocks.append(block_id)

        # Define target poses for stacking blocks in a gradient pattern on the pallet.
        # Each block is slightly rotated from the one below to form a spiral pattern.
        stack_height = 0.04  # Initial height of the first block
        rotation_angle = 0  # Initial rotation angle
        stack_poses = []
        for i in range(len(blocks)):
            pos = (
            0.5, 0, stack_height + i * (block_size[2] + 0.005))  # Increment height for each block
            rot = p.getQuaternionFromEuler((0, 0, rotation_angle))
            stack_poses.append((pos, rot))
            rotation_angle += np.pi / 6  # Increment rotation angle for spiral pattern

        env.add_object('zone/zone.urdf', ((0.5, 0, 0.01), (0, 0, 0, 1)), 'fixed',
                       scale=2, color=utils.get_random_color()[1][0])

        # Add goals for each block to be stacked in the color gradient tower.
        for i, block_id in enumerate(blocks):
            color = block_colors[i // 2]  # Determine color based on block index
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[stack_poses[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(blocks),
                          language_goal=language_goal)
