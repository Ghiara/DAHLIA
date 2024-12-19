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


class SequentialBlockInsertionAndStacking(Task):
    """Sequentially insert colored blocks into corresponding zones, then stack them on a pallet."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "insert {color} block into the corresponding zone, then stack in sequence on the pallet"
        self.task_completed_desc = "completed block insertion and stacking."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors, sizes, and URDFs for the blocks, zones, and pallet.
        block_colors = ['red', 'blue', 'green', 'yellow']
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions
        block_urdf = 'block/block.urdf'
        zone_urdf = 'zone/zone.urdf'
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_size = (0.15, 0.15, 0.02)  # x, y, z dimensions for the pallet

        # Add pallet at a fixed position.
        pallet_pose = ((0.5, 0, 0.01), (0, 0, 0, 1))  # Fixed position and orientation
        env.add_object(pallet_urdf, pallet_pose, 'fixed', scale=1.2)

        blocks = []
        zones = []

        # Add blocks and zones for each color.
        for color in block_colors:
            # Set block pose randomly within bounds but ensure it's not on the pallet initially.
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[color] + [1])
            blocks.append((block_id, color))
            if color == 'green':
                id_1 = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(id_1, -1, rgbaColor=utils.COLORS['blue'] + [1])
                blocks.append((id_1, 'blue'))
                id_2 = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(id_2, -1, rgbaColor=utils.COLORS['green'] + [1])
                blocks.append((id_2, 'green'))

            # Set zone pose near the pallet but ensure each zone is distinct.
            zone_pose = ((0.35 + (block_colors.index(color)) * 0.1, 0, 0.01),
                         (0, 0, 0, 1))  # Increment x position for each zone
            zone_id = env.add_object(zone_urdf, zone_pose, 'fixed')
            p.changeVisualShape(zone_id, -1,
                                rgbaColor=utils.COLORS[color] + [0.5])  # Slightly transparent
            zones.append(zone_id)

        # Define target poses for stacking on the pallet in a pyramid shape.
        stack_poses = [
            ((0.5, -0.05, 0.06), (0, 0, 0, 1)),  # Bottom row: red
            ((0.5, 0, 0.06), (0, 0, 0, 1)),  # Bottom row: blue
            ((0.5, 0.05, 0.06), (0, 0, 0, 1)),  # Bottom row: green
            ((0.5, -0.025, 0.1), (0, 0, 0, 1)),  # Middle row: blue
            ((0.5, 0.025, 0.1), (0, 0, 0, 1)),  # Middle row: green
            ((0.5, 0, 0.14), (0, 0, 0, 1)),  # Top row: yellow
        ]

        # Add goals for each block to be inserted into the corresponding zone, then stacked.
        for i, (block_id, color) in enumerate(blocks):
            # Insertion goal: block into corresponding zone.
            insertion_goal_pose = ((0.35 + (block_colors.index(color)) * 0.1, 0, 0.02),
                                   (0, 0, 0, 1))  # Same as zone pose but slightly higher
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)),
                          targ_poses=[insertion_goal_pose], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 8,
                          language_goal=self.lang_template.format(color=color))

            # Stacking goal: block on the pallet in sequence.
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)), targ_poses=[stack_poses[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 8,
                          language_goal=self.lang_template.format(color=color))
