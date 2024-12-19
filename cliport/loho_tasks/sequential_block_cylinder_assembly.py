import numpy as np
import os
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class SequentialBlockAndCylinderAssembly(Task):
    """
    Assemble a structure in a zone starting with a base layer of four blocks 
    (red, blue, green, yellow) arranged side by side, followed by stacking a cylinder 
    of matching color on top of each block, and finally placing a small block on top 
    of each cylinder, creating a four-step, color-coordinated tower for each color.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "assemble a structure in a zone with blocks and cylinders in color order: red, blue, green, yellow."
        self.task_completed_desc = "completed sequential block and cylinder assembly."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""

    def reset(self, env):
        super().reset(env)

        # Define colors for easy reference
        colors = ['red', 'blue', 'green', 'yellow']
        color_map = {color: utils.COLORS[color] for color in colors}

        # Add pallet
        pallet_size = (0.6, 0.6, 0.02)  # Slightly larger pallet to hold all structures
        pallet_pose = (
        (0.5, 0, 0.01), utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2)))  # Fixed pose on the table
        env.add_object('zone/zone.urdf', pallet_pose, 'fixed', scale=2)

        # Initial positions for the base blocks on the pallet
        base_positions = [(0.45, -0.05, 0.02), (0.55, -0.05, 0.02), (0.45, 0.05, 0.02),
                          (0.55, 0.05, 0.02)]
        block_size = 1  # Uniform scaling factor for blocks
        base_block_size = np.array([0.04, 0.04, 0.04])
        block_pose = []
        cylinder_pose = []
        small_block_pose = []
        block_id = []
        cylinder_id = []
        small_block_id = []
        # Add base layer blocks
        for i, color in enumerate(colors):
            po = self.get_random_pose(env, base_block_size)
            block_pose.append((base_positions[i], (0, 0, 0, 1)))
            block_id.append(env.add_object('block/block.urdf', po, scale=block_size, color=color))

        # Add cylinders on top of each base block
        cylinder_positions = [(pos[0], pos[1], pos[2] + 0.04) for pos in
                              base_positions]  # Slightly above the blocks
        for i, color in enumerate(colors):
            po = self.get_random_pose(env, base_block_size)
            cylinder_pose.append((cylinder_positions[i], (0, 0, 0, 1)))
            cylinder_id.append(
                env.add_object('cylinder/cylinder-template.urdf', po, scale=block_size / 2,
                               color=color))

        # Add small blocks on top of each cylinder
        small_block_positions = [(pos[0], pos[1], pos[2] + 0.08) for pos in
                                 cylinder_positions]  # On top of cylinders
        for i, color in enumerate(colors):
            po = self.get_random_pose(env, base_block_size)
            small_block_pose.append((small_block_positions[i], (0, 0, 0, 1)))
            small_block_id.append(
                env.add_object('block/small.urdf', po, scale=2 * block_size, color=color))

        for i in range(4):
            self.add_goal(objs=[block_id[i]], matches=np.ones((1, 1)), targ_poses=[block_pose[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 12,
                          symmetries=[np.pi / 2],
                          language_goal="place a block in sequence")
            self.add_goal(objs=[cylinder_id[i]], matches=np.ones((1, 1)),
                          targ_poses=[cylinder_pose[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 12,
                          language_goal="place a cylinder in sequence")
            self.add_goal(objs=[small_block_id[i]], matches=np.ones((1, 1)),
                          targ_poses=[small_block_pose[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 12,
                          symmetries=[np.pi / 2],
                          language_goal="place a small block in sequence")
