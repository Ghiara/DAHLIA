import numpy as np
import pybullet as p
import os
from cliport.tasks.task import Task
from cliport.utils import utils


class MultiLevelCylinderTower(Task):
    """Construct a multi-level tower with alternating layers of cylinders and three blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "construct a multi-level tower with alternating layers of cylinders and 3 blocks in the zone"
        self.task_completed_desc = "done constructing multi-level cylinder tower."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the base position for the tower.
        anchor_pose = (
        np.array([0.5, 0.2, 0.015]), np.array([0, 0, 0, 1]))  # Base height is 0.03/2 = 0.015
        self.add_corner_anchor_for_pose(env, anchor_pose)

        # Define colors for cylinders and blocks.
        cylinder_color = utils.COLORS['blue']
        block_color = utils.COLORS['white']
        n_cylinder = 2
        n_block = n_cylinder - 1

        # Define sizes for cylinders and blocks.
        cylinder_size = 0.04  # Scale factor for the cylinder URDF.
        block_size = 0.04  # Scale factor for the block URDF.

        cylinders = []
        blocks = []

        cylinder_shape = os.path.join(self.assets_root, 'kitting', '11.obj')
        block_shape = os.path.join(self.assets_root, 'kitting', '08.obj')

        template = 'kitting/object-template.urdf'
        block_urdf = 'block/block.urdf'
        cylinder_urdf = 'cylinder/cylinder-template.urdf'

        # Load URDFs for cylinder and block.
        for i in range(n_block):
            blocks.append([])
            for j in range(3):
                block_pose = self.get_random_pose(env, (0.04, 0.04, 0.01), bound=np.array(
                    [[0.2, 0.45], [-0.5, 0], [0, 0.3]]))
                # scale = [0.002, 0.002, 0.002]
                # replace = {'FNAME': (block_shape,), 'SCALE': scale, 'COLOR': block_color}
                # block_urdf = self.fill_template(template, replace)
                block_id = env.add_object(block_urdf, block_pose, color=block_color)
                blocks[i].append(block_id)

        for i in range(n_cylinder):
            cylinder_pose = self.get_random_pose(env, (0.04, 0.04, 0.001))
            scale = [0.008, 0.008, 0.001]
            replace = {'FNAME': (cylinder_shape,), 'SCALE': scale, 'COLOR': cylinder_color}
            cylinder_urdf = self.fill_template(template, replace)
            replace = {'DIM': (0.08, 0.08, 0.04), 'COLOR': cylinder_color}
            cylinder_urdf = self.fill_template(cylinder_urdf, replace)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
            cylinders.append(cylinder_id)

        # Initialize lists to keep track of objects and target poses.

        cylinder_targs = []
        block_targs = []

        # Add target of cylinders and blocks in alternating layers.
        for i in range(n_cylinder):  # Three cylinders and two layers of blocks in between.
            cylinder_pose = (np.array([0.5, 0.2, 0.04 + i * 0.04 + (i - 1) * 0.04]),
                             np.array([0, 0, 0, 1]))  # Adjust height for each layer.
            cylinder_targs.append(cylinder_pose)

            if i < n_block:  # Add a layer of blocks only between the cylinders.
                block_targs.append([])
                for j in range(3):  # Three blocks in each layer.
                    # Calculate block positions to form a triangle.
                    angle = 2 * np.pi / 3 * j
                    offset = np.array([np.sin(angle) * 0.04, np.cos(angle) * 0.04,
                                       0.06 + i * 0.04])  # Adjust height and radial position.
                    block_pose = (
                    anchor_pose[0] + offset, utils.eulerXYZ_to_quatXYZW((0, 0, angle)))
                    block_targs[i].append(block_pose)

        env.add_object('zone/zone.urdf',
                       ((cylinder_pose[0][0], cylinder_pose[0][1], 0.01), (0, 0, 0, 1)), 'fixed',
                       scale=2, color=utils.get_random_color[1][0])

        # Define goals for each object to be in its target pose.
        for i in range(n_cylinder):
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)),
                          targ_poses=[cylinder_targs[i]], replace=False,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / n_cylinder / 2,
                          symmetries=[0], language_goal=f'set the cylinder layer {i}')
            if i < n_block:
                self.add_goal(objs=blocks[i], matches=np.ones((3, 3)), targ_poses=block_targs[i],
                              replace=False,
                              rotations=True, metric='pose', params=None,
                              step_max_reward=1 / n_block / 2,
                              language_goal=f'set blocks layer {i}')
