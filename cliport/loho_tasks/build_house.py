import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class BuildHouse(Task):
    """Construct a house structure using blocks and a cylinder."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "Construct a house structure using blocks and a cylinder in the zone."
        "Begin by forming the base of the house with four red blocks arranged in a square shape."
        "Then build the walls by stacking two blue blocks on top of each base block."
        "Create a roof by placing two yellow blocks on the uppermost blue blocks, angled to form an apex."
        "Finally, position a green cylinder in the center of the square created by the base blocks to represent a chimney."
        self.task_completed_desc = "done building house."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add blocks for the base.
        base_blocks = []
        block_size = (0.04, 0.04, 0.04)  # x, y, z dimensions for the block size
        block_urdf = 'box/box-template.urdf'
        for _ in range(4):
            block_pose = self.get_random_pose(env, block_size, bound=np.array(
                [[0.25, 0.75], [-0.1, 0.5], [0, 0.3]]))
            base_block_urdf = self.fill_template(block_urdf, {'DIM': (0.06, 0.06, 0.04)})

            block_id = env.add_object(base_block_urdf, block_pose, color=utils.COLORS['red'])
            base_blocks.append(block_id)

        # Add blocks for the walls.
        wall_blocks = []
        for _ in range(4):
            block_pose = self.get_random_pose(env, block_size, bound=np.array(
                [[0.25, 0.75], [-0.1, 0.5], [0, 0.3]]))
            wall_block_urdf = self.fill_template(block_urdf, {'DIM': (0.04, 0.04, 0.04)})

            block_id = env.add_object(wall_block_urdf, block_pose, color=utils.COLORS['blue'])
            wall_blocks.append(block_id)

        # Add blocks for the roof.
        roof_blocks = []
        for _ in range(4):
            block_pose = self.get_random_pose(env, block_size, bound=np.array(
                [[0.25, 0.75], [-0.2, 0.5], [0, 0.3]]))
            if block_pose[0][1] > 0.5:
                pos = block_pose[0]
                rot = block_pose[1]
                pos = list(pos)
                pos[1] = -0.2
                pos = tuple(pos)
                block_pose = (pos, rot)
            roof_block_urdf = self.fill_template(block_urdf, {'DIM': (0.04, 0.14, 0.04)})

            block_id = env.add_object(roof_block_urdf, block_pose, color=utils.COLORS['yellow'])
            roof_blocks.append(block_id)

        # Add cylinder for the chimney.
        cylinder_template = 'cylinder/cylinder-template.urdf'
        cylinder_size = (0.03, 0.03, 0.05)
        replace = {'DIM': cylinder_size}  # radius and height dimensions for the cylinder size
        cylinder_urdf = self.fill_template(cylinder_template, replace)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        chimney_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS['green'])

        # Define the target poses for the base, walls, roof, and chimney.
        base_target_poses = [(0.7, -0.3, 0.02), (0.7, -0.2, 0.02), (0.6, -0.3, 0.02),
                             (0.6, -0.2, 0.02)]
        base_target_poses = [(pos, (0, 0, 0, 1)) for pos in base_target_poses]
        wall_target_poses = [(0.7, -0.3, 0.06), (0.7, -0.2, 0.06), (0.6, -0.3, 0.06),
                             (0.6, -0.2, 0.06)]
        wall_target_poses = [(pos, (0, 0, 0, 1)) for pos in wall_target_poses]
        ro = utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2))
        roof_target_poses = [((0.7, -0.25, 0.1), (0, 0, 0, 1)), ((0.6, -0.25, 0.1), (0, 0, 0, 1)),
                             ((0.65, -0.3, 0.14), ro), ((0.65, -0.2, 0.14), ro)]
        chimney_target_pose = [((0.65, -0.2, 0.19), (0, 0, 0, 1))]
        self.add_corner_anchor_for_pose(env, base_target_poses[0])

        env.add_object('zone/zone.urdf', ((0.65, -0.25, 0.01), (0, 0, 0, 1)), 'fixed',
                       scale=2, color=utils.get_random_color()[1][0])

        # Add goals for each step of the house construction.
        # Break the language prompt step-by-step
        self.add_goal(objs=base_blocks, matches=np.ones((4, 4)), targ_poses=base_target_poses,
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                      language_goal="Construct a house structure using blocks and a cylinder. Begin by forming the base of the house with four red blocks arranged in a square shape.")

        self.add_goal(objs=wall_blocks, matches=np.ones((4, 4)), targ_poses=wall_target_poses,
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                      language_goal="Then build the walls by stacking two blue blocks on top of each base block. ")

        self.add_goal(objs=roof_blocks[:2], matches=np.ones((2, 2)),
                      targ_poses=roof_target_poses[:2], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                      language_goal="Create a roof by placing two yellow blocks on the uppermost blue blocks, angled to form an apex. ")

        self.add_goal(objs=roof_blocks[2:], matches=np.ones((2, 2)),
                      targ_poses=roof_target_poses[2:], replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                      language_goal="Create a roof by placing two yellow blocks on the uppermost blue blocks, angled to form an apex. ")

        self.add_goal(objs=[chimney_id], matches=np.ones((1, 1)), targ_poses=chimney_target_pose,
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 5,
                      language_goal="Finally, position a green cylinder in the center of the square created by the base blocks to represent a chimney.")
