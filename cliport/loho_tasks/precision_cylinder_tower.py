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


class PrecisionCylinderTower(Task):
    """
    Task to stack three cylinders of different colors (red, blue, green) on top of a small block
    to form a tower, ensuring the red cylinder is at the bottom, followed by blue, and green on the top,
    then place this tower on a pallet while avoiding contact with a zone marked on the tabletop.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the {color} cylinder on top of the {base_color} one"
        self.task_completed_desc = "completed the precision cylinder tower."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the small block as the base of the tower.
        base_size = (0.1, 0.1, 0.02)  # x, y, z dimensions
        base_urdf = 'block/block.urdf'
        base_pose = ((0.5, 0, base_size[2] / 2), (0, 0, 0, 1))  # Fixed position and orientation
        base_id = env.add_object(base_urdf, base_pose, 'fixed', scale=3)

        # Define cylinder size and colors.
        cylinder_size = (0.05, 0.05, 0.1)  # x, y, z dimensions
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        colors = ['red', 'blue', 'green']

        # Initialize list to store cylinder IDs.
        cylinders = []

        # Add cylinders in the environment with specified colors.
        for color in colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
            p.changeVisualShape(cylinder_id, -1, rgbaColor=utils.COLORS[color] + [1])
            cylinders.append(cylinder_id)

        # Define target positions for stacking cylinders on the base.
        targ_poses = [
            ((0.5, 0, base_size[2] + cylinder_size[2] / 2), (0, 0, 0, 1)),  # Red
            ((0.5, 0, base_size[2] + 1.5 * cylinder_size[2]), (0, 0, 0, 1)),  # Blue
            ((0.5, 0, base_size[2] + 2.5 * cylinder_size[2]), (0, 0, 0, 1))  # Green
        ]

        # Add goals for each cylinder to be placed in the correct sequence.
        for i, color in enumerate(colors[::-1]):  # Reverse to start with red at the bottom
            language_goal = self.lang_template.format(color=color,
                                                      base_color='small block' if i == 0 else
                                                      colors[2 - i])
            self.add_goal(objs=[cylinders[2 - i]], matches=np.ones((1, 1)),
                          targ_poses=[targ_poses[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 3,
                          symmetries=[0], language_goal=language_goal)

        # Add a pallet to the environment and define its pose.
        pallet_size = (0.3, 0.3, 0.02)  # x, y, z dimensions
        pallet_urdf = 'pallet/pallet.urdf'
        pallet_pose = ((0.4, 0.65, pallet_size[2] / 2), (0, 0, 0, 1))  # Position on the right side
        env.add_object(pallet_urdf, pallet_pose, 'fixed', scale=1)

        # Add a zone to avoid.
        zone_size = (0.2, 0.2, 0.01)  # x, y, z dimensions
        zone_urdf = 'zone/zone.urdf'
        zone_pose = ((0.35, -0.35, zone_size[2] / 2), (0, 0, 0, 1))  # Position on the left side
        env.add_object(zone_urdf, zone_pose, 'fixed')
        zone_pose = ((0.35, 0.35, zone_size[2] / 2), (0, 0, 0, 1))  # Position on the left side
        env.add_object(zone_urdf, zone_pose, 'fixed')
        zone_pose = ((0.7, -0.35, zone_size[2] / 2), (0, 0, 0, 1))  # Position on the left side
        env.add_object(zone_urdf, zone_pose, 'fixed')
        zone_pose = ((0.7, 0.35, zone_size[2] / 2), (0, 0, 0, 1))  # Position on the left side
        env.add_object(zone_urdf, zone_pose, 'fixed')

        # Define the final goal to place the tower on the pallet while avoiding the zone.
        tower_final_pose = (
        (0.65, 0, base_size[2] + 2.5 * cylinder_size[2] + pallet_size[2] / 2), (0, 0, 0, 1))
        self.add_goal(objs=cylinders + [base_id], matches=np.ones((1, 1)),
                      targ_poses=[tower_final_pose], replace=False,
                      rotations=False, metric='pose', params=None, step_max_reward=1,
                      symmetries=[0],
                      language_goal="place the tower on the pallet while avoiding the zone")
