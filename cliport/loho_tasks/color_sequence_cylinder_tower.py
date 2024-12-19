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


class ColorSequenceCylinderTower(Task):
    """
    Task to construct a tower by stacking cylinders of three different colors
    (red, blue, green) on a square base in a specific sequence - blue at the bottom,
    followed by green, and finally red at the top, ensuring each cylinder is perfectly
    aligned with the one below.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the {color} cylinder"
        self.task_completed_desc = "done constructing the color sequence cylinder tower."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the base size and add the base to the environment.
        base_size = (0.15, 0.15, 0.01)  # x, y, z dimensions
        base_urdf = 'square/square-template.urdf'
        base_pose = ((0.5, 0, 0.005), (0, 0, 0, 1))  # Fixed position and orientation
        env.add_object(base_urdf, base_pose, 'fixed')

        # Define cylinder size and colors.
        cylinder_size = (0.05, 0.05, 0.1)  # x, y, z dimensions
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        colors = ['blue', 'green', 'red']

        # Initialize list to store cylinder IDs.
        cylinders = []

        # Add cylinders in the environment with specified colors.
        for color in colors:
            cylinder_pose = self.get_random_pose(env, cylinder_size)
            cylinder_id = env.add_object(cylinder_urdf, cylinder_pose)
            p.changeVisualShape(cylinder_id, -1, rgbaColor=utils.COLORS[color] + [1])
            cylinders.append(cylinder_id)

        # Define target positions for stacking cylinders on the base.
        # The z-coordinate is incremented by the height of the cylinders to stack them.
        targ_poses = [
            ((0.5, 0, base_size[2] + cylinder_size[2] / 2), (0, 0, 0, 1)),  # Blue
            ((0.5, 0, base_size[2] + 1.5 * cylinder_size[2]), (0, 0, 0, 1)),  # Green
            ((0.5, 0, base_size[2] + 2.5 * cylinder_size[2]), (0, 0, 0, 1))  # Red
        ]

        # Add goals for each cylinder to be placed in the correct sequence.
        for i, color in enumerate(colors):
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[cylinders[i]], matches=np.ones((1, 1)), targ_poses=[targ_poses[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 3,
                          symmetries=[0], language_goal=language_goal)
