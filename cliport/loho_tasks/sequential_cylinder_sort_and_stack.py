import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
import os
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class SequentialCylinderSortAndStack(Task):
    """
    Sort six cylinders into two groups based on their sizes (three small and three large) and colors
    (red, blue, yellow for each size group), then stack each group on its corresponding colored pallet
    (red for small, blue for large) with a specific color order: red at the bottom, then yellow, and blue on top.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "sort and stack the cylinder on the pallet,"
        "with the smaller on the left and bigger on the right, with ascending color order of red-yellow-blue."
        self.task_completed_desc = "done sorting and stacking cylinders."
        self.obj = {}
        self.goal = """Sort six cylinders into two groups based on their sizes (small and large) and stack small group on red pallet and large group on blue pallet, both with a specific cylinder color order: red at the bottom, then yellow, then blue."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define cylinder sizes and colors
        sizes = {'small': (0.04, 0.04, 0.02), 'large': (0.06, 0.06, 0.02)}
        colors = ['red', 'yellow', 'blue']

        # Define pallet sizes and colors
        pallet_sizes = {'red': (0.3, 0.3, 0.02), 'blue': (0.3, 0.3, 0.02)}
        pallet_poses = {'red': ((0.5, -0.3, 0.01), (0, 0, 0, 1)),
                        'blue': ((0.5, 0.3, 0.01), (0, 0, 0, 1))}

        # Add pallets
        for color, size in pallet_sizes.items():
            pallet_urdf = 'pallet/pallet.urdf'
            pallet_id = env.add_object(pallet_urdf, pallet_poses[color], 'fixed')
            p.changeVisualShape(pallet_id, -1, rgbaColor=utils.COLORS[color] + [1])
            self.obj[pallet_id] = {'catagory': 'fixed', 'color': color, 'shape': 'pallet'}

        # Add cylinders and define their target positions
        cylinders = []
        target_positions = {'small': [], 'large': []}
        for size, dims in sizes.items():
            for color in colors:
                cylinder_urdf = self.fill_template('cylinder/cylinder-template.urdf',
                                                   {'DIM': dims})
                cylinder_pose = self.get_random_pose(env, dims)
                cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=color)
                cylinders.append((cylinder_id, size, color))

                # Define target positions for stacking
                if size == 'small':
                    target_positions['small'].append(
                        ((0.5, -0.3, 0.02 + 0.08 * colors.index(color)), (0, 0, 0, 1)))
                    self.obj[cylinder_id] = {'catagory': 'small', 'color': color,
                                             'shape': 'cylinder'}
                else:
                    target_positions['large'].append(
                        ((0.5, 0.3, 0.02 + 0.12 * colors.index(color)), (0, 0, 0, 1)))
                    self.obj[cylinder_id] = {'catagory': 'large', 'color': color,
                                             'shape': 'cylinder'}

        # Add goal steps for sorting and stacking cylinders
        for cylinder_id, size, color in cylinders:
            if size == 'small':
                targ_pose = target_positions['small'][colors.index(color)]
                color_p = 'red'
            else:
                targ_pose = target_positions['large'][colors.index(color)]
                color_p = 'blue'

            language_goal = self.lang_template.format(size=size, color_c=color, color_p=color_p)
            self.add_goal(objs=[(cylinder_id, (np.pi / 2, None))], matches=np.ones((1, 1)),
                          targ_poses=[targ_pose],
                          replace=False, rotations=True, metric='pose', params=None,
                          step_max_reward=1 / 6,
                          language_goal=language_goal)
