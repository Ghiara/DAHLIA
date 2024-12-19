import numpy as np
import os
import pybullet as p
import random

import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula


class AligningRopesToDifferentColorSquare(Task):
    """rearrange the ropes such that it connects the two endpoints of a 3-sided square of different color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "rearrange the {color_name1} rope to connect the two endpoints of the {color_name2} 3-sided square"
        self.task_completed_desc = "done manipulating two ropes."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        n_parts = 20
        radius = 0.005
        length = 2 * radius * n_parts * np.sqrt(2)

        # Add 3-sided square for the rope.
        _, color_list = utils.get_colors(n_colors=3)
        obj = []
        targ = []
        match = []
        for color_name in color_list:
            square_size = np.array([length, length, 0])
            square_pose = self.get_random_pose(env, square_size / 2)
            square_template = 'square/square-template.urdf'

            # IMPORTANT: REPLACE THE TEMPLATE URDF  with `fill_template`
            replace = {'DIM': (length,), 'HALF': (np.float32(length) / 2 - 0.005,)}
            urdf = self.fill_template(square_template, replace)
            env.add_object(urdf, square_pose, 'fixed', color=utils.COLORS[color_name])

            # compute corners
            corner0 = (length / 2, length / 2, 0.001)
            corner1 = (-length / 2, length / 2, 0.001)
            corner_0 = utils.apply(square_pose, corner0)
            corner_1 = utils.apply(square_pose, corner1)

            objects, targets, matches = self.make_ropes(env, corners=(corner_0, corner_1),
                                                        color_name=color_name)
            obj.append((objects, color_name))
            targ.append(targets)
            match.append(matches)

        k = random.randint(1, len(obj) - 1)
        shuffled_obj = self.cyclic_permutation(obj, k)

        for i in range(len(color_list)):
            self.add_goal(objs=shuffled_obj[i][0], matches=match[i], targ_poses=targ[i],
                          replace=False,
                          rotations=False, metric='pose', params=None,
                          step_max_reward=1 / len(color_list),
                          language_goal=self.lang_template.format(color_name1=shuffled_obj[i][1],
                                                                  color_name2=color_list[i]))

        for i in range(480):
            p.stepSimulation()

    def cyclic_permutation(self, lst, k):
        return lst[k:] + lst[:k]
