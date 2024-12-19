import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class RotateAndInsertCylinder(Task):
    """Pick up a red cylinder, rotate it horizontally, and insert it into a green fixture that is positioned upright on the table."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "pick up the red cylinder, rotate it horizontally, and insert it into the green fixture"
        self.task_completed_desc = "done rotating and inserting cylinder."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add fixture.
        # x, y, z dimensions for the asset size
        fixture_size = (0.1, 0.1, 0.1)
        fixture_pose = self.get_random_pose(env, fixture_size)
        fixture_urdf = 'insertion/fixture.urdf'
        env.add_object(fixture_urdf, fixture_pose, 'fixed', scale=2)

        # Add cylinder.
        # x, y, z dimensions for the asset size
        cylinder_size = (0.04, 0.04, 0.01)
        cylinder_pose = self.get_random_pose(env, cylinder_size)
        cylinder_urdf = 'cylinder/cylinder-template.urdf'
        cylinder_id = env.add_object(cylinder_urdf, cylinder_pose, color=utils.COLORS['white'],
                                     scale=2)
        targ = []
        dummy_pose = self.get_random_pose(env, cylinder_size)
        cylinder_pose = p.getBasePositionAndOrientation(cylinder_id)
        targ.append((np.array(cylinder_pose[0]) + np.array([0.2, 0, 0]),
                     utils.eulerXYZ_to_quatXYZW((0, np.pi / 2, 0))))
        targ.append(fixture_pose)

        # Goal: the cylinder is inserted into the fixture.
        self.add_goal(objs=[cylinder_id], matches=np.ones((1, 1)), targ_poses=[targ[0]],
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2,
                      language_goal=self.lang_template)
        self.add_goal(objs=[cylinder_id], matches=np.ones((1, 1)), targ_poses=[targ[1]],
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2,
                      language_goal=self.lang_template)
