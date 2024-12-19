import os

import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class AlignRopeAroundShape(Task):
    """Align a deformable rope around a shape on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "align the rope around the shape"
        self.task_completed_desc = "done aligning."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add zone.
        color = utils.get_random_color()[0]
        rand_shape = self.get_kitting_shapes(n_objects=1)[0]
        shape = os.path.join(self.assets_root, 'kitting', f'{rand_shape:02d}.obj')
        shape_pose = self.get_random_pose(env, (0.06, 0.06, 0.01))
        scale = [0.008, 0.008, 0.00001]
        replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': color}
        template = 'kitting/object-template-nocollision.urdf'
        urdf = self.fill_template(template, replace)
        shape_id = env.add_object(urdf, shape_pose, 'fixed')

        # Add rope.
        rope_size = (0.2, 0.01, 0.01)
        shape_size = p.getAABB(shape_id)[0]
        rope_pose = self.get_random_pose(env, rope_size)
        corner1_pose = utils.apply(shape_pose, (shape_size[0] / 2, shape_size[1] / 2, 0.01))
        corner2_pose = utils.apply(shape_pose, (-shape_size[0] / 2, -shape_size[1] / 2, 0.01))
        rope_id, targets, matches = self.make_rope(env, (corner1_pose, corner2_pose))

        # Goal: rope is aligned with the diagonal of the zone.
        self.add_goal(objs=rope_id, matches=matches, targ_poses=targets, replace=False,
                      rotations=False, metric='pose', params=None, step_max_reward=1,
                      language_goal=self.lang_template)

        # wait for the scene to settle down
        for i in range(480):
            p.stepSimulation()
