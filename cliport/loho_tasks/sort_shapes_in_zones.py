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


class SortShapesInZones(Task):
    """Pick up blocks of different shapes and place them into separate zones of matching shape."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_shapes = 0
        self.lang_template = "pick up the {shape} block and place it in the {shape} zone"
        self.task_completed_desc = "done sorting shapes in zones"
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""

    def reset(self, env):
        super().reset(env)

        self.n_shapes = np.random.randint(2, 5)
        all_shape_names = ['circle', 'square', 'triangle', 'pentagon', 'hexagon']
        selected_shape_names = np.random.choice(all_shape_names, self.n_shapes, replace=False)
        shapes = [shape for shape in selected_shape_names]

        self.max_steps = self.n_shapes + 2

        # Add zones
        zone_size = (0.1, 0.1, 0)
        zone_poses = []
        for i in range(self.n_shapes):
            zone_pose = self.get_random_pose(env, obj_size=zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            while not zone_obj_id:
                zone_size = (zone_size[0] - 0.01, zone_size[1] - 0.01, 0)
                zone_pose = self.get_random_pose(env, obj_size=zone_size)
                zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=utils.COLORS[shapes[i]] + [1])
            zone_poses.append(zone_pose)

        # Add blocks
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        for i in range(self.n_shapes):
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[shapes[i]] + [1])
            blocks.append(block_id)

        # Goal: each block is in a different zone of matching shape.
        self.add_goal(objs=blocks, matches=np.eye(self.n_shapes), targ_poses=zone_poses,
                      replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1,
                      language_goal=self.lang_template)

        self.scene_description = f"On the table, there are {self.n_shapes} blocks. Their shapes are {', '.join(shapes)}. " \
                                 f"There are {self.n_shapes} zones. Each zone matches the shape of a block."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
