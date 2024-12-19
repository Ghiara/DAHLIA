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


class ColorCoordinatedBoxPyramid(Task):
    """Stack boxes of different colors to form a color-coordinated pyramid."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack boxes of different colors to form a colorful tower on the pallet"
        self.task_completed_desc = "done stacking color-coordinated box pyramid."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define box colors and order.
        colors = ['red', 'blue', 'green', 'yellow']
        order = [0, 1, 2, 3]

        # Shuffle the order of colors.
        np.random.shuffle(order)

        # Add pallet.
        pallet_size = (0.2, 0.2, 0.02)
        pallet_pose = self.get_random_pose(env, pallet_size)
        pallet_urdf = 'pallet/pallet.urdf'
        env.add_object(pallet_urdf, pallet_pose, category='fixed')

        # Add boxes.
        box_size = (0.05, 0.05, 0.05)
        box_urdf = 'box/box-template.urdf'
        boxes = []
        for i in range(len(colors)):
            color = colors[order[i]]
            box_pose = self.get_random_pose(env, box_size)
            box_id = env.add_object(box_urdf, box_pose, color=color)
            boxes.append(box_id)

        # Add stacking goals.
        for i in range(len(colors)):
            # color = colors[order[i]]
            # prev_color = colors[order[i-1]]
            language_goal = self.lang_template
            self.add_goal(objs=[boxes[i]], matches=np.ones((1, 1)), targ_poses=[pallet_pose],
                          replace=False,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(colors),
                          language_goal=language_goal)
