import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class PushPilesIntoZone(Task):
    """Push piles of small objects into a target goal zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "push the each pile of blocks into corresponding zone"
        self.task_completed_desc = "done sweeping."
        self.primitive = primitives.push
        self.ee = Spatula
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        num_blocks = 30
        colors = ['red', 'blue']
        target_poses = []
        # Add the target zone and piles
        urdf = 'zone/zone.urdf'
        zone_size = (0.2, 0.2, 0.01)

        for color in colors:
            zone_pose = self.get_random_pose(env, zone_size)
            zone_id = env.add_object(urdf, zone_pose, 'fixed', color=color)

            # Sample point from the zone as the target poses for the piles
            target_poses.append(zone_pose)

        # Add pile of small blocks with `make_piles` function
        piles = self.make_piles(env, block_color=colors, count=len(colors), num_blocks=num_blocks)

        # Add goal
        for i in range(len(colors)):
            self.add_goal(objs=piles[i], matches=np.ones((num_blocks, 1)),
                          targ_poses=[target_poses[i]], replace=True,
                          rotations=False, metric='zone', params=[(target_poses[i], zone_size)],
                          step_max_reward=1 / len(colors), language_goal=self.lang_template)
