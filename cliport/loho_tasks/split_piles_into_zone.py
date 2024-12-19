import numpy as np
import os
import pybullet as p
import random
from cliport.tasks import primitives
from cliport.tasks.grippers import Spatula
from cliport.tasks.task import Task
from cliport.utils import utils


class SplitPilesIntoZone(Task):
    """Split piles of small objects into a several target goal zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "split and push each pile of blocks into zone"
        self.task_completed_desc = "done sweeping."
        self.primitive = primitives.push
        self.ee = Spatula
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        num_blocks = 60
        colors = ['green']
        target_poses = []
        # Add the target zone and piles
        urdf = 'zone/zone.urdf'
        zone_size = (0.2, 0.2, 0.01)
        zone_count = 3

        for color in colors:
            for i in range(zone_count):
                zone_pose = self.get_random_pose(env, zone_size)
                zone_id = env.add_object(urdf, zone_pose, 'fixed', color=color)

                # Sample point from the zone as the target poses for the piles
                target_poses.append(zone_pose)

        # Add pile of small blocks with `make_piles` function
        piles = self.make_piles(env, block_color=colors, count=len(colors), num_blocks=num_blocks)

        # Add goal
        obj = self.devide(piles, zone_count)
        for i in range(len(colors)):
            for j in range(zone_count):
                objs = obj[j]
                self.add_goal(objs=objs, matches=np.ones((len(objs), 1)),
                              targ_poses=[target_poses[j]], replace=True,
                              rotations=False, metric='zone',
                              params=[(target_poses[j], zone_size)],
                              step_max_reward=1 / zone_count,
                              language_goal=self.lang_template)

    def devide(self, a, x):
        l = len(a)
        if l % x == 0:
            p = [a[i * (l // x):(i + 1) * (l // x)] for i in range(x)]
        else:
            p = [a[i * (l // x):(i + 1) * (l // x)] for i in range(x)]
            for i, e in enumerate(a[-(l % x):]):
                p[i].append(e)
            index = 0
            for sublist in p:
                for i in range(len(sublist)):
                    sublist[i] = a[index]
                    index += 1
        return p
