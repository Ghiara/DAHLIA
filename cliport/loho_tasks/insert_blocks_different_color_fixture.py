import numpy as np
import os
import pybullet as p
import random
from cliport.tasks.task import Task
from cliport.utils import utils


class InsertBlocksToDifferentColorFixture(Task):
    """Pick up blocks and insert them into the different colored fixtures."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "insert the block with different colors into the {color} fixture"
        self.task_completed_desc = "done inserting blocks."
        self.obj = {}
        self.goal = """Each L-shaped fixture can hold three blocks, suppose the block size is (a,a,a), then in fixture's local coordinate system, the three places that can hold blocks are [(0,0,0),(a,0,0),(0,a,0)]. Fill in all the fixtures which have random position and rotation with blocks, and make sure in the end in every fixture there are three blocks with different colors"""

    def reset(self, env):
        super().reset(env)

        # Define colors for blocks and fixtures
        colors = ['red', 'blue', 'silver', 'gold']

        # Add fixtures.
        fixture_size = (0.04, 0.04, 0.04)
        fixture_urdf = 'insertion/fixture.urdf'
        fixture_poses = []
        for i in range(len(colors)):
            fixture_pose = self.get_random_pose(env, fixture_size)
            fixture_id = env.add_object(fixture_urdf, fixture_pose, color=colors[i],
                                        category='fixed', scale=1.02)
            fixture_set_poses = [fixture_pose,
                                 (utils.apply(fixture_pose, (0.04, 0, 0)), fixture_pose[1]),
                                 (utils.apply(fixture_pose, (0, 0.04, 0)), fixture_pose[1])]
            fixture_poses.append(fixture_set_poses)
            self.obj[fixture_id] = {'catagory': 'fixed', 'color': colors[i], 'shape': 'fixture'}

        # Add blocks.
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        blocks = []
        for k in range(len(colors)):
            blocks.append([])
            for i in range(3):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=utils.COLORS[colors[k]])
                blocks[k].append(block_id)
                self.obj[block_id] = {'color': colors[k], 'shape': 'block'}

        # Goal: each block is in the corresponding color fixture.
        for i, color in enumerate(colors):
            temp_blocks = []
            for j in range(len(colors)):
                if j != i:
                    temp_blocks.append(blocks[j][-1])
                    blocks[j] = blocks[j][:-1]
            temp_poses = fixture_poses[i].copy()
            random.shuffle(temp_poses)
            self.add_goal(objs=temp_blocks, matches=np.ones((3, 3)), targ_poses=temp_poses,
                          replace=False,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(colors),
                          language_goal=self.lang_template.format(color=color))
