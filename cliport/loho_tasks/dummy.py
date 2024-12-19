import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks.grippers import Grip
import time

class SimpleStack(Task):
    """Dummy task for testing everything."""

    def __init__(self):
        super().__init__()
        self.max_steps = 5
        self.lang_template = "stack block in the zone"
        self.task_completed_desc = "completed stacking blocks in the zone."
        self.ee = Grip
        self.obj = {}
        self.goal = """stack all blocks in the zone."""

    def reset(self, env):
        super().reset(env)
        

        # Define the color for all blocks.
        n_blocks = 2

        # Define block size.
        block_size = (0.04, 0.04, 0.04)  # Uniform size for all blocks.

        # Load the block URDF.
        zone_urdf = 'zone/zone.urdf'
        block_urdf = 'block/block.urdf'
        blocks = []

        # Add blocks to the environment and set their colors.
        for _ in range(n_blocks):
            pose = self.get_random_pose(env, block_size)  # No rotation needed, quaternion format.
            block_id = env.add_object(block_urdf, pose, color=utils.get_random_color()[1][0])
            blocks.append(block_id)
            self.obj[block_id] = {'shape': 'block'}

        zone_pose = self.get_random_pose(env, (0.1, 0.1, 0.01))
        zone = env.add_object(zone_urdf, zone_pose, 'fixed',
                              scale=1, color=utils.get_random_color()[1][0])
        self.obj[zone] = {'catagory': 'fixed', 'shape': 'zone'}

        targ_poses = [((zone_pose[0][0],zone_pose[0][1],zone_pose[0][2]+i*0.04),zone_pose[1]) for i in range(n_blocks)]

        # Add goal for each block placement with detailed language instructions.
        self.add_goal(objs=blocks, matches=np.ones((n_blocks, n_blocks)),
                      targ_poses=targ_poses, replace=False, rotations=True, metric='pose',
                      params=None,
                      symmetries=[np.pi / 2], step_max_reward=1, language_goal=self.lang_template)
        