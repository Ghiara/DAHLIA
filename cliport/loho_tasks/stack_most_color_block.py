import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class StackMostColorBlock(Task):
    """Task to stack one color of blocks that has the largest quantity."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack one color of blocks that has the largest quantity in the zone"
        self.task_completed_desc = "completed stacking one color of blocks that has the largest quantity."
        self.obj = {}
        self.goal = """choose one color of blocks that has the largest quantity and stack them in the zone."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the color for all blocks.
        n_blocks = np.random.randint(10, 16)
        n_colors = np.random.randint(3, 6)
        color, color_name = utils.get_colors(n_colors=n_colors)

        # Define block size.
        block_size = (0.04, 0.04, 0.04)  # Uniform size for all blocks.

        # Load the block URDF.
        block_urdf = 'block/block.urdf'
        blocks = []

        # Add blocks to the environment and set their colors.
        for _ in range(n_blocks):
            pose = self.get_random_pose(env, block_size)  # No rotation needed, quaternion format.
            block_id = env.add_object(block_urdf, pose)
            blocks.append(block_id)

        block_set = self.partition_array(blocks, n_colors)
        for i, set in enumerate(block_set):
            for id in set:
                p.changeVisualShape(id, -1, rgbaColor=color[i] + [1])
                self.obj[id] = {'color': color_name[i], 'shape': 'block'}

        # block_group={color_name[i]:block_set[i] for i in range(n_colors)}

        # Define target positions for the blocks to form the letter R.
        # Note: In a real implementation, these would be carefully calculated to form the letter R.
        # Here, we use the same positions as the initial ones for simplicity.
        zone_pose = self.get_random_pose(env, (0.1, 0.1, 0.01))
        zone = env.add_object('zone/zone.urdf', zone_pose, 'fixed',
                              scale=1, color=utils.get_random_color()[1][0])
        length = len(block_set[-1])
        self.obj[zone] = {'catagory': 'fixed', 'shape': 'zone'}

        targ_poses = [((zone_pose[0][0], zone_pose[0][1],
                        zone_pose[0][2] + (i + 0.5) * block_size[2]), zone_pose[1])
                      for i in range(length)]

        # Add goal for each block placement with detailed language instructions.
        self.add_goal(objs=block_set[-1], matches=np.ones((length, length)),
                      targ_poses=targ_poses, replace=False, rotations=True, metric='pose',
                      params=None,
                      symmetries=[np.pi / 2], step_max_reward=1, language_goal=self.lang_template)
        '''
        for i in range(length):
            self.add_goal(objs=[block_set[-1][i]], matches=np.ones((1,1)), 
                      targ_poses=[targ_poses[i]], replace=False, rotations=True, metric='pose', params=None,
                      symmetries=[np.pi/2], step_max_reward=1/length, language_goal=self.lang_template)
        '''

    def partition_array(self, a, m):
        if m <= 0:
            return "m should > 0"
        if len(a) // 2 < m:
            return "m should <= len(array)//2"

        partitions = []
        max_length = len(a) // 2
        avg_length = (len(a) - max_length) // (m - 1)

        for i in range(m - 1):
            current_partition = []
            length = avg_length + (1 if i < (len(a) - max_length) % (m - 1) else 0)
            start_index = sum(
                [avg_length + (1 if j < (len(a) - max_length) % (m - 1) else 0) for j in range(i)])

            current_partition.extend(a[start_index:start_index + length])
            partitions.append(current_partition)

        partitions.append(a[-max_length:])

        return partitions
