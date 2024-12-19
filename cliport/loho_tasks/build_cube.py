import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class BuildCubeWithSameColorBlock(Task):
    """
    Task to construct a cube structure using six blocks of the same color.
    The blocks are first identified and gathered from around the table,
    then accurately stacked to form a cube without any external support.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "build a cube with {color} blocks in the zone"
        self.task_completed_desc = "done building cube with same color blocks."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block size and color
        block_size = (0.04, 0.04, 0.04)  # Size of each block
        color_rgb, color_name = utils.get_random_color()  # Get a random color for the blocks
        color_rgb = color_rgb[0]
        color_name = color_name[0]

        zone_pose = self.get_random_pose(env, (0.1, 0.1, 0.01))

        # Add eight blocks of the same color
        block_urdf = 'block/block.urdf'
        blocks = []
        for _ in range(8):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=color_rgb + [1])
            blocks.append(block_id)
            self.obj[block_id] = {'color': color_name, 'shape': 'block'}

        # Define target positions for the cube structure
        # Assuming the center of the table is at [0.5, 0, 0] and blocks are 0.04m in size
        base_height = 0.02  # Half of block's height
        positions = [
            (-0.035, -0.035, base_height), (0.035, -0.035, base_height),  # Bottom layer
            (-0.035, 0.035, base_height), (0.035, 0.035, base_height),
            (-0.035, -0.035, base_height + 0.04), (0.035, -0.035, base_height + 0.04),  # Top layer
            (-0.035, 0.035, base_height + 0.04), (0.035, 0.035, base_height + 0.04)
        ]

        targ_poses = [(utils.apply(zone_pose, pos), zone_pose[1]) for pos in positions] 

        zone_color = utils.get_random_color()[1][0]

        zone_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed',
                                 scale=1, color=zone_color)
        self.obj[zone_id] = {'catagory': 'fixed', 'color': zone_color, 'shape': 'zone'}

        # Add goals for each block to be in the correct position
        language_goal = self.lang_template.format(color=color_name)
        self.add_goal(objs=blocks, matches=np.ones((8, 8)), targ_poses=targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1,
                      language_goal=language_goal)
