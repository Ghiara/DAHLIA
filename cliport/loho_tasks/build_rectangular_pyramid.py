import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class BuildRectangularPyramidWithSameColorBlock(Task):
    """
    Task to construct a cube structure using six blocks of the same color.
    The blocks are first identified and gathered from around the table,
    then accurately stacked to form a cube without any external support.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "build a rectangular pyramid with {color} blocks in the zone"
        self.task_completed_desc = "done building cube with same color blocks."
        self.obj = {}
        self.goal = """Construct a 9-4-1 rectangular pyramid structure in the zone using 14 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block size and color
        block_size = (0.04, 0.04, 0.04)  # Size of each block
        color_rgb, color_name = utils.get_random_color()  # Get a random color for the blocks
        color_rgb = color_rgb[0]
        color_name = color_name[0]

        # Add six blocks of the same color
        block_urdf = 'block/block.urdf'
        blocks = []
        for _ in range(14):
            block_pose = self.get_random_pose(env, block_size,
                                              bound=np.array([[0.3, 0.6], [-0.5, -0.3], [0, 0.3]]))
            block_id = env.add_object(block_urdf, block_pose, color=color_name)
            blocks.append(block_id)
            self.obj[block_id] = {'color': color_name, 'shape': 'block'}

        # Define target positions for the cube structure
        # Assuming the center of the table is at [0.5, 0, 0] and blocks are 0.04m in size
        base_height = 0.02  # Half of block's height
        gap = 0.05
        positions = [
            (0.35, -0.05, base_height), (0.35 + gap, -0.05, base_height),
            (0.35 + 2 * gap, -0.05, base_height),  # Bottom layer
            (0.35, -0.05 + gap, base_height), (0.35 + gap, -0.05 + gap, base_height),
            (0.35 + 2 * gap, -0.05 + gap, base_height),
            (0.35, -0.05 + 2 * gap, base_height), (0.35 + gap, -0.05 + 2 * gap, base_height),
            (0.35 + 2 * gap, -0.05 + 2 * gap, base_height),
            (0.35 + 0.5 * gap, -0.05 + 0.5 * gap, base_height + 0.04),
            (0.35 + 1.5 * gap, -0.05 + 0.5 * gap, base_height + 0.04),  # Middle layer
            (0.35 + 0.5 * gap, -0.05 + 1.5 * gap, base_height + 0.04),
            (0.35 + 1.5 * gap, -0.05 + 1.5 * gap, base_height + 0.04),
            (0.35 + gap, -0.05 + gap, base_height + 0.08)  # Top layer
        ]
        targ_poses = [(pos, (0, 0, 0, 1)) for pos in positions]  # No rotation

        zone = env.add_object('zone/zone.urdf',
                              ((0.35 + gap, -0.05 + gap, base_height), (0, 0, 0, 1)), 'fixed',
                              scale=2, color=utils.get_random_color()[1][0])
        self.obj[zone] = {'catagory': 'fixed', 'scale': '2x scaled', 'shape': 'zone'}

        # Add goals for each block to be in the correct position
        language_goal = self.lang_template.format(color=color_name)
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i]], matches=np.eye(1), targ_poses=[targ_poses[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(blocks), language_goal=language_goal)
