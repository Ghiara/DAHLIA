import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class BuildConcentricCircles(Task):
    """Construct two distinct circles on the tabletop in the zone using red and blue blocks.
    Each circle should consist of blocks of the same color, with the blue circle larger and surrounding the red circle."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "construct two concentric circles in the zone using 6 red and 10 blue blocks in the zone"
        self.task_completed_desc = "done building two circles."
        self.obj = {}
        self.goal = """Rearrange the blocks to construct two concentric circles in the zone using {num_red} red and {num_blue} blue blocks. Each circle should consist of blocks of the same color, with the blue circle larger and surrounding the red circle."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        num_inner = np.random.randint(4, 8)
        num_outer = num_inner + 4

        block_size = (0.04, 0.04, 0.04)
        # Add 6 red blocks.
        red_blocks = []
        for _ in range(num_inner):
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object('block/block.urdf', block_pose, color='violet')
            self.obj[block_id] = {'color': 'violet', 'shape': 'block'}
            red_blocks.append(block_id)
        radius = 0.1
        center = (0.45, 0.22, block_size[2] / 2)
        angles = np.linspace(0, 2 * np.pi, len(red_blocks), endpoint=False)
        zone_color = utils.get_random_color()[1][0]
        zone_id = env.add_object('zone/zone.urdf', ((center[0], center[1], 0.01), (0, 0, 0, 1)),
                                 'fixed',
                                 scale=3, color=zone_color)
        self.obj[zone_id] = {'catagory': 'fixed', 'scale': '3x scaled', 'color': zone_color,
                             'shape': 'zone'}
        # Define initial and target poses for the red circles.
        red_targ_poses = [
            ((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
             utils.eulerXYZ_to_quatXYZW((0, 0, angle))) for angle in angles]

        # Add 10 blue blocks.
        blue_blocks = []
        for _ in range(num_outer):
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object('block/block.urdf', block_pose, color='brown')
            self.obj[block_id] = {'color': 'brown', 'shape': 'block'}
            blue_blocks.append(block_id)
        radius = 0.2
        angles = np.linspace(0, 2 * np.pi, len(blue_blocks), endpoint=False)

        # Define initial and target poses for the blue circles.
        blue_targ_poses = [
            ((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
             utils.eulerXYZ_to_quatXYZW((0, 0, angle))) for angle in angles]

        self.goal = self.goal.format(num_red=num_inner, num_blue=num_outer)

        # Goal: each red block is in the red circle, each blue block is in the blue circle.
        self.add_goal(objs=red_blocks, matches=np.ones((num_inner, num_inner)),
                      targ_poses=red_targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2,
                      language_goal='build red circle')
        self.add_goal(objs=blue_blocks, matches=np.ones((num_outer, num_outer)),
                      targ_poses=blue_targ_poses, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1 / 2,
                      language_goal='build blue circle')
