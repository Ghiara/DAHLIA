import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class BuildCircumcircle(Task):
    """Stack a set of small red blocks and small blue blocks in a circular shape on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the {color} blocks in a circular shape around the zone"
        self.task_completed_desc = "done constructing circumcircle ."
        self.obj = {}
        self.goal = """Construct a circumcircle of the zone using blocks of same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors and sizes.
        color = utils.get_random_color()[1]
        block_size = (0.04, 0.04, 0.04)
        num_block = np.random.randint(6,13)

        # Add blocks.
        blocks = []
        for _ in range(num_block):
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object('block/small.urdf', block_pose, color=color, scale=2)
            blocks.append(block_id)
            self.obj[block_id] = {'color': color, 'shape': 'block'}

        # Calculate target poses for circular shape.
        zone_pose = self.get_random_pose(env, (0.1,0.1,0.1))
        radius = np.sqrt(2) * 0.1
        angles = np.linspace(0, 2 * np.pi, len(blocks), endpoint=False)
        targ_poses = [
            ((zone_pose[0] + radius * np.cos(angle), zone_pose[1] + radius * np.sin(angle), zone_pose[2]),
             utils.eulerXYZ_to_quatXYZW((0, 0, angle))) for angle in angles]

        zone = env.add_object('zone/zone.urdf', zone_pose,
                              'fixed',
                              scale=2.2, color=utils.get_random_color()[1][0])
        self.obj[zone] = {'catagory': 'fixed', 'scale': '2.2x scaled', 'color': color,
                          'shape': 'zone'}

        # Create language and motion goals for each block.
        language_goal = self.lang_template.format(color=color)
        self.add_goal(objs=blocks,
                      matches=np.ones((len(blocks), len(blocks))),
                      targ_poses=targ_poses, replace=False, rotations=True, metric='pose',
                      params=None, step_max_reward=1, language_goal=language_goal)
