import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ConstructCircleWithBlocks(Task):
    """Stack a set of small red blocks and small blue blocks in a circular shape on the tabletop."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "stack the {color} blocks in a circular shape in the zone"
        self.task_completed_desc = "done constructing circle with blocks."
        self.obj = {}
        self.goal = """Construct a circle with alternating red and blue blocks in the zone. To be concrete, the circle should be built by one red block, then one blue block and then again one red block and so on."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors and sizes.
        colors = ['red', 'blue']
        block_size = (0.04, 0.04, 0.04)
        num_block = 5

        # Add blocks.
        blocks = []
        for color in colors:
            for _ in range(num_block):
                block_pose = self.get_random_pose(env, obj_size=block_size)
                block_id = env.add_object('block/small.urdf', block_pose, color=color, scale=2)
                blocks.append(block_id)
                self.obj[block_id] = {'color': color, 'shape': 'block'}

        # Calculate target poses for circular shape.
        center = (0.55, 0.0, 0.02)
        radius = 0.1
        angles = np.linspace(0, 2 * np.pi, len(blocks), endpoint=False)
        targ_poses = [
            ((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
             utils.eulerXYZ_to_quatXYZW((0, 0, angle))) for angle in angles]

        zone = env.add_object('zone/zone.urdf', ((center[0], center[1], 0.001), (0, 0, 0, 1)),
                              'fixed',
                              scale=2.2, color=utils.get_random_color()[1][0])
        self.obj[zone] = {'catagory': 'fixed', 'scale': '2.2x scaled', 'color': color,
                          'shape': 'zone'}

        targ_0 = []
        targ_1 = []
        for i in range(len(targ_poses)):
            if i % 2 == 0:
                targ_0.append(targ_poses[i])
            else:
                targ_1.append(targ_poses[i])

        # Create language and motion goals for each block.
        color = colors[0]
        language_goal = self.lang_template.format(color=color)
        self.add_goal(objs=blocks[:len(blocks) // 2],
                      matches=np.ones((len(blocks) // 2, len(blocks) // 2)),
                      targ_poses=targ_0, replace=False, rotations=True, metric='pose',
                      params=None, step_max_reward=1 / 2, language_goal=language_goal)

        color = colors[1]
        language_goal = self.lang_template.format(color=color)
        self.add_goal(objs=blocks[len(blocks) // 2:],
                      matches=np.ones((len(blocks) // 2, len(blocks) // 2)),
                      targ_poses=targ_1, replace=False, rotations=True, metric='pose',
                      params=None, step_max_reward=1 / 2, language_goal=language_goal)
