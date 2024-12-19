import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ConstructCircleWithBallInMiddle(Task):
    """Stack a set of small red blocks and small blue blocks in a circular shape around the ball."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "stack the {color} blocks in a circular shape around the ball"
        self.task_completed_desc = "done constructing circle with blocks."
        self.obj = {}
        self.goal = """Construct a circle with alternating red and blue blocks around the ball. The circle should be built by one red block, then one blue block and then again one red block and so on, alternatively"""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define block colors and sizes.
        colors = ['pink', 'purple']
        block_size = (0.04, 0.04, 0.04)

        # Add blocks.
        blocks = []
        for i in range(8):
            color = colors[i // 4]
            block_pose = self.get_random_pose(env, obj_size=block_size)
            block_id = env.add_object('block/block.urdf', block_pose, color=color)
            blocks.append(block_id)
            self.obj[block_id] = {'color': color, 'shape': 'block'}

        # Calculate target poses for circular shape.
        center = (0.43, -0.32, 0.02)
        radius = 0.15
        angles = np.linspace(0, 2 * np.pi, len(blocks), endpoint=False)
        targ_poses = [
            ((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), center[2]),
             utils.eulerXYZ_to_quatXYZW((0, 0, angle))) for angle in angles]

        ball_size = (0.04, 0.04, 0.04)
        ball_pose = (center, (0, 0, 0, 1))
        ball_id = env.add_object('ball/ball.urdf', ball_pose, 'fixed', color='gold')
        self.obj[ball_id] = {'catagory': 'fixed', 'color': 'gold', 'shape': 'ball'}

        # Create language and motion goals for each block.
        '''
        for i, _ in enumerate(blocks):
            color = colors[i // 2 % 2]
            language_goal = self.lang_template.format(color=color)
            self.add_goal(objs=[blocks[i]], matches=np.ones((1, 1)), targ_poses=[targ_poses[i]], 
                          replace=False,rotations=True, metric='pose', params=None, step_max_reward=1 / len(blocks),
                          language_goal=language_goal)
        '''
        targ_0 = []
        targ_1 = []
        for i in range(len(targ_poses)):
            if i % 2 == 0:
                targ_0.append(targ_poses[i])
            else:
                targ_1.append(targ_poses[i])

        self.add_goal(objs=blocks[:len(blocks) // 2],
                      matches=np.ones((len(blocks) // 2, len(blocks) // 2)),
                      targ_poses=targ_0, replace=False, rotations=True, metric='pose', params=None,
                      step_max_reward=1 / 2,
                      language_goal=self.lang_template.format(color=colors[0]))

        self.add_goal(objs=blocks[len(blocks) // 2:],
                      matches=np.ones((len(blocks) // 2, len(blocks) // 2)),
                      targ_poses=targ_1, replace=False, rotations=True, metric='pose', params=None,
                      step_max_reward=1 / 2,
                      language_goal=self.lang_template.format(color=colors[1]))
