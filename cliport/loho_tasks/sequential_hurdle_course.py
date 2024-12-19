import numpy as np
import os
import pybullet as p
from cliport.tasks.task import Task
from cliport.tasks.grippers import Spatula
from cliport.tasks import primitives
from cliport.utils import utils


class SequentialHurdleCourse(Task):
    """
    Navigate sequentially through a course by placing blocks to form a bridge and finally, 
    organizing balls by color into corresponding colored bowls placed beyond the bridge.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "navigate {color} ball through the course"
        self.task_completed_desc = "completed the sequential hurdle course."
        self.primitive = primitives.push
        self.ee = Spatula
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors for easy reference
        colors = ['red', 'white', 'yellow']

        # Stage 1: Place a bigger block to create a ball bank for the bridge
        # Define the bigger block
        bigger_block_template = 'box/box-template.urdf'
        bigger_block_size = (0.4, 0.2, 0.1)
        bigger_block_pose = ((0.5, -0.3, 0.05), (0, 0, 0, 1))
        replace = {'DIM': bigger_block_size, 'COLOR': utils.COLORS['indigo']}
        bigger_block_urdf = self.fill_template(bigger_block_template, replace)
        big = env.add_object(bigger_block_urdf, bigger_block_pose, 'fixed')

        # Stage 2: Place the bridge body and surface
        # Define the blocks for the bridge pillar
        base_pose = np.array([0.35, -0.18, 0.02])
        x_gap = np.array([0.15, 0, 0])
        y_gap = np.array([0, 0.2, 0])
        z_gap = np.array([0, 0, 0.04])
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    block_size = (0.04, 0.04, 0.04)
                    block_pose = base_pose + i * x_gap + j * y_gap + k * z_gap
                    block_pose = (block_pose, (0, 0, 0, 1))
                    block_urdf = self.fill_template(bigger_block_template, {'DIM': block_size})
                    env.add_object(block_urdf, block_pose, 'fixed', color=colors[i])

        body_base_pose = np.array([0.35, -0.08, 0.09])
        for i in range(3):
            body_size = (0.08, 0.24, 0.02)  # x, y, z dimensions for the asset size
            body_block_urdf = self.fill_template(bigger_block_template, {'DIM': body_size})
            body_block_pose = body_base_pose + i * x_gap
            body_block_pose = (body_block_pose, (0, 0, 0, 1))
            env.add_object(body_block_urdf, body_block_pose, 'fixed', color=colors[i])

        # Stage 3: Place balls by color and corresponding colored bowls
        ball_size = (0.02, 0.02, 0.02)
        bowl_size = np.array([0.1, 0.1, 0.1])
        balls = []
        targs = []
        for i, color in enumerate(colors):
            ball_pose = ((0.35 + i * 0.15, -0.3, 0.11), (0, 0, 0, 1))
            ball_template = 'box/box-template.urdf'
            ball_urdf = self.fill_template(ball_template, {'DIM': ball_size})
            balls.append(env.add_object(ball_urdf, ball_pose, 'rigid', color=color))

            bowl_pose = (np.array([0.37 + i * 0.15, 0.1, 0.05]), (0, 0, 0, 1))
            bowl_urdf = 'bowl/bowl.urdf'
            env.add_object(bowl_urdf, bowl_pose, 'fixed', color=color)
            targs.append(bowl_pose)

        adjust_targs = [(pos[0] + np.array([-0.02, 0, 0.05]), pos[1]) for pos in targs]
        # adjust_size = bowl_size + np.array([0, 0, 0.065])

        for i in range(len(colors)):
            self.add_goal(objs=[balls[i]], matches=np.ones((1, 1)), targ_poses=[adjust_targs[i]],
                          replace=True,
                          rotations=False, metric='zone', params=[(targs[i], bowl_size)],
                          step_max_reward=1 / len(colors),
                          language_goal=self.lang_template.format(color=colors[i]))
