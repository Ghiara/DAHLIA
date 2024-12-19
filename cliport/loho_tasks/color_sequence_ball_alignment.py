import numpy as np
import os
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class ColorSequenceBallAlignment(Task):
    """
    Align four balls of different colors (red, blue, green, yellow) on corresponding colored lines,
    ensuring each ball is placed at the endpoint of its matching line without touching the others.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "align the {color} ball at the end of the {color} line"
        self.task_completed_desc = "all balls are aligned with their corresponding lines."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and their corresponding lines.
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'silver']
        line_positions = [(0.35, -0.25, 0.01), (0.35, 0, 0.01), (0.35, 0.25, 0.01),
                          (0.55, -0.25, 0.01), (0.55, 0, 0.01), (0.55, 0.25, 0.01)]
        ball_size = (0.04, 0.04, 0.04)  # Diameter of the ball
        ball_positions = [self.get_random_pose(env, ball_size) for _ in range(6)]
        balls = []
        lines = []
        targs = []
        # Add lines and balls to the environment.
        for i, color in enumerate(colors):
            # Add line.
            line_template = 'line/line-template.urdf'
            line_urdf = self.fill_template(line_template, {'COLOR': utils.COLORS[color]})
            line_pose = (line_positions[i], (0, 0, 0, 1))
            line_id = env.add_object(line_urdf, line_pose, 'fixed')
            lines.append(line_id)

            # Add ball.
            ball_urdf = 'ball/ball.urdf'
            ball_pose = ball_positions[i]
            ball_id = env.add_object(ball_urdf, ball_pose)
            p.changeVisualShape(ball_id, -1, rgbaColor=utils.COLORS[color] + [1])
            balls.append(ball_id)

            # Define the target position for the ball at the end of the line.
            target_position = (line_positions[i][0] + 0.02, line_positions[i][1], 0.02)
            target_pose = (target_position, (0, 0, 0, 1))
            targs.append(target_pose)

        # Add goal for each ball to be aligned at the end of its corresponding line.
        for color in colors:
            i = colors.index(color)
            self.add_goal(objs=[balls[i]], matches=np.eye(1), targ_poses=[targs[i]], replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 6,
                          symmetries=[0], language_goal=self.lang_template.format(color=color))
