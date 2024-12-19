import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os


class AlignColoredBoxOnLine(Task):
    """Arrange a set of colored blocks with rainbow colors on a line from left to right."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place the {} box on the line"
        self.task_completed_desc = "done aligning boxes."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        n_box = 7
        line_x = float(format(np.random.uniform(0.3, 0.68), '.3f'))

        # Add line
        line_size = (0.8, 0.01, 0.01)
        line_template = 'line/line-template.urdf'
        line_pose = ((line_x, 0, 0.01), utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2)))
        replace = {'DIM': line_size, 'COLOR': utils.COLORS[random.sample(colors, 1)[0]]}
        line_urdf = self.fill_template(line_template, replace)
        env.add_object(line_urdf, line_pose, 'fixed')

        # Add boxs for each color
        base_size = np.array([0.03, 0.03, 0.03])
        box_template = 'box/box-template.urdf'
        boxs = []
        for i in range(n_box):
            color = colors[i]
            box_size = base_size * np.random.uniform(1, 2)
            box_pose = self.get_random_pose(env, box_size,
                                            bound=np.array([[0.36, 0.7], [-0.5, 0.5], [0, 0.3]]))
            replace = {'DIM': box_size, 'COLOR': utils.COLORS[color]}
            box_urdf = self.fill_template(box_template, replace)
            box_id = env.add_object(box_urdf, box_pose)
            boxs.append((box_id, box_size))

        targ = []
        ini_pose = np.array([line_x, -0.4, boxs[0][1][2] / 2])
        targ.append(ini_pose)
        for i in range(n_box - 1):
            targ.append(np.zeros(3))
            targ[i + 1][0] = line_x
            targ[i + 1][1] = targ[i][1] + (boxs[i][1] / 2)[1] + (boxs[i + 1][1] / 2)[1] + 0.09
            targ[i + 1][2] = (boxs[i + 1][1] / 2)[2]

        targ = [(pos, np.array([0, 0, 0, 1])) for pos in targ]

        # Add goals
        for i in range(n_box):
            self.add_goal(objs=[boxs[i][0]], matches=np.ones((1, 1)), targ_poses=[targ[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / n_box,
                          language_goal=self.lang_template.format(colors[i]))
