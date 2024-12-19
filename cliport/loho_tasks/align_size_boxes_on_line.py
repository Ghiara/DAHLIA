import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os


class AlignSizedBoxOnLine(Task):
    """Arrange a set of colored blocks with increasing size on a line."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "place {num} boxes in order of ascending sizes on the line from left to right averagely"
        self.task_completed_desc = "done aligning boxes."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        n_box = np.random.randint(3, 8)
        line_x = float(format(np.random.uniform(0.3, 0.68), '.3f'))

        # Add line
        line_size = (0.8, 0.01, 0.01)
        line_template = 'line/line-template.urdf'
        line_pose = ((line_x, 0, 0.01), utils.eulerXYZ_to_quatXYZW((0, 0, np.pi / 2)))
        replace = {'DIM': line_size, 'COLOR': utils.COLORS[random.sample(colors, 1)[0]]}
        line_urdf = self.fill_template(line_template, replace)
        env.add_object(line_urdf, line_pose, 'fixed')

        # Add boxs for each color
        box_size = np.array([0.03, 0.03, 0.03])
        box_template = 'box/box-template.urdf'
        increase = np.array([0.005, 0.005, 0.005])
        boxs = []
        for i in range(n_box):
            color = random.sample(colors, 1)[0]
            box_pose = self.get_random_pose(env, box_size + i * increase,
                                            bound=np.array([[0.36, 0.75], [-0.5, 0.5], [0, 0.3]]))
            replace = {'DIM': box_size + i * increase, 'COLOR': utils.COLORS[color]}
            box_urdf = self.fill_template(box_template, replace)
            box_id = env.add_object(box_urdf, box_pose)
            boxs.append((box_id, color))

        targ = []
        ini_pose = np.array([line_x, -0.4, 0.02])
        targ.append(ini_pose)
        for i in range(n_box):
            targ.append(np.zeros(3))
            targ[i + 1][0] = line_x
            targ[i + 1][1] = targ[i][1] + 0.8 / (n_box - 1)
            targ[i + 1][2] = ((box_size + (i + 1) * increase) / 2)[2]

        targ = targ[:-1]
        targ = [(pos, np.array([0, 0, 0, 1])) for pos in targ]

        # Add goals
        for i in range(n_box):
            self.add_goal(objs=[boxs[i][0]], matches=np.ones((1, 1)), targ_poses=[targ[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / n_box,
                          language_goal=self.lang_template.format(num=n_box))
