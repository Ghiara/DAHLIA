import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import random
import pybullet as p
import os


class AlignSizedBoxOnCircle(Task):
    """Arrange a set of colored blocks with increasing size on a circle."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {} box on the circle"
        self.task_completed_desc = "done aligning boxes."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'violet']
        n_box = np.random.randint(3, 9)

        # Add circle
        circle_y = np.random.uniform(-0.4, 0.4)
        shape = os.path.join(self.assets_root, 'kitting', f"{self.kit['O']:02d}.obj")
        circle_pose = ((0.5, circle_y, 0.01), (0, 0, 0, 1))
        scale = [0.02, 0.02, 0.00001]  # .0005
        replace = {'FNAME': (shape,), 'SCALE': scale,
                   'COLOR': utils.COLORS[random.sample(colors, 1)[0]]}
        template = 'kitting/object-template-nocollision.urdf'
        urdf = self.fill_template(template, replace)
        circle_id = env.add_object(urdf, circle_pose, 'fixed')
        r = p.getAABB(circle_id)[0][0] / 2 - 0.06
        r = round(r, 2)

        # Add boxs for each color
        base_size = np.array([0.03, 0.03, 0.03])
        box_template = 'box/box-template.urdf'
        increase = np.array([0.005, 0.005, 0.005])
        boxs = []
        for i in range(n_box):
            color = random.sample(colors, 1)[0]
            box_size = base_size + i * increase
            box_pose = self.get_random_pose(env, box_size,
                                            bound=np.array([[0.25, 0.6], [-0.5, 0.5], [0, 0.3]]))
            replace = {'DIM': box_size, 'COLOR': utils.COLORS[color]}
            box_urdf = self.fill_template(box_template, replace)
            box_id = env.add_object(box_urdf, box_pose)
            boxs.append((box_id, box_size, color))

        angle = 2 * np.pi / n_box
        targ = [(utils.apply(circle_pose,
                             (r * np.cos(i * angle), r * np.sin(i * angle), boxs[i][1][2] / 2)),
                 utils.eulerXYZ_to_quatXYZW((0, 0, -angle * i))) for i in range(n_box)]

        # Add goals
        for i in range(n_box):
            self.add_goal(objs=[boxs[i][0]], matches=np.ones((1, 1)), targ_poses=[targ[i]],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / n_box,
                          language_goal=self.lang_template.format(boxs[i][2]))
