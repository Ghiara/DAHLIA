import os
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ArrangeSameColorShapes(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Arrange the blocks into two separate rows on the table with the red blocks on the left and the blue blocks on the right."
        self.task_completed_desc = "done arranging blocks"
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define the colors and shapes for the task
        colors = ['red', 'red', 'red', 'blue', 'blue']
        shapes = ['block', 'block', 'block', 'block', 'sphere']

        # Shuffle the colors and shapes
        np.random.shuffle(colors)
        np.random.shuffle(shapes)

        # Create the objects and targets for the task
        objects = []
        targets = []

        # Add the objects and targets to the environment
        for i in range(len(colors)):
            color = colors[i]
            shape = shapes[i]

            # Create the object
            obj_size = self.get_random_size(0.03, 0.05, 0.03, 0.05, 0.03, 0.05)
            obj_pose = self.get_random_pose(env, obj_size)
            if shape == 'block':
                obj_id = env.add_object(f'{shape}/{shape}.urdf', obj_pose, color=color)
            else:
                obj_id = env.add_object(f'{shape}/{shape}-template.urdf', obj_pose, color=color)

            objects.append(obj_id)

            # Create the target
            if color == 'red' and obj_pose[0][1] > 0:
                target_pose = self.set_random_pose(env, obj_size, True)
            elif color == 'blue' and obj_pose[0][1] < 0:
                target_pose = self.set_random_pose(env, obj_size, False)
            else:
                objects.pop()
                continue
            targets.append(target_pose)

        matches = np.ones((len(objects), len(objects)))

        # Add the goal to the task
        self.add_goal(objs=objects, matches=matches, targ_poses=targets, replace=False,
                      rotations=True, metric='pose', params=None, step_max_reward=1.0,
                      language_goal=self.lang_template)

    def get_random_size(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.uniform((min_x, min_y, min_z), (max_x, max_y, max_z))
        return size

    def set_random_pose(self, env, obj_size, flag=True):
        """Get random pose within workspace bounds."""
        if flag:
            bounds = np.array([[0.25, 0.75], [-0.5, 0], [0, 0.3]])
        else:
            bounds = np.array([[0.25, 0.75], [0, 0.5], [0, 0.3]])
        pose = np.random.uniform(bounds[:, 0] + obj_size / 2, bounds[:, 1] - obj_size / 2)
        rotation = np.random.uniform(0, 2 * np.pi)
        return (pose[0], pose[1], obj_size[2] / 2), (0, 0, rotation, 1)
