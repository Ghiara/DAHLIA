import random
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class ContainerStackPuzzle(Task):
    """Pick up different-sized blocks of various colors and stack them inside a container according to their sizes and colors, with the biggest block at the bottom and the smallest block at the top, and blocks of the same color stacked together."""

    def __init__(self):
        super().__init__()
        self.max_steps = 30
        self.lang_template = "stack all the blocks inside the container"
        self.task_completed_desc = "done stacking blocks inside the container"
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Add container.
        container_template = 'container/container-template.urdf'
        container_size = (0.2, 0.2, 0.04)
        replace = {'DIM': container_size,
                   'HALF': (container_size[0] / 2, container_size[1] / 2, container_size[2] / 2)}
        container_urdf = self.fill_template(container_template, replace)
        container_pose = self.get_random_pose(env, container_size)
        container_id = env.add_object(container_urdf, container_pose, 'fixed')

        # Add blocks.
        n_blocks = random.randint(5, 8)
        block_urdf = 'block/block.urdf'
        blocks = []
        scale = [np.random.uniform(1, 3) for _ in range(n_blocks)]
        scale.sort(reverse=True)

        for i in range(n_blocks):
            block_size = (0.03, 0.03, 0.03)
            # block_size = (round(i*scale,2) for i in block_size)
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, scale=scale[i])
            self.color_random_brown(block_id)
            blocks.append((block_id, block_size))

        # Sort blocks by size in descending order.
        # sort_blocks = sorted(blocks, key=lambda x: x[1][1], reverse=True)

        # Stack blocks inside the container.
        targs = []
        stack_height = 0
        for _, block_size in blocks:
            stack_height += block_size[2]
            stack_pose = (
            container_pose[0][0], container_pose[0][1], container_pose[0][2] + stack_height)
            stack_pose = (stack_pose, container_pose[1])
            targs.append(stack_pose)

        # Goal: all blocks are stacked inside the container.
        for i in range(len(blocks)):
            self.add_goal(objs=[blocks[i][0]], matches=np.eye(len(blocks)), targ_poses=[targs[i]],
                          replace=True,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(blocks), symmetries=[2 * np.pi],
                          language_goal=self.lang_template)
