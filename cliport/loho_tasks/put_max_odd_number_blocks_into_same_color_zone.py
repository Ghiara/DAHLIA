import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils
import pybullet as p


class PutOddNumberBlocksIntoSameColorZone(Task):
    """Place the maximal odd number of blocks with different colors into each zone that has the same color as the blocks."""

    def __init__(self):
        super().__init__()
        self.max_steps = 15
        self.lang_template = "place {num} {color} blocks into the {color} zone"
        self.task_completed_desc = "done placing blocks in zone"
        self.obj = {}
        self.goal = """In each zone, place the maximal odd number of blocks from the same color that corresponds the zone."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Randomly select the number of blocks to place
        n_blocks = np.random.randint(7, 15)
        n_colors = 3

        # Randomly select the color and shape of the blocks
        colors, color_names = utils.get_colors(mode=self.mode, n_colors=n_colors)
        targ = []

        # Add the blocks
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_template = 'box/box-template.urdf'
        block_urdf = self.fill_template(block_template, {'DIM': block_size})
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            color_index = np.random.randint(0, n_colors)
            block_color = colors[color_index]
            block_id = env.add_object(block_urdf, block_pose, color=block_color)
            blocks.append((block_id, color_index))
            self.obj[block_id] = {'color': color_names[color_index], 'shape': 'block'}

        # Add the zone
        zone_size = (0.1, 0.1, 0.01)
        zone_urdf = 'zone/zone.urdf'
        for i in range(3):
            zone_pose = self.get_random_pose(env, zone_size)
            zone_id = env.add_object(zone_urdf, zone_pose, category='fixed', color=colors[i])
            targ.append(((zone_pose[0][0], zone_pose[0][1], zone_pose[0][2] + block_size[2] / 2),
                         zone_pose[1]))
            self.obj[zone_id] = {'catagory': 'fixed', 'color': color_names[i], 'shape': 'zone'}

        # Group the blocks by color and shape
        groups = {}
        for block_id, color_index in blocks:
            color = color_names[color_index]
            if color not in groups:
                groups[color] = []
            groups[color].append(block_id)

        flag = 0
        for color in groups:
            group = groups[color]
            # t = [1+2*i for i in range(len(group)//2)]
            # count = np.random.choice(t)
            count = len(group) if len(group) % 2 == 1 else len(group) - 1
            if count <= 0:
                flag += 1

        # Create the language goals and placement goals for each group
        for color in groups:
            group = groups[color]
            # t = [1+2*i for i in range(len(group)//2)]
            # count = np.random.choice(t)
            count = len(group) if len(group) % 2 == 1 else len(group) - 1
            if count <= 0:
                continue
            language_goal = self.lang_template.format(num=count, color=color)
            i = color_names.index(color)
            block_targ = [
                ((targ[i][0][0], targ[i][0][1], targ[i][0][2] + p * block_size[2]), targ[i][1]) for
                p in range(count)]
            self.add_goal(objs=group[0:count], matches=np.eye(count), targ_poses=block_targ,
                          replace=False, rotations=True, metric='pose', params=None,
                          step_max_reward=1 / (len(groups) - flag),
                          language_goal=language_goal)
