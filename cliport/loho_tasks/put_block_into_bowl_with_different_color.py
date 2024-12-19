import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils


class PutBlockIntoBowlWithDifferentColor(Task):
    """Place each block into a bowl of a different color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Place the {} block into the {} bowl."
        self.task_completed_desc = "Done placing blocks into bowls."
        self.obj = {}
        self.goal = """Place all blocks into the bowls and make sure the blocks in each bowl are of different color from the bowl and each bowl has a similar number of cubes."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and corresponding names
        colors = {'red': utils.COLORS['red'], 'blue': utils.COLORS['blue'],
                  'green': utils.COLORS['green'], 'yellow': utils.COLORS['yellow']}
        color_names = list(colors.keys())

        # Add bowls
        bowl_size = (0.1, 0.1, 0.1)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for color in color_names:
            bowl_pose = self.get_random_pose(env, bowl_size * 2)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, category='fixed', color=colors[color],
                                     scale=1.5)
            bowl_poses.append(bowl_pose)
            self.obj[bowl_id] = {'catagory': 'fixed', 'color': color, 'shape': 'bowl'}

        # Add blocks
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'block/block.urdf'
        n_block = 6
        blocks = []
        for i in range(n_block):
            block_pose = self.get_random_pose(env, block_size)
            block_color_name = np.random.choice(color_names)
            block_id = env.add_object(block_urdf, block_pose, color=colors[block_color_name],
                                      scale=0.5)
            blocks.append((block_id, block_color_name))
            self.obj[block_id] = {'color': block_color_name, 'shape': 'block'}

        # Add goals
        for i in range(n_block):
            ex = bowl_poses[color_names.index(blocks[i][1])]
            targ = self.random_choice_except(bowl_poses, ex)
            language_goal = self.lang_template.format(blocks[i][1],
                                                      color_names[bowl_poses.index(targ)])
            self.add_goal(objs=[blocks[i][0]], matches=np.ones((1, 1)), targ_poses=[targ],
                          replace=False,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / n_block,
                          language_goal=language_goal)

    def random_choice_except(self, arr, exclude):
        remaining_elements = [element for element in arr if element != exclude]
        return remaining_elements[np.random.randint(0, len(remaining_elements))]
