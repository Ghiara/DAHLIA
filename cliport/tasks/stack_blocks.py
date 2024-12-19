"""Stacking Block Pyramid Sequence task."""

import random

import numpy as np
import pybullet as p

from cliport.tasks.color_reasoning import COLOR_CATEGORY_NAMES
from cliport.tasks.task import Task
from cliport.utils import utils
from cliport.tasks.grippers import Grip


class StackBlockPyramidSeqUnseenColors(Task):
    """Stacking Block Pyramid Sequence base class."""

    def __init__(self):
        super().__init__()
        self.max_steps = 12
        self.lang_template = "put the {pick} block on {place}"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[0],
                                                         place="the lightest brown block"))

        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[1],
                                                         place="the middle brown block"))

        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(pick=color_names[2],
                                                         place="the darkest brown block"))

        # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(
            pick=color_names[3], place=f"the {color_names[0]} and {color_names[1]} blocks")
        )

        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(
            pick=color_names[4], place=f"the {color_names[1]} and {color_names[2]} blocks")
        )

        # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template.format(
            pick=color_names[5], place=f"the {color_names[3]} and {color_names[4]} blocks")
        )

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockPyramidSeqSeenColors(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class StackBlockPyramidSeqFull(StackBlockPyramidSeqUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors


class StackBlockPyramidWithoutSeq(Task):
    """Stacking Block Pyramid without step-by-step instruction base class.
    There is just a high-level instruction: Stack blocks in a pyramid shape """

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "stack blocks in a pyramid shape"
        self.task_completed_desc = "done stacking block pyramid."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(6):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Associate placement locations for goals.
        place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
                     (0, 0.05, 0.03), (0, -0.025, 0.08),
                     (0, 0.025, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        # self.goals.append((objs, np.eye(6), targs, False, True, 'pose', None, 1))
        # self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.
        self.goals.append(([objs[0]], np.ones((1, 1)), [targs[0]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[0],
        #                                                  place="the lightest brown block"))
        #
        self.goals.append(([objs[1]], np.ones((1, 1)), [targs[1]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[1],
        #                                                  place="the middle brown block"))
        #
        self.goals.append(([objs[2]], np.ones((1, 1)), [targs[2]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(pick=color_names[2],
        #                                                  place="the darkest brown block"))
        #
        # # Goal: make middle row.
        self.goals.append(([objs[3]], np.ones((1, 1)), [targs[3]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(
        # pick=color_names[3], place=f"the {color_names[0]} and {color_names[1]} blocks")
        # )
        #
        self.goals.append(([objs[4]], np.ones((1, 1)), [targs[4]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(
        # pick=color_names[4], place=f"the {color_names[1]} and {color_names[2]} blocks")
        # )
        #
        # # Goal: make top row.
        self.goals.append(([objs[5]], np.ones((1, 1)), [targs[5]],
                           False, True, 'pose', None, 1 / 6))
        self.lang_goals.append(self.lang_template)
        # self.lang_goals.append(self.lang_template.format(
        # pick=color_names[5], place=f"the {color_names[3]} and {color_names[4]} blocks")
        # )

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocks(Task):
    def __init__(self):
        super().__init__()

        self.smaller_block_size, self.bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        self.smaller_block_urdf, self.bigger_block_urdf = ('stacking/block.urdf',
                                                           'stacking/bigger_block.urdf')
        self.all_color_names = self.get_colors()
        self.n_blocks = None
        self.block_color_names, self.block_colors = None, None
        self.n_block_distractors, self.n_zone_distractors = None, None
        #self.ee=Grip

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True

        # TODO: implement this part.
        if self.task_difficulty_level == "easy":
            # No distractor blocks or zones, and no blocks with repetitive colors.
            self.n_blocks = np.random.randint(4, 8)
            self.block_color_names = np.random.choice(
                a=self.all_color_names, size=self.n_blocks, replace=False
            )
        elif self.task_difficulty_level == "medium":
            # There exist distractor blocks and zones.
            self.n_blocks = np.random.randint(4, 8)
            self.block_color_names = np.random.choice(
                a=self.all_color_names, size=self.n_blocks, replace=False
            )

            self.n_block_distractors = np.random.randint(1, 3)
            self.n_zone_distractors = np.random.randint(1, 3)

        elif self.task_difficulty_level == "hard":
            # More blocks and their colors can be repetitive.
            self.n_blocks = np.random.randint(8, 11)
            self.block_color_names = np.random.choice(
                a=self.all_color_names, size=self.n_blocks, replace=True
            )
        else:
            raise ValueError(f"Unknown task_difficulty_level: {self.task_difficulty_level}")

        self.block_colors = [utils.COLORS[cn] for cn in self.block_color_names]
        self.max_steps = self.n_blocks + 2

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlock(Task):
    """Stacking ALL Blocks without step-by-step instruction base class.
    There is just a high-level instruction: Stack all blocks. """

    # TODO: discarded.

    def __init__(self):
        super().__init__()
        self.n_blocks = random.randint(2, 5)
        self.max_steps = self.n_blocks + 2

        self.lang_template = "stack all the blocks"
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)

        # Add base.
        base_size = (0.05, 0.15, 0.005)
        base_urdf = 'stacking/stand.urdf'
        base_pose = self.get_random_pose(env, base_size)
        env.add_object(base_urdf, base_pose, 'fixed')

        # Block colors.
        color_names = self.get_colors()

        # Shuffle the block colors.
        random.shuffle(color_names)
        colors = [utils.COLORS[cn] for cn in color_names]

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        self.scene_description = f"On the table, there are {self.n_blocks} blocks. " \
                                 f"Their colors are {color_names[:self.n_blocks]}. "

        # Associate placement locations for goals.
        # place_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #              (0, 0.05, 0.03), (0, -0.025, 0.08),
        #              (0, 0.025, 0.08), (0, 0, 0.13)]
        place_pos = [(0, 0, 0.03 + 0.05 * i) for i in range(self.n_blocks)]
        # place_pos = [(0, 0, 0.03), (0, 0, 0.08), (0, 0, 0.13)]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]

        match_matrix = np.eye(self.n_blocks)
        # match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, targs, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)
        # Goal: make bottom row.

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlocksOnAZone(StackBlocks):
    """Stacking ALL Blocks In A Zone without step-by-step instruction base class.
    There is just a high-level instruction: Stack all blocks in a zone. """

    def __init__(self):
        super().__init__()
        self.lang_template = "stack all the blocks on the {zone_color} zone."
        self.task_completed_desc = "done stacking block."
        self.obj = {}
        self.goal = 'Stack all blocks in the {zone_color} zone.'

    def reset(self, env):
        super().reset(env)
        self.print_debug_info = False
        self.n_blocks = np.random.randint(2, 6)
        self.block_color_names = np.random.choice(
            a=self.all_color_names, size=self.n_blocks, replace=False
        )
        self.block_colors = [utils.COLORS[cn] for cn in self.block_color_names]
        self.max_steps = self.n_blocks + 2

        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            self.obj[block_id] = {'color': self.block_color_names[i], 'shape': 'block'}

        # Add zones.
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_size = (0.1, 0.1, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])
        self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': zone_color_name, 'shape': 'zone'}

        goal_poses = []
        pos, rot = zone_pose
        for i in range(self.n_blocks):
            height = block_size[2] / 2 + block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, goal_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(zone_color=zone_color_name))
        self.goal = self.goal.format(zone_color=zone_color_name)

        self.scene_description = (f"On the table, there are {self.n_blocks} blocks. "
                                  f"Their colors are {self.block_color_names[:self.n_blocks]}. "
                                  f"There is a zone on the table, "
                                  f"and its color is {zone_color_name}. ")
        
        filtered_color = [color for color in self.all_color_names if color != zone_color_name]
        dis_zone_color = np.random.choice(a=filtered_color, size=1)[0]
        zone_pose = self.get_random_pose(env, zone_size)
        distractor_zone = env.add_object('zone/zone.urdf', zone_pose, 'fixed', color=utils.COLORS[dis_zone_color])
        self.obj[distractor_zone] = {'catagory': 'fixed', 'color': dis_zone_color, 'shape': 'zone'}

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackAllBlocksOnAZoneWithDetails(StackBlocks):
    """Stacking ALL Blocks In A Zone without step-by-step instruction base class.
    There is just a high-level instruction: Stack all blocks in a zone. """

    def __init__(self):
        super().__init__()
        self.lang_template = ("The goal is that stack all the blocks on the {zone_color} zone. "
                              "The step-by-step instructions are: {step_instructions}")
        self.task_completed_desc = "done stacking block."

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = True

        self.print_debug_info = False
        self.n_blocks = np.random.randint(2, 6)
        self.block_color_names = np.random.choice(
            a=self.all_color_names, size=self.n_blocks, replace=False
        )
        self.block_colors = [utils.COLORS[cn] for cn in self.block_color_names]
        self.max_steps = self.n_blocks + 2

        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        # Add zones.
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_size = (0.1, 0.1, 0)
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        goal_poses = []
        pos, rot = zone_pose
        for i in range(self.n_blocks):
            height = block_size[2] / 2 + block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.ones((self.n_blocks, self.n_blocks))
        self.goals.append((objs, match_matrix, goal_poses, False, True, 'pose', None, 1))

        # Step instructions.
        step_instructions = []
        first_step_instruction_template = ("Pick up the {block_color} block "
                                           "and place it on the {zone_color} zone.")
        later_step_instruction_template = ("Pick up the {first_block_color} block "
                                           "and place it on the {second_block_color} block.")
        step_instructions.append(
            first_step_instruction_template.format(block_color=self.block_color_names[0],
                                                   zone_color=zone_color_name)
        )
        for i in range(1, self.n_blocks):
            step_instructions.append(
                later_step_instruction_template.format(
                    first_block_color=self.block_color_names[i],
                    second_block_color=self.block_color_names[i - 1]
                )
            )
        step_instructions = " ".join(step_instructions)

        self.lang_goals.append(self.lang_template.format(zone_color=zone_color_name,
                                                         step_instructions=step_instructions))

        self.scene_description = (f"On the table, there are {self.n_blocks} blocks. "
                                  f"Their colors are {self.block_color_names[:self.n_blocks]}. "
                                  f"There is a zone on the table, "
                                  f"and its color is {zone_color_name}. ")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksOfSameSize(StackBlocks):
    """Stack blocks of the same size.
    Stack all bigger blocks, and stack all smaller blocks.
    There are at most 4 blocks of the same size.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_bigger_blocks, self.n_smaller_blocks = 0, 0
        self.lang_template = ("stack blocks of the same size "
                              "on the {first_zone} zone and {second_zone} zone respectively.")
        self.task_completed_desc = "done stacking all the objects"
        self.obj = {}
        self.goal = "stack blocks of the same size on the {first_zone} zone and {second_zone} zone respectively."

    def reset(self, env):
        super().reset(env)

        all_color_names = self.get_colors()
        self.n_bigger_blocks = np.random.randint(1, 5)
        self.n_smaller_blocks = np.random.randint(1, 5)
        bigger_color_names = random.sample(all_color_names, self.n_bigger_blocks)
        smaller_color_names = random.sample(all_color_names, self.n_smaller_blocks)
        bigger_colors = [utils.COLORS[cn] for cn in bigger_color_names]
        smaller_colors = [utils.COLORS[cn] for cn in smaller_color_names]

        self.max_steps = self.n_bigger_blocks + self.n_smaller_blocks + 2

        # Add zones
        zone_color=[]
        zone_poses = []
        for i in range(2):
            zone_size = (0.1, 0.1, 0)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            color = bigger_colors[0] if i == 0 else smaller_colors[0]
            color_name = bigger_color_names[0] if i == 0 else smaller_color_names[0]
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=color + [1])
            zone_poses.append(zone_pose)
            zone_color.append(color_name)
            self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': color_name, 'shape': 'zone'}

        # Add blocks
        bigger_blocks, smaller_blocks = [], []
        for i in range(self.n_bigger_blocks):
            block_pose = self.get_random_pose(env, self.bigger_block_size)  # (pos, rot)
            block_id = env.add_object(self.bigger_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=bigger_colors[i] + [1])
            bigger_blocks.append((block_id, (0, None)))
            self.obj[block_id] = {'color': bigger_color_names[i], 'shape': 'big block'}
        for i in range(self.n_smaller_blocks):
            block_pose = self.get_random_pose(env, self.smaller_block_size)  # (pos, rot)
            block_id = env.add_object(self.smaller_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=smaller_colors[i] + [1])
            smaller_blocks.append((block_id, (0, None)))
            self.obj[block_id] = {'color': smaller_color_names[i], 'shape': 'small block'}

        # Goal
        pos, rot = zone_poses[0]
        bigger_block_goal_poses = []
        for i in range(self.n_bigger_blocks):
            goal_height = self.bigger_block_size[2] * i + self.bigger_block_size[2] / 2
            bigger_block_goal_poses.append(((pos[0], pos[1], goal_height), rot))

        pos, rot = zone_poses[1]
        smaller_block_goal_poses = []
        for i in range(self.n_smaller_blocks):
            goal_height = self.smaller_block_size[2] * i + self.smaller_block_size[2] / 2
            smaller_block_goal_poses.append(((pos[0], pos[1], goal_height), rot))

        match_matrix = np.zeros((self.n_bigger_blocks + self.n_smaller_blocks,
                                 self.n_bigger_blocks + self.n_smaller_blocks))
        match_matrix[:self.n_bigger_blocks, :self.n_bigger_blocks] = 1
        match_matrix[self.n_bigger_blocks:, self.n_bigger_blocks:] = 1

        self.goals.append(
            (
                bigger_blocks + smaller_blocks,
                match_matrix,
                bigger_block_goal_poses + smaller_block_goal_poses,
                False,
                True,
                'pose',
                None,
                1
            )
        )
        self.lang_goals.append(self.lang_template.format(first_zone=bigger_color_names[0],
                                                         second_zone=smaller_color_names[0]))
        self.goal = self.goal.format(first_zone=bigger_color_names[0], second_zone=smaller_color_names[0])

        block_color_list = bigger_color_names + smaller_color_names
        np.random.shuffle(block_color_list)
        self.scene_description = (
            f"On the table, there are {self.n_bigger_blocks + self.n_smaller_blocks} blocks. "
            f"Their colors are {', '.join(block_color_list)}. "
            f"The size of the blocks are different. "
            f"Some blocks are 'bigger' and some blocks are 'smaller'.")
        
        filtered_color = [color for color in self.all_color_names if color not in zone_color]
        dis_zone_color = np.random.choice(a=filtered_color, size=1)[0]
        zone_pose = self.get_random_pose(env, zone_size)
        distractor_zone = env.add_object('zone/zone.urdf', zone_pose, 'fixed', color=utils.COLORS[dis_zone_color])
        self.obj[distractor_zone] = {'catagory': 'fixed', 'color': dis_zone_color, 'shape': 'zone'}

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockOfSameColor(Task):
    """
    Stack ALL Blocks of the same color together on the same colored zone.
    There are at most three kinds of color and at most 3 blocks with the same color.
   """

    def __init__(self):
        super().__init__()
        self.n_block_colors, self.n_blocks, self.n_distractors = None, None, None
        self.n_zones = None

        self.lang_template = ("stack all the blocks of the "
                              "same color together on the same colored zone.")
        self.task_completed_desc = "done stacking block."
        self.obj = {}
        self.goal = "stack all the blocks of the same color together in the same colored zone."

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True

        self.n_block_colors = np.random.randint(1, 4)
        if self.n_block_colors == 1:
            self.n_blocks = np.random.randint(2, 4)
        elif self.n_block_colors == 2:
            self.n_blocks = np.random.randint(4, 7)
        else:  # 3
            self.n_blocks = np.random.randint(6, 10)

        self.n_zones = np.random.randint(self.n_block_colors, self.n_block_colors + 2)

        self.n_distractors = np.random.randint(0, 3)
        self.max_steps = self.n_blocks + 2

        # Block colors.
        color_names = self.get_colors()
        random.shuffle(color_names)
        block_color_names = random.sample(color_names, self.n_block_colors)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add zones
        zone_poses = []
        for i in range(self.n_block_colors):
            zone_size = (0.1, 0.1, 0)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=block_colors[i] + [1])
            zone_poses.append(zone_pose)
            self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': block_color_names[i], 'shape': 'zone'}

        # Add blocks.
        block_dict = {c:[] for c in block_color_names}
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        min_color_num = self.n_blocks // self.n_block_colors
        for color in block_dict.keys():
            for i in range(min_color_num):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose, color=color)
                block_dict[color].append((block_id, (np.pi / 2, None)))
                self.obj[block_id] = {'color': color, 'shape': 'block'}
        remain_num = self.n_blocks % self.n_block_colors
        for i in range(remain_num):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose, color=list(block_dict.keys())[i])
            block_dict[list(block_dict.keys())[i]].append((block_id, (np.pi / 2, None)))
            self.obj[block_id] = {'color': list(block_dict.keys())[i], 'shape': 'block'}
        for value in block_dict.values():
            objs += value

        goal_poses = []
        for i, blocks in enumerate(block_dict.values()):
            pos, rot = zone_poses[i]
            for j in range(len(blocks)):
                height = block_size[2] / 2 + block_size[2] * j
                goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.zeros((self.n_blocks, self.n_blocks))
        start = 0
        end = 0
        for blocks in block_dict.values():
            end += len(blocks)
            match_matrix[start:end, start:end] = 1
            start += len(blocks)

        self.goals.append(
            (objs, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template)

        # Add distractor zones
        n_distractor_zones = self.n_zones - self.n_block_colors
        remaining_names = [c for c in color_names if c not in block_color_names]
        distractor_zone_color_names = []
        if n_distractor_zones > 0:
            distractor_zone_color_names = np.random.choice(a=remaining_names,
                                                           size=n_distractor_zones)
            for i in range(self.n_zones - self.n_block_colors):
                zone_size = (0.1, 0.1, 0)
                zone_color = utils.COLORS[distractor_zone_color_names[i]]
                zone_pose = self.get_random_pose(env, zone_size)
                zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
                p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])
                self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': distractor_zone_color_names[i], 'shape': 'zone'}

        # Add distractor blocks
        remaining_names = [c for c in remaining_names if c not in distractor_zone_color_names]
        distractor_color = [utils.COLORS[c] for c in remaining_names]
        n_distractors = min(self.n_distractors, len(distractor_color))
        for i in range(n_distractors):
            pose = self.get_random_pose(env, block_size)
            distractor_id = env.add_object(block_urdf, pose)
            p.changeVisualShape(distractor_id, -1, rgbaColor=distractor_color[i] + [1])
            self.obj[distractor_id] = {'color': remaining_names[i], 'shape': 'block'}

        block_color_list = block_color_names + remaining_names[:n_distractors]
        np.random.shuffle(block_color_list)
        self.scene_description = (f"On the table, there are {self.n_blocks + n_distractors} "
                                  f"blocks, "
                                  f"and their colors are {', '.join(block_color_list)}.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksWithAlternateColor(Task):
    """Stack all blocks in the form of color crossed on a zone, starting from a specific color.
    There are always two kinds of color and at most 3 blocks with the same color.
   """

    def __init__(self):
        super().__init__()
        self.n_blocks, self.max_steps = None, None
        self.n_block_colors = None

        self.lang_template = ("stack blocks with alternate colors "
                              "on the {zone_color} zone, starting with the {block_color} color.")
        self.task_completed_desc = "done stacking block."
        self.obj = {}
        self.goal = "stack blocks with alternate colors on the {zone_color} zone, starting with the {block_color} color."

    def reset(self, env):
        super().reset(env)
        self.n_blocks = np.random.randint(2, 6)
        self.n_block_colors = 2
        self.max_steps = self.n_blocks + 2
        self.input_manipulate_order = True

        # Block colors.
        color_names = self.get_colors()
        block_color_names = random.sample(color_names, self.n_block_colors)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add a zone
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])
        self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': zone_color_name, 'shape': 'zone'}

        # Add blocks.
        objs = []
        # sym = np.pi / 2
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1,
                                rgbaColor=block_colors[i % self.n_block_colors] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            self.obj[block_id] = {'color': block_color_names[i % self.n_block_colors], 'shape': 'block'}

        goal_poses = []
        pos, rot = zone_pose
        for i in range(self.n_blocks):
            height = block_size[2] / 2 + block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.zeros((self.n_blocks, self.n_blocks))
        for i in range(self.n_blocks):
            if i % self.n_block_colors == 0:
                match_matrix[i][0::2] = 1
            else:
                match_matrix[i][1::2] = 1

        self.goals.append((objs, match_matrix, goal_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(zone_color=zone_color_name,
                                                         block_color=block_color_names[0]))
        self.goal = self.goal.format(zone_color=zone_color_name, block_color=block_color_names[0])

        filtered_color = [color for color in self.get_colors() if color != zone_color]
        dis_zone_color = np.random.choice(a=filtered_color, size=1)[0]
        zone_pose = self.get_random_pose(env, zone_size)
        distractor_zone = env.add_object('zone/zone.urdf', zone_pose, 'fixed', color=utils.COLORS[dis_zone_color])
        self.obj[distractor_zone] = {'catagory': 'fixed', 'color': dis_zone_color, 'shape': 'zone'}

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByColor(StackBlocks):
    """
    Instruction example:
    Stack only blocks of [cool] colors on the [yellow] zone.
   """

    def __init__(self):
        super().__init__()

        self.color_category, self.selected_color_names = None, None
        self.lang_template = (
            "stack only the blocks of {color_category} colors on the {zone_color} zone."
        )
        self.task_completed_desc = "done stacking blocks."

    def reset(self, env):
        super().reset(env)

        self.color_category = np.random.choice(
            a=["cool", "warm", "primary", "secondary"], size=1
        )[0]
        self.selected_color_names = COLOR_CATEGORY_NAMES[self.color_category.upper()]

        # Add a zone
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        # Add blocks.
        objs = []
        selected_blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        first_block_pose = None
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            # Select blocks of cool colors
            if self.block_color_names[i] in self.selected_color_names:
                if not first_block_pose:
                    first_block_pose = block_pose
                selected_blocks.append((block_id, (np.pi / 2, None)))

        goal_poses = []
        pos, rot = zone_pose
        for i in range(len(selected_blocks)):
            height = block_size[2] / 2 + block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        # match_matrix = np.eye((len(selected_blocks)))
        match_matrix = np.ones((len(selected_blocks), len(selected_blocks)))
        self.goals.append(
            (selected_blocks, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template.format(color_category=self.color_category,
                                                         zone_color=zone_color_name))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByColorAndSize(StackBlocks):
    """
    Instruction example:
    Stack only the bigger blocks of warm color on the yellow zone.
    """

    def __init__(self):
        super().__init__()

        self.color_category, self.size_category = None, None
        self.selected_color_names = None

        self.lang_template = ("stack only the {block_size} blocks "
                              "of {color_category} color on the {zone_color} zone.")
        self.task_completed_desc = "done stacking blocks."
        self.obj = {}
        self.goal = 'Stack only the {block_size} blocks of {color_category} color in the {zone_color} zone.'

    def reset(self, env):
        super().reset(env)

        self.size_category = np.random.choice(
            a=["bigger", "smaller"], size=1
        )[0]
        self.color_category = np.random.choice(
            a=["cool", "warm",], size=1 #"primary", "secondary"
        )[0]
        self.selected_color_names = COLOR_CATEGORY_NAMES[self.color_category.upper()]

        # Block colors.
        color_names = self.get_colors()
        random.shuffle(color_names)
        block_color_names = np.random.choice(a=color_names, size=self.n_blocks, replace=True)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        # Add a zone.
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])
        self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': zone_color_name, 'shape': 'zone'}

        # Add blocks.
        objs = []
        selected_blocks = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf, bigger_block_urdf = ('stacking/block.urdf',
                                                 "stacking/bigger_block.urdf")
        n_bigger_blocks = np.random.randint(0, self.n_blocks + 1)
        for i in range(self.n_blocks):
            if i < n_bigger_blocks:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
                self.obj[block_id] = {'color': block_color_names[i], 'shape': 'big block'}
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
                self.obj[block_id] = {'color': block_color_names[i], 'shape': 'small block'}

            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            # Select blocks of selected colors
            if block_color_names[i] in self.selected_color_names:
                if (
                        (self.size_category == "bigger" and i < n_bigger_blocks)
                        or
                        (self.size_category == "smaller" and i >= n_bigger_blocks)
                ):
                    selected_blocks.append((block_id, (np.pi / 2, None)))

        goal_poses = []
        pos, rot = zone_pose
        for i in range(len(selected_blocks)):
            if self.size_category == "bigger":
                height = bigger_block_size[2] / 2 + bigger_block_size[2] * i
            else:
                height = smaller_block_size[2] / 2 + smaller_block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        # match_matrix = np.eye(len(selected_blocks))
        match_matrix = np.ones((len(selected_blocks), len(selected_blocks)))
        self.goals.append(
            (selected_blocks, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template.format(block_size=self.size_category,
                                                         color_category=self.color_category,
                                                         zone_color=zone_color_name))
        self.goal = self.goal.format(block_size=self.size_category, color_category=self.color_category,
                         zone_color=zone_color_name)

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByColorInSizeOrder(StackBlocks):
    """
    Example: "Stack blocks of cool colors in ascending order from big to small on a red zone."
    """

    def __init__(self):
        super().__init__()

        self.color_category = None
        self.selected_color_names = None

        self.lang_template = (
            "stack only the blocks of {color_category} colors "
            "in ascending order from big to small on the {zone_color} zone."
        )
        self.task_completed_desc = "done stacking blocks."

    def reset(self, env):
        super().reset(env)

        self.color_category = np.random.choice(
            a=["cool", "warm", "primary", "secondary"], size=1
        )[0]
        self.selected_color_names = COLOR_CATEGORY_NAMES[self.color_category.upper()]

        # Add a zone.
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        # Add blocks.
        objs = []
        selected_blocks = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf, bigger_block_urdf = ('stacking/block.urdf',
                                                 "stacking/bigger_block.urdf")
        n_bigger_blocks = np.random.randint(0, self.n_blocks + 1)
        selected_n_bigger = 0
        for i in range(self.n_blocks):
            if i < n_bigger_blocks:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)

            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            # Select blocks of selected colors
            if self.block_color_names[i] in self.selected_color_names:
                selected_blocks.append((block_id, (np.pi / 2, None)))
                if i < n_bigger_blocks:
                    selected_n_bigger += 1

        goal_poses_for_bigger, goal_poses_for_smaller = [], []
        pos, rot = zone_pose
        for i in range(selected_n_bigger):
            height = bigger_block_size[2] / 2 + bigger_block_size[2] * i
            goal_poses_for_bigger.append(((pos[0], pos[1], height), rot))
        for i in range(selected_n_bigger, len(selected_blocks)):
            height = (
                    bigger_block_size[2] * selected_n_bigger +
                    smaller_block_size[2] / 2 + smaller_block_size[2] * (i - selected_n_bigger)
            )
            goal_poses_for_smaller.append(((pos[0], pos[1], height), rot))

        self.goals.append(
            (
                selected_blocks[:selected_n_bigger],
                np.ones((selected_n_bigger, selected_n_bigger)),
                goal_poses_for_bigger,
                False, True, 'pose', None, 1
            )
        )
        self.lang_goals.append(self.lang_template.format(color_category=self.color_category,
                                                         zone_color=zone_color_name))

        selected_n_smaller = len(selected_blocks) - selected_n_bigger
        self.goals.append(
            (
                selected_blocks[selected_n_bigger:],
                np.ones((selected_n_smaller, selected_n_smaller)),
                goal_poses_for_smaller,
                False, True, 'pose', None, 1
            )
        )
        self.lang_goals.append(self.lang_template.format(color_category=self.color_category,
                                                         zone_color=zone_color_name))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByRelativePositionAndColor(StackBlocks):
    """
    Task instruction:
    "Stack all the blocks on the left/right/top/bottom
    of the {color} block on the {color} zone."
    """

    def __init__(self):
        super().__init__()
        self.position_category = None
        self.lang_template = ("stack all the blocks on the {rel_pos} of "
                              "the {block_color} block on the {zone_color} zone.")
        self.task_completed_desc = "done stacking blocks."
        self.max_steps = 6
        self.obj = {}
        self.goal = 'Stack all the blocks, which are to the {rel_pos} of the {block_color} block with {pos_typ} larger than {dis}, in the {zone_color} zone.'

    def reset(self, env):
        super().reset(env)
        self.position_category = np.random.choice(
            a=["left", "right", "rear", "front"], size=1
        )[0]

        # Add a zone
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])
        self.obj[zone_obj_id] = {'catagory': 'fixed', 'color': zone_color_name, 'shape': 'zone'}

        # Add blocks.
        objs = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        # Use the first block as the reference block
        first_block_pose = None
        block_poses = []
        for i in range(self.n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            block_poses.append((block_id, block_pose))
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            self.obj[block_id] = {'color': self.block_color_names[i], 'shape': 'block'}
            # Select blocks of primary colors
            if i == 0:
                first_block_pose = block_pose

        selected_blocks = []
        for i, (block_id, block_pose) in enumerate(block_poses):
            pos, rot = block_pose
            if (
                    (
                            self.position_category == "left"
                            and
                            pos[1] + self.VISIBLE_DIFF < first_block_pose[0][1]
                    )
                    or
                    (
                            self.position_category == "right"
                            and
                            pos[1] - self.VISIBLE_DIFF > first_block_pose[0][1]
                    )
                    or
                    (
                            self.position_category == "rear"
                            and
                            pos[0] + self.VISIBLE_DIFF < first_block_pose[0][0]
                    )
                    or
                    (
                            self.position_category == "front"
                            and
                            pos[0] - self.VISIBLE_DIFF > first_block_pose[0][0]
                    )
            ):
                selected_blocks.append(objs[i])

        goal_poses = []
        pos, rot = zone_pose
        for i in range(len(selected_blocks)):
            height = block_size[2] / 2 + block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.ones((len(selected_blocks), len(selected_blocks)))
        self.goals.append(
            (selected_blocks, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        pos_typ = 'front-back distance' if self.position_category in ['rear','front'] else 'horizontal distance'
        self.lang_goals.append(self.lang_template.format(rel_pos=self.position_category,
                                                         dis=self.VISIBLE_DIFF,
                                                         block_color=self.block_color_names[0],
                                                         zone_color=zone_color_name,
                                                         pos_typ=pos_typ))
        self.goal = self.goal.format(rel_pos=self.position_category, block_color=self.block_color_names[0],
                         dis=self.VISIBLE_DIFF, zone_color=zone_color_name, pos_typ=pos_typ)

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByRelativePositionAndColorAndSize(StackBlocks):
    """
    Task instruction:
    "Stack all the bigger/smaller blocks on the left/right/top/bottom of the {color} block
    on the {color} zone."
    """

    def __init__(self):
        super().__init__()
        self.position_category, self.size_category = None, None
        self.lang_template = ("stack all the {size_category} blocks on the {rel_pos} of "
                              "the {block_color} block on the {zone_color} zone.")
        self.task_completed_desc = "done stacking blocks."

    def reset(self, env):
        super().reset(env)
        self.position_category = np.random.choice(
            a=["left", "right", "top", "bottom"], size=1
        )[0]
        self.size_category = np.random.choice(a=["smaller", "bigger"], size=1)[0]

        # Add a zone
        zone_size = (0.1, 0.1, 0)
        zone_color_name = np.random.choice(a=self.all_color_names, size=1)[0]
        zone_color = utils.COLORS[zone_color_name]
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        # Add blocks.
        objs = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf, bigger_block_urdf = ('stacking/block.urdf',
                                                 "stacking/bigger_block.urdf")
        n_bigger_blocks = np.random.randint(0, self.n_blocks + 1)
        # Use the first block as the reference block
        first_block_pose = None
        block_poses = []
        for i in range(self.n_blocks):
            if i < n_bigger_blocks:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
            block_poses.append((block_id, block_pose))
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))
            if i == 0:
                first_block_pose = block_pose

        selected_blocks = []

        for i, (block_id, block_pose) in enumerate(block_poses):
            pos, rot = block_pose
            if (
                    (self.position_category == "left" and pos[1] < first_block_pose[0][1])
                    or
                    (self.position_category == "right" and pos[1] > first_block_pose[0][1])
                    or
                    (self.position_category == "top" and pos[0] > first_block_pose[0][0])
                    or
                    (self.position_category == "bottom" and pos[0] < first_block_pose[0][0])
            ):
                if (
                        (self.size_category == "bigger" and i < n_bigger_blocks)
                        or
                        (self.size_category == "smaller" and i >= n_bigger_blocks)
                ):
                    selected_blocks.append(objs[i])

        goal_poses = []
        pos, rot = zone_pose
        for i in range(len(selected_blocks)):
            if self.size_category == "bigger":
                height = bigger_block_size[2] / 2 + bigger_block_size[2] * i
            else:
                height = smaller_block_size[2] / 2 + smaller_block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.ones((len(selected_blocks), len(selected_blocks)))
        self.goals.append(
            (selected_blocks, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template.format(size_category=self.size_category,
                                                         rel_pos=self.position_category,
                                                         block_color=self.block_color_names[0],
                                                         zone_color=zone_color_name, ))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByAbsolutePositionAndColorInSizeOrder(StackBlocks):
    """
    Task instruction:
    "Stack all the blocks on the {abs_pos} area on the {color} zone in size order."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.size_category, self.abs_position_category = None, None
        self.n_zones = None

        self.lang_template = ("stack all the blocks on the {abs_pos}"
                              " on the {zone_color} zone in size order.")
        self.task_completed_desc = "done stacking blocks."

    def reset(self, env):
        super().reset(env)

        self.abs_position_category = np.random.choice(
            a=["top left area", "top right area", "bottom left area", "bottom right area"],
            size=1,
        )[0]
        # self.size_category = np.random.choice(a=["smaller", "bigger"], size=1)[0]

        # Add zones.
        self.n_zones = np.random.randint(2, 4)
        zone_color_names = np.random.choice(
            self.all_color_names, size=self.n_zones, replace=False
        )
        zone_colors = [utils.COLORS[cn] for cn in zone_color_names]

        # Add a zone
        target_zone_pose = None
        for i in range(self.n_zones):
            zone_size = (0.1, 0.1, 0)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_colors[i] + [1])
            # Use the first zone as the target zone.
            if i == 0:
                target_zone_pose = zone_pose

        # Add blocks.
        objs = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf, bigger_block_urdf = ('stacking/block.urdf',
                                                 "stacking/bigger_block.urdf")
        n_bigger_blocks = np.random.randint(0, self.n_blocks + 1)
        # Use the first block as the reference block
        block_poses = []
        for i in range(self.n_blocks):
            if i < n_bigger_blocks:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
            block_poses.append((block_id, block_pose))
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        selected_blocks = []

        for i, (block_id, block_pose) in enumerate(block_poses):
            pos, rot = block_pose
            if (
                    (
                            self.area_boundary[self.abs_position_category]["x_start"]
                            <= pos[0] <=
                            self.area_boundary[self.abs_position_category]["x_end"]
                    )
                    and
                    (
                            self.area_boundary[self.abs_position_category]["y_start"]
                            <= pos[1] <=
                            self.area_boundary[self.abs_position_category]["y_end"]

                    )
            ):
                selected_blocks.append((i, objs[i]))

        goal_poses = []
        pos, rot = target_zone_pose
        selected_smaller_blocks, selected_bigger_blocks = [], []
        for idx, blk in selected_blocks:
            if idx < n_bigger_blocks:
                selected_bigger_blocks.append(blk)
            else:
                selected_smaller_blocks.append(blk)
        for i in range(len(selected_bigger_blocks)):
            height = bigger_block_size[2] / 2 + bigger_block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))
        for i in range(len(selected_smaller_blocks)):
            height = smaller_block_size[2] / 2 + smaller_block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.zeros((len(selected_bigger_blocks) + len(selected_smaller_blocks),
                                 len(selected_bigger_blocks) + len(selected_smaller_blocks)))
        match_matrix[:len(selected_bigger_blocks), :len(selected_bigger_blocks)] = 1
        match_matrix[len(selected_bigger_blocks):, len(selected_bigger_blocks):] = 1
        self.goals.append(
            (selected_bigger_blocks + selected_smaller_blocks,
             match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template.format(abs_pos=self.abs_position_category,
                                                         zone_color=zone_color_names[0], ))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlocksByAbsolutePositionAndColorAndSize(StackBlocks):
    """
    Task instruction:
    "Stack all the bigger/smaller blocks on the {abs_pos} area on the {color} zone."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.size_category, self.abs_position_category = None, None
        self.n_zones = None

        self.lang_template = ("stack all the {size_category} blocks "
                              "on the {abs_pos} on the {zone_color} zone.")
        self.task_completed_desc = "done stacking blocks."

    def reset(self, env):
        super().reset(env)

        self.abs_position_category = np.random.choice(
            a=["top left area", "top right area", "bottom left area", "bottom right area"],
            size=1,
        )[0]
        self.size_category = np.random.choice(a=["smaller", "bigger"], size=1)[0]

        # Add zones.
        self.n_zones = np.random.randint(2, 4)
        zone_color_names = np.random.choice(
            self.all_color_names, size=self.n_zones, replace=False
        )
        zone_colors = [utils.COLORS[cn] for cn in zone_color_names]

        # Add a zone
        target_zone_pose = None
        for i in range(self.n_zones):
            zone_size = (0.1, 0.1, 0)
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_colors[i] + [1])
            # Use the first zone as the target zone.
            if i == 0:
                target_zone_pose = zone_pose

        # Add blocks.
        objs = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf, bigger_block_urdf = ('stacking/block.urdf',
                                                 "stacking/bigger_block.urdf")
        n_bigger_blocks = np.random.randint(0, self.n_blocks + 1)
        # Use the first block as the reference block
        block_poses = []
        for i in range(self.n_blocks):
            if i < n_bigger_blocks:
                block_pose = self.get_random_pose(env, bigger_block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
            block_poses.append((block_id, block_pose))
            p.changeVisualShape(block_id, -1, rgbaColor=self.block_colors[i] + [1])
            objs.append((block_id, (np.pi / 2, None)))

        selected_blocks = []

        for i, (block_id, block_pose) in enumerate(block_poses):
            pos, rot = block_pose
            if (
                    (
                            self.area_boundary[self.abs_position_category]["x_start"]
                            <= pos[0] <=
                            self.area_boundary[self.abs_position_category]["x_end"]
                    )
                    and
                    (
                            self.area_boundary[self.abs_position_category]["y_start"]
                            <= pos[1] <=
                            self.area_boundary[self.abs_position_category]["y_end"]

                    )
            ):
                if (
                        (self.size_category == "bigger" and i < n_bigger_blocks)
                        or
                        (self.size_category == "smaller" and i >= n_bigger_blocks)
                ):
                    selected_blocks.append(objs[i])

        goal_poses = []
        pos, rot = target_zone_pose
        for i in range(len(selected_blocks)):
            if self.size_category == "bigger":
                height = bigger_block_size[2] / 2 + bigger_block_size[2] * i
            else:
                height = smaller_block_size[2] / 2 + smaller_block_size[2] * i
            goal_poses.append(((pos[0], pos[1], height), rot))

        match_matrix = np.ones((len(selected_blocks), len(selected_blocks)))
        self.goals.append(
            (selected_blocks, match_matrix, goal_poses, False, True, 'pose', None, 1)
        )
        self.lang_goals.append(self.lang_template.format(size_category=self.size_category,
                                                         abs_pos=self.abs_position_category,
                                                         zone_color=zone_color_names[0], ))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockInAbsoluteArea(Task):
    """Stack all the blocks in an absolute position area.
    Example: ``stack all the blocks in the top right area".
    NOTE: DISCARDED task.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.lang_template = "stack all the blocks in the {absolute_area}."
        self.task_completed_desc = "done stacking all the blocks."

    def reset(self, env):
        super().reset(env)
        abs_area_list = ["top left area", "top right area", "bottom left area",
                         "bottom right area"]
        abs_area = random.choice(abs_area_list)

        self.n_colors = np.random.randint(5, 8)
        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, self.n_colors)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.max_steps = self.n_colors + 2
        place_pos = abs_area

        self.place_obj_names, self.pick_obj_names = [], []
        place_obj_names_1, pick_obj_names_1 = [], []
        place_obj_names_2, pick_obj_names_2 = [], []
        self.task_name = "stack-block-in-absolute-area"

        # Get absolute position
        block_size = (0.04, 0.04, 0.04)
        bowl_size = (0.12, 0.12, 0)
        x_length = self.bounds[0][1] - self.bounds[0][0]
        y_length = self.bounds[1][1] - self.bounds[1][0]
        height = block_size[2] // 2
        center_pos = (self.bounds[0][0] + x_length / 2, self.bounds[1][0] + y_length / 2, height)
        # theta = np.random.rand() * 2 * np.pi
        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        # Define absolute position
        top_left_area = np.array([
            [self.bounds[0, 0], center_pos[0]],
            [self.bounds[1, 0], center_pos[1]],
        ])
        top_right_area = np.array([
            [self.bounds[0, 0], center_pos[0]],
            [center_pos[1], self.bounds[1, 1]],
        ])
        bottom_left_area = np.array([
            [center_pos[0], self.bounds[0, 1]],
            [self.bounds[1, 0], center_pos[1]],
        ])
        bottom_right_area = np.array([
            [center_pos[0], self.bounds[0, 1]],
            [center_pos[1], self.bounds[1, 1]],
        ])
        if place_pos == "center":
            x_start, x_end = center_pos[0], center_pos[0]
            y_start, y_end = center_pos[1], center_pos[1]
        elif place_pos == "top left area":
            x_start, x_end = top_left_area[0, 0], top_left_area[0, 1]
            y_start, y_end = top_left_area[1, 0], top_left_area[1, 1]
        elif place_pos == "top right area":
            x_start, x_end = top_right_area[0, 0], top_right_area[0, 1]
            y_start, y_end = top_right_area[1, 0], top_right_area[1, 1]
        elif place_pos == "bottom left area":
            x_start, x_end = bottom_left_area[0, 0], bottom_left_area[0, 1]
            y_start, y_end = bottom_left_area[1, 0], bottom_left_area[1, 1]
        elif place_pos == "bottom right area":
            x_start, x_end = bottom_right_area[0, 0], bottom_right_area[0, 1]
            y_start, y_end = bottom_right_area[1, 0], bottom_right_area[1, 1]
        else:
            raise ValueError("place_pos value is wrong!")

        # Add blocks
        blocks = []
        bigger_block_positions = []
        smaller_block_size = (0.04, 0.04, 0.04)
        smaller_block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        stacked_blocks, stacked_block_poses = [], []
        while len(stacked_blocks) < 3 or len(stacked_blocks) > 6:
            stacked_blocks, stacked_block_poses = [], []
            for i in range(2 * self.n_colors):
                if i % 2 == 0:
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(smaller_block_urdf, block_pose)
                else:
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(bigger_block_urdf, block_pose)

                if (
                        x_start <= block_pose[0][0] <= x_end
                        and
                        y_start <= block_pose[0][1] <= y_end
                ):
                    stacked_blocks.append((block_id, (0, None)))
                    stacked_block_poses.append(block_pose)
                    if i % 2 == 0:
                        pick_obj_names_1.append(f"smaller {selected_color_names[i // 2]} block")
                    else:
                        pick_obj_names_1.append(f"bigger {selected_color_names[i // 2]} block")
                p.changeVisualShape(block_id, -1, rgbaColor=colors[i // 2] + [1])
                # blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names_1[1:])
        self.place_obj_names.append(pick_obj_names_1[:-1])

        # Goal
        target_goal_positions = []
        for i in range(1, len(stacked_blocks)):
            goal_height = block_size[2] + block_size[2] * i
            target_goal_positions.append(
                (
                    (stacked_block_poses[0][0][0], stacked_block_poses[0][0][1], goal_height),
                    stacked_block_poses[0][1]
                )
            )

        # # Add distractor blocks
        # distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        # distractor_color = [utils.COLORS[c] for c in distractor_color_names]
        # n_distractors = min(np.random.randint(0, 3), len(distractor_color))
        # for i in range(n_distractors):
        #     is_bigger = np.random.rand() > 0.5
        #     if is_bigger:
        #         pose = self.get_random_pose(env, bigger_block_size)
        #         distractor_id = env.add_object(bigger_block_urdf, pose)
        #     else:
        #         pose = self.get_random_pose(env, smaller_block_size)
        #         distractor_id = env.add_object(smaller_block_urdf, pose)
        #     p.changeVisualShape(distractor_id, -1, rgbaColor=distractor_color[i] + [1])

        self.goals.append(
            (
                stacked_blocks[1:],
                np.eye(len(stacked_blocks) - 1),
                target_goal_positions,
                False,
                True,
                'pose',
                None,
                1
            )
        )
        self.lang_goals.append(self.lang_template.format(absolute_area=abs_area))

        block_color_list = selected_color_names[:self.n_colors]
        np.random.shuffle(block_color_list)
        self.scene_description = (f"On the table, there are {self.n_colors} blocks. "
                                  f"Their colors are {', '.join(block_color_list)}. ")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
