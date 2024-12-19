import random
from typing import List

import numpy as np
import pybullet as p

from cliport.tasks.color_reasoning import COLOR_CATEGORY_NAMES
from cliport.tasks.task import Task
from cliport.utils import utils


class MoveBlocks(Task):
    def __init__(self):
        super().__init__()

        self.objs = []
        self.smaller_block_size, self.bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        self.smaller_block_urdf, self.bigger_block_urdf = ("stacking/block.urdf",
                                                           "stacking/bigger_block.urdf")
        self.all_color_names = self.get_colors()
        self.all_block_colors = None
        # self.block_poses, self.block_pts = [], {}
        # self.block_color_names, self.block_colors, self.final_block_color_list = None, None, []
        # self.n_blocks, self.n_bigger_blocks = None, None
        # self.n_block_distractors, self.n_zone_distractors = None, None
        self.min_move, self.max_move = None, None

    def reset(self, env):
        super().reset(env)
        # self.input_manipulate_order = True
        self.consider_z_in_match = False
        self.min_move, self.max_move = 3, 5
        random.shuffle(self.all_color_names)
        self.all_block_colors = [utils.COLORS[cn] for cn in self.all_color_names]

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS

    # def add_blocks(self, env):
    #     self.block_poses, self.block_pts = [], {}
    #     self.block_color_names, self.block_colors, self.final_block_color_list = None, None, []
    #
    #     # TODO: implement this part.
    #     if self.task_difficulty_level == "easy":
    #         # No distractor blocks or zones, and no blocks with repetitive colors.
    #         self.n_blocks = np.random.randint(9, 20)
    #         self.block_color_names = np.random.choice(
    #             a=self.all_color_names, size=math.ceil(self.n_blocks / 2), replace=False
    #         )
    #     elif self.task_difficulty_level == "medium":
    #         # There exist distractor blocks and zones.
    #         self.n_blocks = np.random.randint(9, 20)
    #         self.block_color_names = np.random.choice(
    #             a=self.all_color_names, size=math.ceil(self.n_blocks / 2), replace=False
    #         )
    #
    #         self.n_block_distractors = np.random.randint(1, 3)
    #         self.n_zone_distractors = np.random.randint(1, 3)
    #     elif self.task_difficulty_level == "hard":
    #         # More blocks and their colors can be repetitive.
    #         self.n_blocks = np.random.randint(11, 25)
    #         self.block_color_names = np.random.choice(
    #             a=self.all_color_names, size=self.n_blocks, replace=True
    #         )
    #     else:
    #         raise ValueError(f"Unknown task_difficulty_level: {self.task_difficulty_level}")
    #
    #     self.block_colors = [utils.COLORS[cn] for cn in self.block_color_names]
    #     self.max_steps = self.n_blocks + 2
    #
    #     if self.task_difficulty_level == "hard":
    #         self.n_bigger_blocks = np.random.randint(0, len(self.n_blocks) + 1)
    #     else:
    #         # To ensure there are no repetitive colors for the blocks
    #         # with the same size for non-hard task.
    #         self.n_bigger_blocks = np.random.randint(
    #             self.n_blocks - len(self.block_colors), len(self.block_colors) + 1
    #         )
    #
    #     print(f"size of all_color_names: {len(self.all_color_names)}, "
    #           f"size of n_blocks: {self.n_blocks}, "
    #           f"number of bigger_blocks: {self.n_bigger_blocks}")
    #     # Add blocks.
    #     for i in range(self.n_blocks):
    #         if i < self.n_bigger_blocks:
    #             block_pose = self.get_random_pose(env, self.bigger_block_size)
    #             block_id = env.add_object(self.bigger_block_urdf, block_pose)
    #         else:
    #             block_pose = self.get_random_pose(env, self.smaller_block_size)
    #             block_id = env.add_object(self.smaller_block_urdf, block_pose)
    #         self.block_poses.append((block_id, block_pose))
    #         self.block_pts[block_id] = self.get_box_object_points(block_id)
    #         if self.task_difficulty_level == "hard":
    #             color_idx = i
    #             p.changeVisualShape(block_id, -1,
    #                                 rgbaColor=self.block_colors[color_idx] + [1])
    #         else:
    #             color_idx = i if i < self.n_bigger_blocks else i - self.n_bigger_blocks
    #             p.changeVisualShape(block_id, -1,
    #                                 rgbaColor=self.block_colors[color_idx] + [1])
    #         self.final_block_color_list.append(self.block_color_names[color_idx])
    #         self.objs.append((block_id, (np.pi / 2, None)))

    def is_pose_in_area(self, pose, area):
        pos, _ = pose
        if (
                (
                        self.area_boundary[area]["x_start"]
                        <= pos[0] <=
                        self.area_boundary[area]["x_end"]
                )
                and
                (
                        self.area_boundary[area]["y_start"]
                        <= pos[1] <=
                        self.area_boundary[area]["y_end"]
                )
        ):
            return True
        else:
            return False

    def random_pose_in_area(self, env, area, obj_size):
        pose = self.get_random_pose(env, obj_size)
        while not self.is_pose_in_area(pose, area):
            pose = self.get_random_pose(env, obj_size)
        return pose


class MoveBlocksBetweenAbsolutePositions(MoveBlocks):
    """Move all the blocks in the [X] area to [Y] area.
    X and Y are sampled from [top left, top right, bottom left, bottom right]
    Example: ``Move all the blocks in the top left area to bottom right area".
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.src_abs_pos, self.tgt_abs_pos = None, None
        self.lang_template = "move all the blocks on the {src_abs_pos} to the {tgt_abs_pos}."
        self.task_completed_desc = "done moving all the blocks."
        self.obj = {}
        self.goal = "Move all the blocks in the {X} to {Y}."

    def reset(self, env):
        super().reset(env)
        n_move = np.random.randint(self.min_move, self.max_move)
        self.print_debug_info = True

        areas = ["top left area", "top right area", "bottom left area", "bottom right area"]

        n_blocks_area = {a: 0 for a in areas}
        objs_area = {a: [] for a in areas}
        block_pts = {}
        max_area_blocks = 0
        color_idx_for_bigger, color_idx_for_smaller = 0, 0
        final_block_color_list = []
        while max_area_blocks < n_move:
            is_bigger_block = np.random.choice([True, False])
            if is_bigger_block:
                block_size = self.bigger_block_size
                block_urdf = self.bigger_block_urdf
                color_idx_for_bigger += 1
                color_idx = color_idx_for_bigger
            else:
                block_size = self.smaller_block_size
                block_urdf = self.smaller_block_urdf
                color_idx_for_smaller += 1
                color_idx = color_idx_for_smaller
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)

            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=self.all_block_colors[color_idx % len(self.all_block_colors)] + [1]
            )
            final_block_color_list.append(
                self.all_color_names[color_idx % len(self.all_color_names)]
            )

            block_pts[block_id] = self.get_box_object_points(block_id)
            self.obj[block_id] = {
                'color': self.all_color_names[color_idx % len(self.all_color_names)],
                'shape': 'block'}

            for area in areas:
                if self.is_pose_in_area(block_pose, area):
                    n_blocks_area[area] += 1
                    objs_area[area].append((block_id, (np.pi / 2, None)))
                    break
            max_area_blocks = max(n_blocks_area.values())
        src_area = max(n_blocks_area, key=n_blocks_area.get)
        tgt_area = min(n_blocks_area, key=n_blocks_area.get)

        tgt_poses = []
        for i in range(n_move):
            tgt_pose = self.random_pose_in_area(env, tgt_area, 2 * self.bigger_block_size)
            tgt_poses.append(tgt_pose)

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        # Goal
        pts = {k[0]: block_pts[k[0]] for k in objs_area[src_area]}

        tgt_area_pose = (
            (
                self.area_boundary[tgt_area]["x_start"] + self.x_length / 4,
                self.area_boundary[tgt_area]["y_start"] + self.y_length / 4,
                0
            ), rot
        )
        tgt_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)

        self.goals.append(
            (objs_area[src_area],
             np.eye(len(objs_area[src_area])),
             tgt_poses,
             False, True, 'zone',
             (pts, [(tgt_area_pose, tgt_area_size)]), 1)
        )
        self.lang_goals.append(self.lang_template.format(src_abs_pos=src_area,
                                                         tgt_abs_pos=tgt_area))
        self.goal = self.goal.format(X=src_area, Y=tgt_area)

        # block_color_list = self.block_color_names[:self.n_colors]
        # np.random.shuffle(block_color_list)
        self.scene_description = (f"On the table, there are {len(block_pts)} blocks. "
                                  f"Their colors are {', '.join(final_block_color_list)}. "
                                  f"The sizes of blocks are different. "
                                  f"Some blocks are 'bigger' and some blocks are 'smaller'.")

    # def reset(self, env):
    #     super().reset(env)
    #     self.print_debug_info = True
    #
    #     # print(f"size of all_color_names: {len(self.all_color_names)}, "
    #     #       f"size of n_blocks: {self.n_blocks}, "
    #     #       f"size of block_poses: {len(self.block_poses)}, "
    #     #       f"number of bigger_blocks: {self.n_bigger_blocks}")
    #
    #     selected_blocks = []
    #     target_poses = []
    #     n_try = 0
    #     while len(selected_blocks) < self.min_move and n_try < 7:
    #         n_try += 1
    #         selected_blocks = []
    #         target_poses = []
    #         # print("enter while loop +1")
    #         # env_copy = copy.deepcopy(env)
    #         self.src_abs_pos, self.tgt_abs_pos = np.random.choice(
    #             a=["top left area", "top right area", "bottom left area", "bottom right area"],
    #             size=2,
    #             replace=False,
    #         )
    #
    #         for i, (_, block_pose) in enumerate(self.block_poses):
    #             _, rot = block_pose
    #             if self.is_pose_in_area(block_pose, self.src_abs_pos):
    #                 selected_blocks.append(self.objs[i])
    #                 if i < self.n_bigger_blocks:
    #                     target_pose = self.random_pose_in_area(
    #                         env, self.tgt_abs_pos, 2 * self.bigger_block_size
    #                     )
    #                 else:
    #                     target_pose = self.random_pose_in_area(
    #                         env, self.tgt_abs_pos, 2 * self.smaller_block_size
    #                     )
    #                 target_poses.append(target_pose)
    #                 # target_zones.append((target_pose, target_area_size))
    #
    #     theta = 0
    #     rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
    #
    #     # Goal
    #     pts = {k[0]: self.block_pts[k[0]] for k in selected_blocks}
    #
    #     if self.print_debug_info:
    #         print(f"Size of selected_blocks: {len(selected_blocks)}, "
    #               f"Size of pts: {len(pts)}")
    #
    #     target_area_pose = (
    #         (
    #             self.area_boundary[self.tgt_abs_pos]["x_start"] + self.x_length / 4,
    #             self.area_boundary[self.tgt_abs_pos]["y_start"] + self.y_length / 4,
    #             0
    #         ), rot
    #     )
    #     target_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)
    #
    #     self.goals.append(
    #         (selected_blocks,
    #          np.eye(len(selected_blocks)),
    #          target_poses,
    #          False, True, 'zone',
    #          (pts, [(target_area_pose, target_area_size)]), 1)
    #     )
    #     self.lang_goals.append(self.lang_template.format(src_abs_pos=self.src_abs_pos,
    #                                                      tgt_abs_pos=self.tgt_abs_pos, ))
    #
    #     block_color_list = self.block_color_names[:self.n_colors]
    #     np.random.shuffle(block_color_list)
    #     self.scene_description = (f"On the table, there are {self.n_blocks} blocks. "
    #                               f"Their colors are {', '.join(self.final_block_color_list)}. "
    #                               f"The sizes of blocks are different. "
    #                               f"Some blocks are 'bigger' and some blocks are 'smaller'.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class MoveBlocksBetweenAbsolutePositionsBySize(MoveBlocks):
    """
    Task instruction:
    "Move all the bigger/smaller blocks
    on the {source_abs_pos} area to the {target_abs_pos} area."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.src_abs_pos, self.tgt_abs_pos, self.size_category = None, None, None
        self.lang_template = ("move all the {size_category} blocks "
                              "on the {src_abs_pos} to the {tgt_abs_pos}.")
        self.task_completed_desc = "done moving blocks."
        self.obj = {}
        self.goal = 'Move all of {size_category} blocks, which are on the {src_abs_pos}, to the {tgt_abs_pos}.'

    def reset(self, env):
        super().reset(env)
        n_move = np.random.randint(self.min_move, self.max_move)
        self.print_debug_info = True

        areas = ["top left area", "top right area", "bottom left area", "bottom right area"]

        n_blocks_area = {a: {"bigger": 0, "smaller": 0} for a in areas}
        objs_area = {a: {"bigger": [], "smaller": []} for a in areas}
        block_pts = {}
        max_area_blocks = 0
        color_idx_for_bigger, color_idx_for_smaller = 0, 0
        final_block_color_list = []
        src_area, src_size = None, None
        sample_more_bigger = np.random.choice([True, False])
        while max_area_blocks < n_move:
            if sample_more_bigger:
                is_bigger_block = np.random.choice([True, False, True, True, True])
            else:
                is_bigger_block = np.random.choice([False, True, False, False, False])
            if is_bigger_block:
                block_size = self.bigger_block_size
                block_urdf = self.bigger_block_urdf
                color_idx_for_bigger += 1
                color_idx = color_idx_for_bigger
                size_category = "bigger"
            else:
                block_size = self.smaller_block_size
                block_urdf = self.smaller_block_urdf
                color_idx_for_smaller += 1
                color_idx = color_idx_for_smaller
                size_category = "smaller"
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            self.obj[block_id] = {
                'color': self.all_color_names[color_idx % len(self.all_color_names)],
                'shape': 'big block'} if is_bigger_block else {
                'color': self.all_color_names[color_idx % len(self.all_color_names)],
                'shape': 'small block'}
            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=self.all_block_colors[color_idx % len(self.all_block_colors)] + [1]
            )
            final_block_color_list.append(
                self.all_color_names[color_idx % len(self.all_color_names)]
            )

            block_pts[block_id] = self.get_box_object_points(block_id)

            for area in areas:
                if self.is_pose_in_area(block_pose, area):
                    n_blocks_area[area][size_category] += 1
                    objs_area[area][size_category].append((block_id, (np.pi / 2, None)))
                    break
            src_area, src_size, max_area_blocks = max(
                [(k, kk, vv) for k, v in n_blocks_area.items() for kk, vv in v.items()],
                key=lambda x: x[2]
            )
        tgt_area, _ = min(
            [(k, sum(v.values())) for k, v in n_blocks_area.items()],
            key=lambda x: x[1]
        )

        tgt_poses = []
        for i in range(n_move):
            tgt_pose = self.random_pose_in_area(env, tgt_area, 2 * self.bigger_block_size)
            tgt_poses.append(tgt_pose)

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        # Goal
        pts = {k[0]: block_pts[k[0]] for k in objs_area[src_area][src_size]}

        tgt_area_pose = (
            (
                self.area_boundary[tgt_area]["x_start"] + self.x_length / 4,
                self.area_boundary[tgt_area]["y_start"] + self.y_length / 4,
                0
            ), rot
        )
        tgt_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)

        self.goals.append(
            (objs_area[src_area][src_size],
             np.eye(len(objs_area[src_area][src_size])),
             tgt_poses,
             False, True, 'zone',
             (pts, [(tgt_area_pose, tgt_area_size)]), 1)
        )
        self.lang_goals.append(self.lang_template.format(size_category=src_size,
                                                         src_abs_pos=src_area,
                                                         tgt_abs_pos=tgt_area))
        self.goal = self.goal.format(size_category=src_size, src_abs_pos=src_area, tgt_abs_pos=tgt_area)

        self.scene_description = (f"On the table, there are {len(block_pts)} blocks. "
                                  f"Their colors are {', '.join(final_block_color_list)}. "
                                  f"The sizes of blocks are different. "
                                  f"Some blocks are 'bigger' and some blocks are 'smaller'.")


class MoveBlocksBetweenAbsolutePositionsBySizeAndColor(MoveBlocks):
    """
    Task instruction:
    "Move all the bigger/smaller blocks of primary/secondary/warm/cool color
    on the {source_abs_pos} area to the {target_abs_pos} area."
    Note each block's color is unique thus does not require reference capability.
    """

    def __init__(self):
        super().__init__()

        self.src_abs_pos, self.tgt_abs_pos = None, None
        self.size_category, self.color_category = None, None
        self.lang_template = ("move all the {size_category} blocks of {color_category} color "
                              "on the {src_abs_pos} to the {tgt_abs_pos}.")
        self.task_completed_desc = "done moving blocks."

    def reset(self, env):
        super().reset(env)
        self.min_move, self.max_move = 2, 4
        n_move = np.random.randint(self.min_move, self.max_move)
        self.print_debug_info = True

        areas = ["top left area", "top right area", "bottom left area", "bottom right area"]
        size_categories = ["smaller", "bigger"]
        color_categories = ["primary", "secondary", "warm", "cool"]

        n_blocks_area = {
            a: {
                size: {c.upper(): 0 for c in color_categories} for size in size_categories
            } for a in areas
        }
        objs_area = {
            a: {
                size: {c.upper(): [] for c in color_categories} for size in size_categories
            } for a in areas
        }
        block_pts = {}
        max_area_blocks = 0
        color_idx_for_bigger, color_idx_for_smaller = 0, 0
        final_block_color_list = []
        src_area, src_size_cty, src_color_cty = None, None, None
        sample_more_bigger = np.random.choice([False, True])
        sample_more_color_category = np.random.choice(color_categories)
        while max_area_blocks < n_move:
            if sample_more_bigger:
                is_bigger_block = np.random.choice([True, False, True, True, True])
            else:
                is_bigger_block = np.random.choice([False, True, False, False, False])
            is_bigger_block = np.random.choice([True, False])
            if is_bigger_block:
                block_size = self.bigger_block_size
                block_urdf = self.bigger_block_urdf
                color_idx_for_bigger += 1
                color_idx = color_idx_for_bigger
                size_category = "bigger"
            else:
                block_size = self.smaller_block_size
                block_urdf = self.smaller_block_urdf
                color_idx_for_smaller += 1
                color_idx = color_idx_for_smaller
                size_category = "smaller"
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)

            p.changeVisualShape(
                block_id,
                -1,
                rgbaColor=self.all_block_colors[color_idx % len(self.all_block_colors)] + [1]
            )
            final_block_color_list.append(
                self.all_color_names[color_idx % len(self.all_block_colors)]
            )

            block_pts[block_id] = self.get_box_object_points(block_id)

            for area in areas:
                if self.is_pose_in_area(block_pose, area):
                    color_category = self.classify_color(
                        self.all_color_names[color_idx % len(self.all_block_colors)]
                    )
                    if len(color_category) > 0:
                        for c in color_category:
                            n_blocks_area[area][size_category][c] += 1
                            objs_area[area][size_category][c].append(
                                (block_id, (np.pi / 2, None))
                            )
                    break
            src_area, src_size_cty, src_color_cty, max_area_blocks = max(
                [(k, kk, kkk, vvv) for k, v in n_blocks_area.items()
                 for kk, vv in v.items() for kkk, vvv in vv.items()],
                key=lambda x: x[3]
            )
        tgt_area, _ = min(
            [(k, sum(vv.values())) for k, v in n_blocks_area.items() for kk, vv in v.items()],
            key=lambda x: x[1]
        )

        tgt_poses = []
        for i in range(n_move):
            tgt_pose = self.random_pose_in_area(env, tgt_area, 2 * self.bigger_block_size)
            tgt_poses.append(tgt_pose)

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

        # Goal
        pts = {k[0]: block_pts[k[0]] for k in objs_area[src_area][src_size_cty][src_color_cty]}

        tgt_area_pose = (
            (
                self.area_boundary[tgt_area]["x_start"] + self.x_length / 4,
                self.area_boundary[tgt_area]["y_start"] + self.y_length / 4,
                0
            ), rot
        )
        tgt_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)

        self.goals.append(
            (objs_area[src_area][src_size_cty][src_color_cty],
             np.eye(len(objs_area[src_area][src_size_cty][src_color_cty])),
             tgt_poses,
             False, True, 'zone',
             (pts, [(tgt_area_pose, tgt_area_size)]), 1)
        )
        self.lang_goals.append(self.lang_template.format(size_category=src_size_cty,
                                                         color_category=src_color_cty,
                                                         src_abs_pos=src_area,
                                                         tgt_abs_pos=tgt_area))

        self.scene_description = (f"On the table, there are {len(block_pts)} blocks. "
                                  f"Their colors are {', '.join(final_block_color_list)}. "
                                  f"The sizes of blocks are different. "
                                  f"Some blocks are 'bigger' and some blocks are 'smaller'.")

        # super().reset(env)
        #
        # selected_blocks = []
        # target_poses = []
        # while len(selected_blocks) < self.min_move:
        #     selected_blocks = []
        #     target_poses = []
        #     self.src_abs_pos, self.tgt_abs_pos = np.random.choice(
        #         a=["top left area", "top right area", "bottom left area", "bottom right area"],
        #         size=2,
        #         replace=False,
        #     )
        #     self.size_category = np.random.choice(a=["smaller", "bigger"], size=1)[0]
        #     self.color_category = np.random.choice(
        #         a=["primary", "secondary", "warm", "cool"], size=1
        #     )[0]
        #     selected_color_names = COLOR_CATEGORY_NAMES[self.color_category.upper()]
        #
        #     for i, (_, block_pose) in enumerate(self.block_poses):
        #         _, rot = block_pose
        #         if self.is_pose_in_area(block_pose, self.src_abs_pos):
        #             color_idx = i if i < self.n_bigger_blocks else i - self.n_bigger_blocks
        #             if (
        #                     self.block_color_names[color_idx]
        #                     in
        #                     selected_color_names
        #             ):
        #                 if (
        #                         (self.size_category == "bigger")
        #                         and
        #                         (i < self.n_bigger_blocks)
        #                 ):
        #                     selected_blocks.append(self.objs[i])
        #                     target_pose = self.random_pose_in_area(
        #                         env, self.tgt_abs_pos, 2 * self.bigger_block_size
        #                     )
        #                 elif (
        #                         (self.size_category == "smaller")
        #                         and
        #                         (i >= self.n_bigger_blocks)
        #                 ):
        #                     selected_blocks.append(self.objs[i])
        #                     target_pose = self.random_pose_in_area(
        #                         env, self.tgt_abs_pos, 2 * self.smaller_block_size
        #                     )
        #                 else:
        #                     continue
        #                 target_poses.append(target_pose)
        #             else:
        #                 continue
        #
        # theta = 0
        # rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        # target_area_pose = (
        #     (
        #         self.area_boundary[self.tgt_abs_pos]["x_start"] + self.x_length / 4,
        #         self.area_boundary[self.tgt_abs_pos]["y_start"] + self.y_length / 4,
        #         0
        #     ), rot
        # )
        # target_area_size = (self.x_length / 2 * 1.2, self.y_length / 2 * 1.2, 0)
        #
        # # Goal
        # pts = {k[0]: self.block_pts[k[0]] for k in selected_blocks}
        # self.goals.append((selected_blocks, np.eye(len(selected_blocks)), target_poses,
        #                    False, True, 'zone',
        #                    (pts, [(target_area_pose, target_area_size)]), 1))
        # self.lang_goals.append(self.lang_template.format(size_category=self.size_category,
        #                                                  color_category=self.color_category,
        #                                                  src_abs_pos=self.src_abs_pos,
        #                                                  tgt_abs_pos=self.tgt_abs_pos, ))

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS

    @staticmethod
    def classify_color(color: str) -> List[str]:
        categories = []
        for category, colors in COLOR_CATEGORY_NAMES.items():
            if color in colors:
                categories.append(category)
        return categories
