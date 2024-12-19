import random
from typing import List, Set, Union

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class PickAndPlacePrimitive(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    Pick up the [block1] and place it on the [block2/bowl/zone].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick_color} block and place it on the {place_color} {place}."
        self.task_completed_desc = "done placing blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        target_idx = random.randint(0, 2)
        target_objs = ["block", "bowl", "zone"]
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_zones = np.random.randint(1, 3)
        n_bowls = np.random.randint(2, 4)

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)
        zone_colors = random.sample(all_color_names, n_zones)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]
        zone_util_colors = [utils.COLORS[cn] for cn in zone_colors]

        self.scene_description = f"On the table, there are {n_blocks} blocks. Their colors are {block_colors}. " \
                                 f"There are {n_zones} zones. Their colors are {zone_colors}. " \
                                 f"There are {n_bowls} bowls. Their colors are {bowl_colors}. "

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        place_pose = None
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                block_pts[block_id] = self.get_box_object_points(block_id)
            elif i == 1:
                place_pose = block_pose
            if target_idx == 1:
                blocks.append((block_id, (0, None)))
            else:
                blocks.append((block_id, (np.pi / 2, None)))

        # Add zones.
        zone_size = (0.1, 0.1, 0)
        zone_poses = []
        for i in range(n_zones):
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_util_colors[i] + [1])
            zone_poses.append(zone_pose)
            # zone_poses.append((zone_obj_id, (0, None)))

        if target_objs[target_idx] == "block":
            target_pose = (
                (place_pose[0][0], place_pose[0][1], block_size[2] * 2),
                place_pose[1]
            )
            self.goals.append((blocks[:1], np.eye(1),
                               [target_pose], False, True, 'pose', None, 1))
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=block_colors[1],
                                                             place=target_objs[target_idx]))
        elif target_objs[target_idx] == "bowl":
            # target_matrix = np.zeros((n_blocks, n_bowls))
            # target_matrix[0, 0] = 1
            # print(target_matrix)
            target_matrix = np.ones((1, 1))
            self.goals.append(([blocks[0]], target_matrix,
                               [bowl_poses[0]], False, True, 'pose', None, 1))
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=bowl_colors[0],
                                                             place=target_objs[target_idx]))
        else:
            target_matrix = np.ones((1, 1))
            self.goals.append(([blocks[0]], target_matrix,
                               [zone_poses[0]], True, False, 'zone',
                               (block_pts, [(zone_poses[0], zone_size)]), 1))
            self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                             place_color=zone_colors[0],
                                                             place=target_objs[target_idx]))

    def get_colors(self) -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithSize(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate two sizes: bigger and smaller.
    Pick up the [bigger/smaller] [block1] and place it on the [bigger/smaller] [block2/bowl/zone].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_size} {pick_color} block and "
                              "place it on the {place_size} {place_color} block.")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_bowls = np.random.randint(2, 4)
        n_bigger_blocks = np.random.randint(2, n_blocks + 1)
        bigger_block_indexes = np.random.choice(
            list(range(n_blocks + n_bigger_blocks)),
            size=n_bigger_blocks,
            replace=False
        )

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)

        # Add bigger block and bowl colors, their colors are the same as smaller objects' colors,
        # bigger blocks and bowls are at the beginning part
        bigger_block_colors = block_colors[:n_bigger_blocks]
        for i, idx in enumerate(bigger_block_indexes):
            block_colors.insert(idx, bigger_block_colors[i])

        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]

        self.scene_description = f"On the table, there are {n_blocks + n_bigger_blocks} blocks. " \
                                 f"Their colors are {block_colors}. " \
                                 f"There are {n_bowls} bowls. Their colors are {bowl_colors}. "

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            if i == 0:  # Always choose the first bowl as the target bowl
                place_bowl_pose = bowl_pose
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        pick_obj_pose, place_obj_pose = None, None
        pick_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)
        place_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)
        while place_block_idx == pick_block_idx:
            place_block_idx = np.random.randint(0, n_blocks + n_bigger_blocks)

        for i in range(n_blocks + n_bigger_blocks):
            if i in bigger_block_indexes:
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])

            if i == pick_block_idx:
                pick_obj_pose = block_pose
            elif i == place_block_idx:
                block_pts[block_id] = self.get_box_object_points(block_id)
                place_obj_pose = block_pose
            blocks.append((block_id, (np.pi / 2, None)))

        pick_size_desc = "bigger" if pick_block_idx in bigger_block_indexes else "smaller"
        place_block_size_desc = "bigger" if place_block_idx in bigger_block_indexes else "smaller"

        target_pos = (
            place_obj_pose[0][0],
            place_obj_pose[0][1],
            place_obj_pose[0][2] + pick_obj_pose[0][2] // 2
        )
        self.goals.append(([blocks[pick_block_idx]], np.eye(1),
                           [(target_pos, place_obj_pose[1])], False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(
            pick_size=pick_size_desc,
            pick_color=block_colors[pick_block_idx],
            place_size=place_block_size_desc,
            place_color=block_colors[place_block_idx],
        ))

    def get_colors(self) -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithRelativePosition(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate the relative position: top, down, left, right.
    Pick up the [block1] and place it on the [top/bottom/left/right of] [block2/bowl/zone].
    If the block2's rotation is not horizontal or vertical, the robot should change its rotation first.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = ("pick up the {pick_color} block and "
                              "place it on the {place_relative_pos} of {place_color} block")
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_zones = np.random.randint(1, 3)
        n_bowls = np.random.randint(2, 4)
        rel_pos_list = ["top", "bottom", "left", "right"]
        rel_pos = random.choice(rel_pos_list)
        # n_blocks = 4
        # n_zones = 3
        # n_bowls = 4

        block_colors = random.sample(all_color_names, n_blocks)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]

        self.scene_description = f"On the table, there are {n_blocks} blocks. " \
                                 f"Their colors are {block_colors}. "

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        pick_obj_pose, place_obj_pose = None, None
        if n_blocks < 2:
            raise ValueError("Too few blocks! Please increase the number of blocks!")
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                pick_obj_pose = block_pose
            elif i == 1:
                block_pts[block_id] = self.get_box_object_points(block_id)
                place_obj_pose = block_pose
            blocks.append((block_id, (np.pi / 2, None)))

        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        place_obj_pose = (np.asarray(place_obj_pose[0]) + np.random.uniform(0.01, 0.02), rot)

        # Target pose
        # TODO: check whether the pos is correct.
        if rel_pos == "left":
            target_pos = (
                place_obj_pose[0][0],
                place_obj_pose[0][1] - block_size[0] * 4,
                place_obj_pose[0][2]
            )
        elif rel_pos == "right":
            target_pos = (
                place_obj_pose[0][0],
                place_obj_pose[0][1] + block_size[0] * 4,
                place_obj_pose[0][2]
            )
        elif rel_pos == "top":
            target_pos = (
                place_obj_pose[0][0] - block_size[1] * 4,
                place_obj_pose[0][1],
                place_obj_pose[0][2]
            )
        elif rel_pos == "bottom":
            target_pos = (
                place_obj_pose[0][0] + block_size[1] * 4,
                place_obj_pose[0][1],
                place_obj_pose[0][2]
            )
        else:
            raise ValueError("The value of rel_pos is wrong. Please check!")

        src = [blocks[1], blocks[0]]
        tgt = [place_obj_pose, (target_pos, place_obj_pose[1])]
        self.goals.append((src, np.eye(2),
                           tgt, False, False, 'pose', None, 1))
        # self.goals.append((blocks[:1], np.eye(1),
        #                    [(target_pos, place_obj_pose[1])], False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(
            pick_color=block_colors[0],
            place_relative_pos=rel_pos,
            place_color=block_colors[1],
        ))

    def get_colors(self) -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)


class PickAndPlacePrimitiveWithAbsolutePosition(Task):
    """
    Pick-and-place primitive for the LLM planner.
    This primitive is trained with all the colors, following the setting of `Inner Monologue`.
    In addition, this primitive is trained to discriminate the absolute position: center of the table.
    Pick up the [block1] and place it on the [place_pos].
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick_color} block and place it on the {place_pos}"
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        all_color_names = self.get_colors()

        n_blocks = np.random.randint(2, 4)
        n_bowls = np.random.randint(2, 4)
        place_pos = random.choice(
            ["center", "top left area", "top right area", "bottom left area", "bottom right area"]
        )

        block_colors = random.sample(all_color_names, n_blocks)
        bowl_colors = random.sample(all_color_names, n_bowls)
        block_util_colors = [utils.COLORS[cn] for cn in block_colors]
        bowl_util_colors = [utils.COLORS[cn] for cn in bowl_colors]

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

        random_area_pose = [
            (
                np.random.uniform(x_start, x_end),
                np.random.uniform(y_start, y_end),
                height
            ),
            rot
        ]

        # Add bowls.
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_util_colors[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_pts = {}
        block_urdf = 'stacking/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                block_pts[block_id] = self.get_box_object_points(block_id)
                base_pose = block_pose
            blocks.append((block_id, (np.pi / 2, None)))

        self.goals.append((blocks[:1], np.eye(1),
                           [random_area_pose], False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(pick_color=block_colors[0],
                                                         place_pos=place_pos))

        self.scene_description = f"On the table, there are {n_blocks} blocks. Their colors are {block_colors}. " \
                                 f"There are {n_bowls} bowls. Their colors are {bowl_colors}. "

    def get_colors(self) -> Union[List[str], Set[str]]:
        return set(utils.TRAIN_COLORS + utils.EVAL_COLORS)
