import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class StackSmallerOverBiggerWithSameColor(Task):
    """Stack smaller blocks over bigger blocks of the same color."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.lang_template = ("stack all the smaller blocks over the bigger blocks of "
                              "the same color.")
        self.task_completed_desc = "done stacking all the blocks."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        self.n_colors = np.random.randint(5, 8)
        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, self.n_colors)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.max_steps = self.n_colors + 2

        self.place_obj_names, self.pick_obj_names = [], []
        place_obj_names, pick_obj_names = [], []
        self.task_name = "stack-smaller-over-bigger-with-same-color"

        # Add blocks
        blocks = []
        bigger_block_positions = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        for i in range(2 * self.n_colors):
            if i % 2 == 0:  # bigger blocks
                block_pose = self.get_random_pose(env, bigger_block_size)  # (pos, rot)
                bigger_block_positions.append(block_pose)
                block_id = env.add_object(bigger_block_urdf, block_pose)
                place_obj_names.append(f"bigger {selected_color_names[i // 2]} block")
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
                pick_obj_names.append(f"smaller {selected_color_names[i // 2]} block")
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i // 2] + [1])
            blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names)
        self.place_obj_names.append(place_obj_names)

        # Goal
        smaller_block_goal_positions = []
        goal_height = bigger_block_size[2] + smaller_block_size[2] // 2
        for pos, rot in bigger_block_positions:
            smaller_block_goal_positions.append(
                (
                    (pos[0], pos[1], goal_height),
                    rot
                )
            )

        # Add distractor blocks
        distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_color = [utils.COLORS[c] for c in distractor_color_names]
        n_distractors = min(np.random.randint(0, 3), len(distractor_color))
        for i in range(n_distractors):
            is_bigger = np.random.rand() > 0.5
            if is_bigger:
                pose = self.get_random_pose(env, bigger_block_size)
                distractor_id = env.add_object(bigger_block_urdf, pose)
            else:
                pose = self.get_random_pose(env, smaller_block_size)
                distractor_id = env.add_object(smaller_block_urdf, pose)
            p.changeVisualShape(distractor_id, -1, rgbaColor=distractor_color[i] + [1])

        self.goals.append(
            (
                blocks[1::2],
                np.eye(self.n_colors),
                smaller_block_goal_positions,
                False,
                True,
                'pose',
                None,
                1
            )
        )
        self.lang_goals.append(self.lang_template)

        block_color_list = selected_color_names[:self.n_colors] + distractor_color_names[
                                                                  :n_distractors]
        np.random.shuffle(block_color_list)
        self.scene_description = (
            f"On the table, there are {2 * self.n_colors + n_distractors} blocks. "
            f"Their colors are {', '.join(block_color_list)}. "
            f"The size of the blocks are different. "
            f"Some blocks are 'bigger' and some blocks are 'smaller'.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackSmallerOverBiggerWithSameColorInSameColorZone(Task):
    """Stack smaller blocks over bigger blocks of the same color in the same color zone."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.lang_template = ("stack all the smaller blocks over the bigger blocks of "
                              "the same color in the zone with the same color.")
        self.task_completed_desc = "done stacking all the blocks."

    def reset(self, env):
        super().reset(env)

        self.n_colors = np.random.randint(3, 6)
        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, self.n_colors)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.max_steps = 2 * self.n_colors + 3

        self.place_obj_names, self.pick_obj_names = [], []
        place_obj_names_1, pick_obj_names_1 = [], []
        place_obj_names_2, pick_obj_names_2 = [], []
        self.task_name = "stack-smaller-over-bigger-with-same-color-in-same-color-zone"

        # Add blocks
        blocks = []
        bigger_block_positions = []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        for i in range(2 * self.n_colors):
            if i % 2 == 0:  # bigger blocks
                block_pose = self.get_random_pose(env, bigger_block_size)  # (pos, rot)
                bigger_block_positions.append(block_pose)
                block_id = env.add_object(bigger_block_urdf, block_pose)
                pick_obj_names_1.append(f"bigger {selected_color_names[i // 2]} block")
                place_obj_names_2.append(f"bigger {selected_color_names[i // 2]} block")
            else:
                block_pose = self.get_random_pose(env, smaller_block_size)
                block_id = env.add_object(smaller_block_urdf, block_pose)
                pick_obj_names_2.append(f"smaller {selected_color_names[i // 2]} block")
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i // 2] + [1])
            blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names_1)
        self.pick_obj_names.append(pick_obj_names_2)

        # Add zone
        zone_size = (0.15, 0.15, 0)
        zone_poses = []
        for i in range(self.n_colors):
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            while not zone_obj_id:
                zone_size = (zone_size[0] - 0.01, zone_size[1] - 0.01, 0)
                zone_pose = self.get_random_pose(env, zone_size)
                zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            place_obj_names_1.append(f"{selected_color_names[i]} zone")

            p.changeVisualShape(zone_obj_id, -1, rgbaColor=colors[i] + [1])
            zone_poses.append(zone_pose)
        self.place_obj_names.append(place_obj_names_1)
        self.place_obj_names.append(place_obj_names_2)

        # Goal
        bigger_block_goal_positions = []
        for pos, rot in zone_poses:
            bigger_block_goal_positions.append(
                (
                    (pos[0], pos[1], bigger_block_size[2]),
                    rot
                )
            )
        smaller_block_goal_positions = []
        goal_height = bigger_block_size[2] + smaller_block_size[2]
        for pos, rot in bigger_block_goal_positions:
            smaller_block_goal_positions.append(
                (
                    (pos[0], pos[1], goal_height),
                    rot
                )
            )

        # Add distractor blocks
        distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_color = [utils.COLORS[c] for c in distractor_color_names]
        n_distractors = min(np.random.randint(0, 3), len(distractor_color))
        for i in range(n_distractors):
            is_bigger = np.random.rand() > 0.5
            if is_bigger:
                pose = self.get_random_pose(env, bigger_block_size)
                distractor_id = env.add_object(bigger_block_urdf, pose)
            else:
                pose = self.get_random_pose(env, smaller_block_size)
                distractor_id = env.add_object(smaller_block_urdf, pose)
            p.changeVisualShape(distractor_id, -1, rgbaColor=distractor_color[i] + [1])

        self.goals.append(
            (
                blocks[::2],
                np.eye(self.n_colors),
                bigger_block_goal_positions,
                False,
                True,
                'pose',
                None,
                1 / 2
            )
        )
        self.lang_goals.append(self.lang_template)
        self.goals.append(
            (
                blocks[1::2],
                np.eye(self.n_colors),
                smaller_block_goal_positions,
                False,
                True,
                'pose',
                None,
                1 / 2
            )
        )
        self.lang_goals.append(self.lang_template)

        block_color_list = selected_color_names[:self.n_colors] + distractor_color_names[
                                                                  :n_distractors]
        np.random.shuffle(block_color_list)
        self.scene_description = (
            f"On the table, there are {2 * self.n_colors + n_distractors} blocks. "
            f"Their colors are {', '.join(block_color_list)}. "
            f"The size of the blocks are different. "
            f"Some blocks are 'bigger' and some blocks are 'smaller'."
            f"There are {self.n_colors} zones. "
            f"Their colors are {', '.join(selected_color_names)}.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class StackBlockOfSameSize(Task):
    """Stack blocks of the same size.
    Stack all bigger blocks, and stack all smaller blocks.
    There are at most 4 blocks of the same size.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_bigger_blocks, self.n_smaller_blocks = 0, 0
        self.lang_template = "stack blocks of the same size."
        self.task_completed_desc = "done stacking all the objects"

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

        # Add blocks
        bigger_blocks, smaller_blocks = [], []
        bigger_block_positions, smaller_block_positions = [], []
        smaller_block_size, bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        smaller_block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        first_bigger_pos, first_smaller_pos = None, None
        for i in range(self.n_bigger_blocks):
            block_pose = self.get_random_pose(env, bigger_block_size)  # (pos, rot)
            if i == 0:
                first_bigger_pos = block_pose
            bigger_block_positions.append(block_pose)
            block_id = env.add_object(bigger_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=bigger_colors[i] + [1])
            bigger_blocks.append((block_id, (0, None)))
        for i in range(self.n_smaller_blocks):
            block_pose = self.get_random_pose(env, smaller_block_size)  # (pos, rot)
            if i == 0:
                first_smaller_pos = block_pose
            smaller_block_positions.append(block_pose)
            block_id = env.add_object(smaller_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=smaller_colors[i] + [1])
            smaller_blocks.append((block_id, (0, None)))

        # Goal
        bigger_block_goal_poses = []
        for i in range(1, self.n_bigger_blocks):
            goal_height = bigger_block_size[2] + bigger_block_size[2] * i
            bigger_block_goal_poses.append(
                (
                    (first_bigger_pos[0][0], first_bigger_pos[0][1], goal_height),
                    first_bigger_pos[1]
                )
            )
        smaller_block_goal_poses = []
        for i in range(1, self.n_smaller_blocks):
            goal_height = smaller_block_size[2] + smaller_block_size[2] * i
            smaller_block_goal_poses.append(
                (
                    (first_smaller_pos[0][0], first_smaller_pos[0][1], goal_height),
                    first_smaller_pos[1]
                )
            )

        self.goals.append(
            (
                bigger_blocks[1:] + smaller_blocks[1:],
                np.eye(self.n_bigger_blocks + self.n_smaller_blocks - 2),
                bigger_block_goal_poses + smaller_block_goal_poses,
                False,
                True,
                'pose',
                None,
                1
            )
        )
        self.lang_goals.append(self.lang_template)

        block_color_list = bigger_color_names + smaller_color_names
        np.random.shuffle(block_color_list)
        self.scene_description = (
            f"On the table, there are {self.n_bigger_blocks + self.n_smaller_blocks} blocks. "
            f"Their colors are {', '.join(block_color_list)}. "
            f"The size of the blocks are different. "
            f"Some blocks are 'bigger' and some blocks are 'smaller'.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
