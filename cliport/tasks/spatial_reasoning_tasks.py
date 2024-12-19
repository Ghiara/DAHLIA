import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class StackBlockInAbsoluteArea(Task):
    """Stack all the blocks in an absolute position area.
    Example: ``stack all the blocks in the top right area".
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.lang_template = "stack all the blocks in the {absolute_area}."
        self.task_completed_desc = "done stacking all the blocks."
        self.additional_reset()

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

                if x_start <= block_pose[0][0] <= x_end and y_start <= block_pose[0][1] <= y_end:
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

        # block_color_list = selected_color_names[:self.n_colors] + distractor_color_names[:n_distractors]
        block_color_list = selected_color_names[:self.n_colors]
        np.random.shuffle(block_color_list)
        self.scene_description = (f"On the table, there are {self.n_colors} blocks. "
                                  f"Their colors are {', '.join(block_color_list)}. ")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class MoveBlockInXAreaToYArea(Task):
    """Move all the blocks in the [X] area to [Y] area.
    X and Y are sampled from [top left, top right, bottom left, bottom right]
    Example: ``Move all the blocks in the top left area to bottom right area".
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.n_colors = 0
        self.lang_template = "move all the blocks in the {source_area} to the {target_area}."
        self.task_completed_desc = "done moving all the blocks."

    def reset(self, env):
        super().reset(env)
        abs_area_list = ["top left area", "top right area", "bottom left area",
                         "bottom right area"]
        source_area, target_area = random.sample(abs_area_list, 2)

        self.n_colors = np.random.randint(5, 8)
        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, self.n_colors)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add blocks
        blocks = []
        block_pts = {}
        source_area_block_ids = []
        source_area_blocks = []
        block_size = (0.04, 0.04, 0.04)
        smaller_block_urdf = 'stacking/block.urdf'
        bigger_block_urdf = 'stacking/bigger_block.urdf'
        while len(source_area_blocks) < 3 or len(source_area_blocks) > 6:
            source_area_blocks = []
            for i in range(2 * self.n_colors):
                if i % 2 == 0:
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(smaller_block_urdf, block_pose)
                else:
                    block_pose = self.get_random_pose(env, block_size)
                    block_id = env.add_object(bigger_block_urdf, block_pose)

                if (
                        (
                                self.area_boundary[source_area]["x_start"]
                                <= block_pose[0][0] <=
                                self.area_boundary[source_area]["x_end"]
                        )
                        and
                        (
                                self.area_boundary[source_area]["y_start"]
                                <= block_pose[0][1] <=
                                self.area_boundary[source_area]["y_end"]
                        )
                ):
                    source_area_blocks.append((block_id, (0, None)))
                    source_area_block_ids.append(block_id)
                block_pts[block_id] = self.get_box_object_points(block_id)
                p.changeVisualShape(block_id, -1, rgbaColor=colors[i // 2] + [1])
                # blocks.append((block_id, (0, None)))

        # TODO: target area zone definition
        theta = 0
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        target_area_pose = (
            (
                self.area_boundary[target_area]["x_start"] + self.x_length / 4,
                self.area_boundary[target_area]["y_start"] + self.y_length / 4,
                0
            ), rot
        )
        target_area_size = (self.x_length / 2, self.y_length / 2, 0)

        # Goal
        pts = {k: block_pts[k] for k in source_area_block_ids}
        self.goals.append(
            (
                source_area_blocks, np.ones((len(source_area_blocks), 1)),
                [target_area_pose],
                True, False, 'zone',
                (pts, [(target_area_pose, target_area_size)]),
                1
            )
        )
        self.lang_goals.append(self.lang_template.format(source_area=source_area,
                                                         target_area=target_area))

        self.max_steps = len(source_area_blocks) + 3

        block_color_list = selected_color_names[:self.n_colors]
        np.random.shuffle(block_color_list)
        self.scene_description = (f"On the table, there are {2 * self.n_colors} blocks. "
                                  f"Their colors are {', '.join(block_color_list)}. "
                                  f"The sizes of blocks are different. "
                                  f"Some blocks are 'bigger' and some blocks are 'smaller'.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
