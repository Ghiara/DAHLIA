import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class PutEvenBlockInSameColorZone(Task):
    """Put the blocks of an even number in the zone with the same color as the block. """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the blocks of an even number in the zone " \
                             "with the matching color."
        self.task_completed_desc = "done placing blocks in the zone."
        self.additional_reset()

    def reset(self, env):
        super().reset(env)
        n_colors = np.random.randint(2, 4)
        n_blocks = np.random.randint(n_colors + 1, min(4 * n_colors, 10))
        while (n_blocks % n_colors == 0) and (n_blocks // n_colors % 2 == 1):
            n_blocks = np.random.randint(n_colors + 1, min(4 * n_colors, 10))
        # n_colors = 3
        # n_blocks = 12

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_colors)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # all_corner_names = ['bottom right corner', 'bottom side', 'bottom left corner']
        # all_corner_target_pos = [(0.65, 0.35, 0), (0.5, 0.25, 0), (0.35, 0.35, 0)]
        # all_corner_size = [(0.2, 0.3, 0), (0.5, 0.3, 0), (0.2, 0.3, 0)]
        # corner_idx = random.sample(range(len(all_corner_names)), 1)[0]

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        block_id_list = []
        block_color_names = []
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            block_id_list.append(block_id)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i % n_colors] + [1])
            block_color_names.append(selected_color_names[i % n_colors])
            block_pts[block_id] = self.get_box_object_points(block_id)
            blocks.append((block_id, (0, None)))

        # Add zone
        zone_size = (0.15, 0.15, 0)
        zone_poses = []
        for i in range(n_colors):
            zone_pose = self.get_random_pose(env, zone_size)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            while not zone_obj_id:
                zone_size = (zone_size[0] - 0.01, zone_size[1] - 0.01, 0)
                zone_pose = self.get_random_pose(env, zone_size)
                zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')

            p.changeVisualShape(zone_obj_id, -1, rgbaColor=colors[i] + [1])
            zone_poses.append(zone_pose)

        self.scene_description = f"On the table, there are {n_blocks} blocks. Their colors are {block_color_names}. " \
                                 f"There are {n_colors} zones. Their colors are {selected_color_names}. "

        # Goal: put each block in the corner.
        if (n_blocks % n_colors == 0) and (n_blocks // n_colors % 2 == 0):
            for i in range(n_colors):
                pts = {k: block_pts[k] for k in block_id_list[i::n_colors]}
                self.goals.append(
                    (
                        blocks[i::n_colors], np.ones((n_blocks // n_colors, 1)),
                        [zone_poses[i]],
                        True, False, 'zone',
                        (pts, [(zone_poses[i], zone_size)]),
                        1 / n_colors
                    )
                )
                self.lang_goals.append(self.lang_template)

        elif n_blocks % n_colors == 1:
            if (n_blocks // n_colors) % 2 == 1:
                pts = {k: block_pts[k] for k in block_id_list[::n_colors]}
                self.goals.append(
                    (
                        blocks[::n_colors], np.ones((n_blocks // n_colors + 1, 1)),
                        [zone_poses[0]],
                        True, False, 'zone',
                        (pts, [(zone_poses[0], zone_size)]),
                        1
                    )
                )
                self.lang_goals.append(self.lang_template)
            else:
                for i in range(1, n_colors):
                    pts = {k: block_pts[k] for k in block_id_list[i::n_colors]}
                    self.goals.append(
                        (
                            blocks[i::n_colors], np.ones((n_blocks // n_colors, 1)),
                            [zone_poses[i]],
                            True, False, 'zone',
                            (pts, [(zone_poses[i], zone_size)]),
                            1 / (n_colors - 1)
                        )
                    )
                    self.lang_goals.append(self.lang_template)
        elif n_blocks % n_colors == 2:
            if n_blocks // n_colors % 2 == 1:
                for i in range(n_colors - 1):
                    pts = {k: block_pts[k] for k in block_id_list[i::n_colors]}
                    self.goals.append(
                        (
                            blocks[i::n_colors], np.ones((n_blocks // n_colors + 1, 1)),
                            [zone_poses[i]],
                            True, False, 'zone',
                            (pts, [(zone_poses[i], zone_size)]),
                            1 / (n_colors - 1)
                        )
                    )
                    self.lang_goals.append(self.lang_template)
            else:
                pts = {k: block_pts[k] for k in block_id_list[n_colors - 1::n_colors]}
                self.goals.append(
                    (
                        blocks[n_colors - 1::n_colors], np.ones((n_blocks // n_colors, 1)),
                        [zone_poses[n_colors - 1]],
                        True, False, 'zone',
                        (pts, [(zone_poses[0], zone_size)]),
                        1
                    )
                )
                self.lang_goals.append(self.lang_template)

        # if n_blocks % 2 == 0:
        #     selected_block_pts_0 = {k: block_pts[k] for k in block_id_list[::2]}
        #     selected_block_pts_1 = {k: block_pts[k] for k in block_id_list[1::2]}
        #
        #     self.goals.append((blocks[::2], np.ones((n_blocks // 2, 1)), [zone_poses[0]],
        #                        True, False, 'zone',
        #                        (selected_block_pts_0, [(zone_poses[0], zone_size)]), 1))
        #     self.goals.append((blocks[1::2], np.ones((n_blocks // 2, 1)), [zone_poses[1]],
        #                        True, False, 'zone',
        #                        (selected_block_pts_1, [(zone_poses[1], zone_size)]), 1))
        # else:
        #     if n_blocks == 3:
        #         selected_blocks = [blocks[0], blocks[2]]
        #         selected_block_pts = {k: block_pts[k] for k in [block_id_list[0], block_id_list[2]]}
        #         selected_zone = zone_poses[0]
        #         match_matrix = np.ones((2, 1))
        #     elif n_blocks == 5:
        #         selected_blocks = [blocks[1], blocks[3]]
        #         selected_block_pts = {k: block_pts[k] for k in [block_id_list[1], block_id_list[3]]}
        #         selected_zone = zone_poses[1]
        #         match_matrix = np.ones((2, 1))
        #     else:
        #         raise ValueError("block number is wrong")
        #     self.goals.append((selected_blocks, match_matrix, [selected_zone],
        #                        True, False, 'zone',
        #                        (selected_block_pts, [(selected_zone, zone_size)]), 1))
        #
        # self.lang_goals.append(self.lang_template)

        # Only two mistake allowed.
        self.max_steps = n_blocks + 2

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
