"""Put Blocks in Bowl Task."""

import random

import numpy as np
import pybullet as p

from cliport.tasks.task import Task
from cliport.utils import utils


class PutBlockInBowlUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} blocks in a {place} bowl"
        self.task_completed_desc = "done placing blocks in bowls."

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_blocks = np.random.randint(1, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for _ in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[1] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for _ in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.ones((len(blocks), len(bowl_poses))),
                           bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0],
                                                         place=selected_color_names[1]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if
                                  c not in selected_color_names]
        distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if
                                   c not in selected_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_block_colors if is_block else distractor_bowl_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockInBowlSeenColors(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutBlockInBowlFull(PutBlockInBowlUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors


class PutBlockInMismatchingBowl(Task):
    """Put the blocks in the bowls with mismatched colors base class and task"""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the blocks in the bowls with mismatched colors."
        self.task_completed_desc = "done placing blocks in bowls."
        self.obj = {}
        self.goal = 'Put the blocks in the bowls with mismatched colors.'

    def reset(self, env):
        super().reset(env)
        self.consider_z_in_match = False

        n_bowls = np.random.randint(2, 4)
        n_blocks = np.random.randint(2, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_bowls)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-mismatching-bowl"

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i, _ in enumerate(range(n_bowls)):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            self.obj[bowl_id] = {'catagory': 'fixed', 'color': selected_color_names[i],
                                 'shape': 'bowl'}
            bowl_poses.append(bowl_pose)
            place_obj_names.append(f"{selected_color_names[i]} bowl")
        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i, _ in enumerate(range(n_blocks)):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            self.obj[block_id] = {'color': selected_color_names[i], 'shape': 'block'}
            blocks.append((block_id, (0, None)))
            pick_obj_names.append(f"{selected_color_names[i]} block")
        self.pick_obj_names.append(pick_obj_names)

        # Goal: put each block in a bowl with mismatching color.
        self.goals.append(
            (blocks, 1 - np.eye(len(blocks)), bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 2

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_color_name = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_name]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        while n_distractors < max_distractors and distractor_colors:
            is_block = False
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            color_name = distractor_color_name[n_distractors % len(colors)]
            distractor_colors.remove(color)
            distractor_color_name.remove(color_name)
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            distractor_block.append(color_name)
            n_distractors += 1

        self.scene_description = f"On the table, there are {n_blocks + len(distractor_block)} blocks. " \
                                 f"Their colors are {selected_color_names[:n_blocks] + distractor_block}. " \
                                 f"There are {n_bowls} bowls. " \
                                 f"Their colors are {selected_color_names[:n_bowls]}."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockInMatchingBowl(Task):
    """Put Block in Matching Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the blocks in the bowls with matching colors."
        self.task_completed_desc = "done placing blocks in bowls."
        self.seed = 0
        self.goal = "put the blocks in the bowls with matching colors."
        self.obj = {}

    def reset(self, env):
        super().reset(env)
        self.consider_z_in_match = False

        n_bowls = np.random.randint(2, 4)
        n_blocks = np.random.randint(2, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_bowls)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            self.obj[bowl_id] = {'catagory': 'fixed', 'color': selected_color_names[i],
                                 'shape': 'bowl'}
            bowl_poses.append(bowl_pose)
            place_obj_names.append(f"{selected_color_names[i]} bowl")
        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            self.obj[block_id] = {'color': selected_color_names[i], 'shape': 'block'}
            blocks.append((block_id, (0, None)))
            pick_obj_names.append(f"{selected_color_names[i]} block")
        self.pick_obj_names.append(pick_obj_names)

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.eye(len(blocks)), bowl_poses, False, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 2

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]
        print(f"selected_color_names: {selected_color_names}, \n"
              f"distractor_color_names: {distractor_color_names}.")

        # Add distractors.
        n_distractors = 0
        max_distractors = 1
        distractor_block = []
        distractor_bowl = []
        while n_distractors < max_distractors and distractor_colors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_colors
            pose = self.get_random_pose(env, size)
            obj_id = env.add_object(urdf, pose)
            if not obj_id:
                break
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            if is_block:
                distractor_block.append(color_name)
            else:
                distractor_bowl.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            print(f"add distractor of the color {color_name}.")
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls] + distractor_bowl
        np.random.shuffle(bowl_list)
        self.scene_description = f"On the table, there are {n_blocks + len(distractor_block)} blocks. " \
                                 f"Their colors are {', '.join(block_list)}. " \
                                 f"There are {n_bowls + len(distractor_bowl)} bowls. " \
                                 f"Their colors are {', '.join(bowl_list)}."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockIntoMatchingBowlWithDetails(Task):
    """Put Block in Matching Bowl with step-by-step instructions."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = ("The goal is that Put the blocks in the bowls with matching colors. "
                              "The step-by-step instructions are: {step_instructions}")
        self.task_completed_desc = "done placing blocks in bowls."
        self.seed = 0

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = False

        n_bowls = np.random.randint(5, 8)
        n_blocks = np.random.randint(5, n_bowls + 1)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_bowls)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            bowl_poses.append(bowl_pose)
            place_obj_names.append(f"{selected_color_names[i]} bowl")
        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            blocks.append((block_id, (0, None)))
            pick_obj_names.append(f"{selected_color_names[i]} block")
        self.pick_obj_names.append(pick_obj_names)

        # Goal: put each block in a different bowl.
        self.goals.append((blocks, np.eye(len(blocks)), bowl_poses, False, True, 'pose', None, 1))

        # Step instructions
        step_instruction_template = ("Pick up the {block_color} block "
                                     "and place it on the {bowl_color} bowl.")
        step_instructions = []
        for c in selected_color_names:
            step_instructions.append(
                step_instruction_template.format(block_color=c, bowl_color=c)
            )
        step_instructions = " ".join(step_instructions)
        self.lang_goals.append(self.lang_template.format(step_instructions=step_instructions))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 2

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_color_names = [c for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        distractor_bowl = []
        while n_distractors < max_distractors and distractor_colors:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_colors
            pose = self.get_random_pose(env, size)
            obj_id = env.add_object(urdf, pose)
            if not obj_id:
                break
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            if is_block:
                distractor_block.append(color_name)
            else:
                distractor_bowl.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls] + distractor_bowl
        np.random.shuffle(bowl_list)
        self.scene_description = f"On the table, there are {n_blocks + len(distractor_block)} blocks. " \
                                 f"Their colors are {', '.join(block_list)}. " \
                                 f"There are {n_bowls + len(distractor_bowl)} bowls. " \
                                 f"Their colors are {', '.join(bowl_list)}."

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutAllBlockInABowl(Task):
    """Put all the blocks in a bowl base class and task"""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put all the blocks in the {color} bowl"
        self.task_completed_desc = "done placing blocks in bowls."

    def reset(self, env):
        super().reset(env)
        # n_bowls = np.random.randint(1, 4)
        # n_blocks = np.random.randint(1, 2)
        n_bowls = 5
        n_blocks = 2

        all_color_names = self.get_colors()
        bowl_color = random.sample(all_color_names, n_bowls)
        block_color = random.sample(all_color_names, n_blocks)
        bowl_color_ = [utils.COLORS[cn] for cn in bowl_color]
        block_color_ = [utils.COLORS[cn] for cn in block_color]

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = 'bowl/bowl.urdf'
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_color_[i] + [1])
            bowl_poses.append(bowl_pose)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_color_[i] + [1])
            blocks.append((block_id, (0, None)))

        self.scene_description = f"On the table, there are {n_blocks} blocks. " \
                                 f"Their colors are {block_color}. " \
                                 f"There are {n_bowls} bowls. " \
                                 f"The colors of bowls are {bowl_color}."

        # Goal: put all the blocks in the bowl of the first color.
        matches = np.zeros((len(blocks), len(bowl_poses)))
        matches[:, 0] = 1
        self.goals.append((blocks, matches, bowl_poses, True, True, 'pose', None, 1))
        self.lang_goals.append(self.lang_template.format(color=bowl_color[0]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Colors of distractor objects.
        # distractor_bowl_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        # distractor_block_colors = [utils.COLORS[c] for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in utils.COLORS if
                             c not in (bowl_color + block_color)]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        while n_distractors < max_distractors and distractor_colors:
            is_block = False
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            colors = distractor_colors
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = colors[n_distractors % len(colors)]
            distractor_colors.remove(color)
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutAllBlockOnCorner(Task):
    """Put all the blocks on the [x corner/side] base class and task.
    corner/side: bottom right corner, bottom side, bottom left corner"""

    # TODO: how to define corner?
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put all the blocks on the {corner}"
        self.task_completed_desc = "done placing blocks on the corner."

    def reset(self, env):
        super().reset(env)
        n_blocks = np.random.randint(1, 5)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_blocks)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        corner_selected_colors = [c for c in all_color_names if c not in selected_color_names]
        corner_colors = [utils.COLORS[cn] for cn in corner_selected_colors]
        all_corner_names = ['bottom right corner', 'bottom side', 'bottom left corner']
        all_corner_target_pos = [(0.65, 0.35, 0), (0.5, 0.25, 0), (0.35, 0.35, 0)]
        all_corner_size = [(0.2, 0.3, 0), (0.5, 0.3, 0), (0.2, 0.3, 0)]
        corner_idx = random.sample(range(len(all_corner_names)), 1)[0]

        # Add blocks.
        blocks = []
        block_pts = {}
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            block_pts[block_id] = self.get_box_object_points(block_id)
            blocks.append((block_id, (0, None)))

        def get_certain_pose(pos):
            theta = np.random.rand() * 2 * np.pi
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            return pos, rot

        zone_size = all_corner_size[corner_idx]
        # zone_pose = get_certain_pose(all_corner_target_pos[corner_idx])
        zone_pose = self.get_random_pose(env, zone_size)
        zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
        zone_color = random.sample(corner_colors, 1)
        zone_target = zone_pose
        p.changeVisualShape(zone_obj_id, -1, rgbaColor=zone_color + [1])

        # Goal: put each block in the corner.
        self.goals.append((blocks, np.ones((n_blocks, 1)), [zone_target],
                           True, False, 'zone',
                           (block_pts, [(zone_target, zone_size)]), 1))
        self.lang_goals.append(self.lang_template.format(corner=all_corner_names[corner_idx]))

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PickAndPlace(Task):
    """Pick up the [block1] and place it on the [block2/bowl/zone]."""

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick_color} block and place it on the {place_color} {place}"
        self.task_completed_desc = "done placing blocks."

    def reset(self, env):
        super().reset(env)
        target_idx = random.randint(0, 2)
        # target_idx = 1
        target_objs = ["block", "bowl", "zone"]
        all_color_names = self.get_colors()

        n_blocks = random.randint(2, 4)
        n_zones = random.randint(1, 3)
        n_bowls = random.randint(2, 4)
        # n_blocks = 4
        # n_zones = 3
        # n_bowls = 4

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
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_util_colors[i] + [1])
            if i == 0:
                block_pts[block_id] = self.get_box_object_points(block_id)
                base_pose = block_pose
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
            place_pos = [(0, 0, 0.03), (0, 0, 0.08)]
            targets = [(utils.apply(base_pose, i), base_pose[1]) for i in place_pos]
            self.goals.append((blocks[:2], np.eye(2),
                               targets, False, True, 'pose', None, 1))
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

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutEvenBlockInCorrespondingZone(Task):
    """Put the blocks of an even number in the zone
     with the corresponding color base class and task. """

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the blocks of an even number in the zone " \
                             "with the corresponding color"
        self.task_completed_desc = "done placing blocks in the zone."

    def reset(self, env):
        super().reset(env)
        n_blocks = random.randint(2, 5)
        n_blocks = 4
        n_colors = 2

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_blocks)
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
            # print("haha zone_pose", zone_pose)
            zone_obj_id = env.add_object('zone/zone.urdf', zone_pose, 'fixed')
            p.changeVisualShape(zone_obj_id, -1, rgbaColor=colors[i] + [1])
            zone_poses.append(zone_pose)

        self.scene_description = f"On the table, there are {n_blocks} blocks. Their colors are {block_color_names}. " \
                                 f"There are two zones. Their colors are {selected_color_names[:2]}. "

        # Goal: put each block in the corner.
        if n_blocks % 2 == 0:
            selected_block_pts_0 = {k: block_pts[k] for k in block_id_list[::2]}
            selected_block_pts_1 = {k: block_pts[k] for k in block_id_list[1::2]}

            self.goals.append((blocks[::2], np.ones((n_blocks // 2, 1)), [zone_poses[0]],
                               True, False, 'zone',
                               (selected_block_pts_0, [(zone_poses[0], zone_size)]), 1))
            self.goals.append((blocks[1::2], np.ones((n_blocks // 2, 1)), [zone_poses[1]],
                               True, False, 'zone',
                               (selected_block_pts_1, [(zone_poses[1], zone_size)]), 1))
        else:
            if n_blocks == 3:
                selected_blocks = [blocks[0], blocks[2]]
                selected_block_pts = {k: block_pts[k] for k in
                                      [block_id_list[0], block_id_list[2]]}
                selected_zone = zone_poses[0]
                match_matrix = np.ones((2, 1))
            elif n_blocks == 5:
                selected_blocks = [blocks[1], blocks[3]]
                selected_block_pts = {k: block_pts[k] for k in
                                      [block_id_list[1], block_id_list[3]]}
                selected_zone = zone_poses[1]
                match_matrix = np.ones((2, 1))
            else:
                raise ValueError("block number is wrong")
            self.goals.append((selected_blocks, match_matrix, [selected_zone],
                               True, False, 'zone',
                               (selected_block_pts, [(selected_zone, zone_size)]), 1))

        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutBlockIntoBowl(Task):
    def __init__(self):
        super().__init__()

        self.pos_eps = 0.05
        self.seed = 0
        self.smaller_block_size, self.bigger_block_size = (0.04, 0.04, 0.04), (0.06, 0.06, 0.06)
        self.smaller_block_urdf, self.bigger_block_urdf = ("stacking/block.urdf",
                                                           "stacking/bigger_block.urdf")
        self.bowl_size = (0.12, 0.12, 0)
        self.bowl_urdf = 'bowl/bowl.urdf'

    def reset(self, env):
        self.input_manipulate_order = True
        self.consider_z_in_match = False
        super().reset(env)


class PutHiddenBlockIntoMatchingBowl(PutBlockIntoBowl):
    """
    Task instruction example:

    Put the {hidden_block_color} block under the {top_block_color} block
    into the {bowl_color} bowl.
    """

    def __init__(self):
        super().__init__()
        self.lang_template = ("put the {hidden_block_color} block under the "
                              "{top_block_color} block into the bowl with matching color.")
        self.task_completed_desc = "done placing the block into the bowl."

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = False

        # n_bowls = np.random.randint(3, 5)
        n_blocks = np.random.randint(4, 8)
        n_bowls = n_blocks
        self.max_steps = n_blocks + 2

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_blocks)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        tgt_bowl_pose = None
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, self.bowl_size)
            bowl_id = env.add_object(self.bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            place_obj_names.append(f"{selected_color_names[i]} bowl")
            if i == 0:
                tgt_bowl_pose = bowl_pose

        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        selected_blocks = []
        bottom_block_pose = None
        for i in range(n_blocks):
            if i == 1:
                pos, rot = bottom_block_pose
                block_pose = (
                    (pos[0], pos[1], self.smaller_block_size[2] + self.bigger_block_size[2] / 2),
                    rot
                )
                block_id = env.add_object(self.bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, self.smaller_block_size)
                block_id = env.add_object(self.smaller_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            pick_obj_names.append(f"{selected_color_names[i]} block")
            if i == 0:
                bottom_block_pose = block_pose
            if i in [0, 1]:
                selected_blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names)

        tgt_pose = self.get_random_pose(env, self.bigger_block_size)
        self.goals.append(
            (
                [selected_blocks[1]], np.eye(1), [tgt_pose],
                False, True, 'pose', None, 0.001
            )
        )
        self.lang_goals.append(
            self.lang_template.format(hidden_block_color=selected_color_names[0],
                                      top_block_color=selected_color_names[1], )
        )
        self.goals.append(
            (
                [selected_blocks[0]], np.eye(1), [tgt_bowl_pose],
                False, True, 'pose', None, 0.999
            )
        )
        self.lang_goals.append(
            self.lang_template.format(hidden_block_color=selected_color_names[0],
                                      top_block_color=selected_color_names[1], )
        )

        # Only two mistakes allowed.
        self.max_steps = len(selected_blocks) + 2

        # Colors of distractor objects.
        distractor_color_names = [selected_color_names[0]]
        distractor_color_names += [c for c in utils.COLORS if c not in selected_color_names]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        while n_distractors < max_distractors and distractor_colors:
            colors = distractor_colors
            pose = self.get_random_pose(env, self.smaller_block_size)
            obj_id = env.add_object(self.smaller_block_urdf, pose)
            if not obj_id:
                break
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            distractor_block.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls]
        np.random.shuffle(bowl_list)
        self.scene_description = (f"On the table, "
                                  f"there are {n_blocks + len(distractor_block)} blocks. "
                                  f"Their colors are {', '.join(block_list)}. "
                                  f"There are {n_bowls} bowls. "
                                  f"Their colors are {', '.join(bowl_list)}.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutHiddenBlocksInTwoLayerTowersIntoMatchingBowls(PutBlockIntoBowl):
    """
    Task instruction example:
    Put all the hidden objects in the two-layer stacked towers into the bowls with matching colors.
    """

    def __init__(self):
        super().__init__()
        self.pos_eps = 0.05
        self.lang_template = ("put all the hidden blocks in the "
                              "two-layer stacked blocks into the bowls with matching colors.")
        self.task_completed_desc = "done placing the block in bowls."
        self.seed = 0

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = False

        # n_bowls = np.random.randint(3, 5)
        n_blocks = np.random.randint(4, 8)
        n_bowls = n_blocks
        self.max_steps = n_blocks + 2

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, n_blocks)

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        bowl_poses = []
        tgt_bowl_color = selected_color_names[0]
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, self.bowl_size)
            bowl_id = env.add_object(self.bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            bowl_poses.append(bowl_pose)
            place_obj_names.append(f"{selected_color_names[i]} bowl")

        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        selected_blocks = []
        bottom_block_pose = None
        block_pts = {}
        for i in range(n_blocks):
            if i % 2 == 1:
                pos, rot = bottom_block_pose
                block_pose = (
                    (pos[0], pos[1], self.smaller_block_size[2] + self.bigger_block_size[2] / 2),
                    rot
                )
                block_id = env.add_object(self.bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, self.smaller_block_size)
                block_id = env.add_object(self.smaller_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            block_pts[block_id] = self.get_box_object_points(block_id)
            pick_obj_names.append(f"{selected_color_names[i]} block")

            if i % 2 == 0:
                bottom_block_pose = block_pose
            selected_blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names)

        tgt_poses_for_top_blocks = []
        for i in range(n_blocks // 2):
            tgt_poses_for_top_blocks.append(self.get_random_pose(env, self.bigger_block_size))
        top_blocks = selected_blocks[1::2]

        # The zone-based goal cannot work
        # theta = 0
        # rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        # center_pose = (
        #     (
        #         self.bounds[0][0] + self.x_length / 2,
        #         self.bounds[1][0] + self.y_length / 2,
        #         self.bigger_block_size[2] / 2
        #     ), rot
        # )
        # self.goals.append(
        #     (
        #         top_blocks,
        #         np.eye(len(top_blocks)), tgt_poses_for_top_blocks,
        #         False, True, 'zone',
        #         (top_block_pts, [
        #             (center_pose, (self.x_length, self.y_length, 0))
        #         ]),
        #         1 / 2
        #     )
        # )
        self.goals.append(
            (
                selected_blocks[1::2], np.eye(n_blocks // 2), tgt_poses_for_top_blocks,
                False, True, 'pose', None, 0.02
            )
        )
        self.lang_goals.append(
            self.lang_template.format(hidden_block_color=selected_color_names[0],
                                      top_block_color=selected_color_names[1],
                                      bowl_color=selected_color_names[0], )
        )
        hidden_blocks = selected_blocks[::2] if n_blocks % 2 == 0 else selected_blocks[:-1:2]
        tgt_bowl_poses = bowl_poses[::2] if n_blocks % 2 == 0 else bowl_poses[:-1:2]
        assert len(hidden_blocks) == len(tgt_bowl_poses) == n_blocks // 2
        self.goals.append(
            (
                hidden_blocks, np.eye(n_blocks // 2), tgt_bowl_poses,
                False, True, 'pose', None, 0.98
            )
        )
        self.lang_goals.append(
            self.lang_template.format(hidden_block_color=selected_color_names[0],
                                      top_block_color=selected_color_names[1],
                                      bowl_color=selected_color_names[0], )
        )

        # Only one mistake allowed.
        self.max_steps = len(selected_blocks) + 2

        # Colors of distractor objects.
        distractor_color_names = selected_color_names[::2]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        while n_distractors < max_distractors and distractor_colors:
            colors = distractor_colors
            pose = self.get_random_pose(env, self.smaller_block_size)
            obj_id = env.add_object(self.smaller_block_urdf, pose)
            if not obj_id:
                break
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            distractor_block.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls]
        np.random.shuffle(bowl_list)
        self.scene_description = (f"On the table, "
                                  f"there are {n_blocks + len(distractor_block)} blocks. "
                                  f"Their colors are {', '.join(block_list)}. "
                                  f"There are {n_bowls} bowls. "
                                  f"Their colors are {', '.join(bowl_list)}.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutHiddenBlocksInThreeLayerTowersIntoMatchingBowls(PutBlockIntoBowl):
    """
    Task instruction example:
    Put all the hidden objects in the two-layer stacked towers into the bowls with matching colors.
    """

    def __init__(self):
        super().__init__()
        self.pos_eps = 0.05
        self.lang_template = ("put all the hidden blocks in the "
                              "three-layer stacked blocks into the bowls with matching colors.")
        self.task_completed_desc = "done placing the block in bowls."
        self.seed = 0
        self.n_tower_layers = None

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = False

        self.n_tower_layers = 3

        # n_bowls = np.random.randint(3, 5)
        n_blocks = np.random.randint(6, 10)
        n_bowls = np.random.randint(n_blocks // self.n_tower_layers,
                                    n_blocks // (self.n_tower_layers - 1) + 1)
        self.max_steps = n_blocks + 2

        all_color_names = self.get_colors()
        selected_color_names = list(np.random.choice(all_color_names, n_blocks, replace=False))

        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        bowl_poses = []
        bowl_color_names = selected_color_names[::self.n_tower_layers]
        n_diff = max(0, n_bowls - len(bowl_color_names))
        diff_colors = [c for c in selected_color_names if c not in bowl_color_names]
        sample_remain = list(np.random.choice(diff_colors, n_diff, replace=False))
        bowl_color_names += sample_remain
        bowl_colors = [utils.COLORS[cn] for cn in bowl_color_names]
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, self.bowl_size)
            bowl_id = env.add_object(self.bowl_urdf, bowl_pose, 'fixed')
            if not bowl_id and i >= n_blocks // self.n_tower_layers:
                break
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)
            place_obj_names.append(f"{selected_color_names[i]} bowl")

        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        selected_blocks = []
        bottom_block_pose = None
        for i in range(n_blocks):
            if i % self.n_tower_layers != 0:
                pos, rot = bottom_block_pose
                block_pose = (
                    (pos[0], pos[1],
                     self.smaller_block_size[2] +
                     self.bigger_block_size[2] * (0.5 + (i % self.n_tower_layers) - 1)),
                    rot
                )
                block_id = env.add_object(self.bigger_block_urdf, block_pose)
            else:
                block_pose = self.get_random_pose(env, self.smaller_block_size)
                block_id = env.add_object(self.smaller_block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            pick_obj_names.append(f"{selected_color_names[i]} block")
            if i % self.n_tower_layers == 0:
                bottom_block_pose = block_pose
            selected_blocks.append((block_id, (0, None)))
        self.pick_obj_names.append(pick_obj_names)

        # top_blocks = [b for i, b in enumerate(selected_blocks) if i % self.n_tower_layers != 0]
        top_blocks = [
            selected_blocks[i * self.n_tower_layers + j]
            for i in range(n_blocks // self.n_tower_layers)
            for j in range(1, self.n_tower_layers)
        ]
        # for i in range(n_blocks // self.n_tower_layers):
        #     top_blocks.append(selected_blocks[(i + 1) * self.n_tower_layers])
        #     top_blocks.append(selected_blocks[(i + 2) * self.n_tower_layers])
        tgt_poses_for_top_blocks = []
        for i in range(len(top_blocks)):
            tgt_poses_for_top_blocks.append(self.get_random_pose(env, self.bigger_block_size))

        self.goals.append(
            (
                top_blocks[::-1],
                np.eye(len(top_blocks)),
                tgt_poses_for_top_blocks,
                False, True, 'pose', None, 0.02
            )
        )
        self.lang_goals.append(self.lang_template)
        hidden_blocks = [selected_blocks[i * self.n_tower_layers]
                         for i in range(n_blocks // self.n_tower_layers)]
        tgt_bowl_poses = [bowl_poses[i]
                          for i in range(n_blocks // self.n_tower_layers)]
        self.goals.append(
            (
                hidden_blocks, np.eye(len(hidden_blocks)), tgt_bowl_poses,
                False, True, 'pose', None, 0.98
            )
        )
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = len(selected_blocks) + 2

        # Colors of distractor objects.
        distractor_color_names = selected_color_names[::3]
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        # Add distractors.
        n_distractors = 0
        max_distractors = 6
        distractor_block = []
        while n_distractors < max_distractors and distractor_colors:
            colors = distractor_colors
            pose = self.get_random_pose(env, self.smaller_block_size)
            obj_id = env.add_object(self.smaller_block_urdf, pose)
            if not obj_id:
                break
            color_name = distractor_color_names[n_distractors % len(colors)]
            color = colors[n_distractors % len(colors)]
            distractor_block.append(color_name)
            distractor_colors.remove(color)
            distractor_color_names.remove(color_name)
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            n_distractors += 1

        block_list = selected_color_names[:n_blocks] + distractor_block
        np.random.shuffle(block_list)
        bowl_list = selected_color_names[:n_bowls]
        np.random.shuffle(bowl_list)
        self.scene_description = (f"On the table, "
                                  f"there are {n_blocks + len(distractor_block)} blocks. "
                                  f"Their colors are {', '.join(block_list)}. "
                                  f"There are {n_bowls} bowls. "
                                  f"Their colors are {', '.join(bowl_list)}.")

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS


class PutHiddenBlocksInPyramidIntoMatchingBowls(PutBlockIntoBowl):
    """
    Task instruction example:
    Put the {hidden_block_color} block on the first layer of the pyramid
    into the {bowl_color} bowl.
    """

    def __init__(self):
        super().__init__()
        self.max_steps = 3
        self.pos_eps = 0.05
        self.n_bottom_blocks, self.n_top_blocks = 3, 3
        self.lang_template = ("put all the hidden blocks on the first layer "
                              "of the pyramid into the bowls with matching colors.")
        self.task_completed_desc = "done placing the block in bowls."
        self.seed = 0

    def reset(self, env):
        super().reset(env)
        self.input_manipulate_order = True
        self.consider_z_in_match = False

        n_bowls = np.random.randint(3, 6)
        n_blocks = np.random.randint(6, 9)

        all_color_names = self.get_colors()
        block_color_names = random.sample(all_color_names, n_blocks)

        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        self.place_obj_names, self.pick_obj_names = [], []
        pick_obj_names, place_obj_names = [], []
        self.task_name = "put-block-in-matching-bowl"

        # Add bowls.
        bowl_poses = []
        bowl_colors = block_colors[:n_bowls]
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, self.bowl_size)
            bowl_id = env.add_object(self.bowl_urdf, bowl_pose, 'fixed')
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)

        self.place_obj_names.append(place_obj_names)

        # Add blocks.
        use_bigger_blocks = np.random.choice([True, False], 1)[0]
        # base_size = (0.05, 0.15, 0.005)
        # base_pose = self.get_random_pose(env, base_size)
        # pyramid_pos = [(0, -0.05, 0.03), (0, 0, 0.03),
        #                (0, 0.05, 0.03), (0, -0.025, 0.08),
        #                (0, 0.025, 0.08), (0, 0, 0.13)]
        # block_initial_poses = [(utils.apply(base_pose, i), base_pose[1]) for i in pyramid_pos]

        block_size = self.bigger_block_size if use_bigger_blocks else self.smaller_block_size
        block_urdf = self.bigger_block_urdf if use_bigger_blocks else self.smaller_block_urdf

        # Add blocks to make them form a pyramid.
        blocks = []
        first_block_pose = self.get_random_pose(env, block_size)
        first_pos, first_rot = first_block_pose
        pyramid_pos = [
            first_pos,
            (first_pos[0], first_pos[1] + 0.05, first_pos[2]),
            (first_pos[0], first_pos[1] + 0.1, first_pos[2]),
            (first_pos[0], first_pos[1] + 0.025, first_pos[2] + block_size[2]),
            (first_pos[0], first_pos[1] + 0.075, first_pos[2] + block_size[2]),
            (first_pos[0], first_pos[1] + 0.05, first_pos[2] + block_size[2] * 2),
        ]
        for i in range(n_blocks):
            if i < self.n_bottom_blocks + self.n_top_blocks:
                block_id = env.add_object(block_urdf, (pyramid_pos[i], first_rot))
            else:
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i] + [1])
            blocks.append((block_id, (np.pi / 2, None)))

        # Set target poses
        tgt_poses_for_top_blocks = []
        for i in range(self.n_top_blocks):
            tgt_poses_for_top_blocks.append(self.get_random_pose(env, block_size))
        self.goals.append(
            (
                blocks[self.n_bottom_blocks: self.n_bottom_blocks + self.n_top_blocks][::-1],
                np.eye(3),
                tgt_poses_for_top_blocks,
                False, True, 'pose', None, 0.02
            )
        )
        self.lang_goals.append(self.lang_template)
        print(
            f"bowl poses: {[[round(x, 2) for x in i[0]] for i in bowl_poses[:self.n_bottom_blocks]]}"
        )
        self.goals.append(
            (
                blocks[:self.n_bottom_blocks], np.eye(3), bowl_poses[:self.n_bottom_blocks],
                False, True, 'pose', None, 0.98
            )
        )
        self.lang_goals.append(self.lang_template)

        # Only one mistake allowed.
        self.max_steps = self.n_bottom_blocks + self.n_top_blocks + 2

        block_list = block_color_names
        np.random.shuffle(block_list)
        bowl_list = block_color_names[:n_bowls]
        np.random.shuffle(bowl_list)
        self.scene_description = (
            f"On the table, there are {n_blocks} blocks. "
            f"Their colors are {', '.join(block_list)}. "
            f"There are {n_bowls} bowls. "
            f"Their colors are {', '.join(bowl_list)}."
        )

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS
