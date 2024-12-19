import numpy as np
import pybullet as p
from cliport.tasks.task import Task
from cliport.utils import utils


class PrecisionBallPlacementOnStackedBlocks(Task):
    """Arrange four balls of different colors on top of four individually stacked block towers."""

    def __init__(self):
        super().__init__()
        self.max_steps = 20
        self.lang_template = "place the {color} ball on the {color} block tower"
        self.task_completed_desc = "completed precision ball placement on stacked blocks."
        self.obj = {}
        self.goal = """Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        # Define colors and assets
        block_colors = ['red', 'blue', 'green', 'yellow']
        ball_urdf = 'ball/ball-template.urdf'
        block_urdf = 'block/block.urdf'
        block_size = (0.05, 0.05, 0.05)  # x, y, z dimensions for the block
        ball_size = (0.02, 0.02, 0.02)  # x, y, z dimensions for the ball
        ball_urdf = self.fill_template(ball_urdf, {'DIM': ball_size})

        # Initialize lists for blocks and balls
        blocks = []
        balls = []
        targ_block_pose = []
        targ_ball_pose = []

        # Define the base position for the first tower and the offset for each subsequent tower
        base_position = np.array([0.3, -0.4, 0.025])
        position_offset = np.array([0.1, 0.15, 0])

        # Create four towers, each with three blocks
        for i, color in enumerate(block_colors):
            tower_position = base_position + i * position_offset
            for j in range(2):
                block_pose = self.get_random_pose(env, block_size)
                block_id = env.add_object(block_urdf, block_pose)
                p.changeVisualShape(block_id, -1, rgbaColor=utils.COLORS[color] + [1])
                blocks.append(block_id)
                targ_block_pose.append(
                    (tower_position + np.array([0, 0, j * block_size[2]]), (0, 0, 0, 1)))

        # Add balls of matching colors to be placed on top of each tower
        for i, color in enumerate(block_colors):
            ball_position = base_position + i * position_offset + np.array(
                [0, 0, 2 * block_size[2] + ball_size[2] / 2])
            ball_pose = self.get_random_pose(env, ball_size)
            ball_id = env.add_object(ball_urdf, ball_pose)
            p.changeVisualShape(ball_id, -1, rgbaColor=utils.COLORS[color] + [1])
            balls.append(ball_id)
            targ_ball_pose.append((ball_position, (0, 0, 0, 1)))

        # Define target poses for each ball to be placed on top of the corresponding color tower
        for i, block_id in enumerate(blocks):
            self.add_goal(objs=[block_id], matches=np.ones((1, 1)),
                          targ_poses=[targ_block_pose[i]], replace=True,
                          rotations=True, metric='pose', params=None,
                          step_max_reward=1 / len(blocks) / 2,
                          language_goal=f'build the {block_colors[i // 2]} block tower')

        for i, ball_id in enumerate(balls):
            self.add_goal(objs=[ball_id], matches=np.ones((1, 1)), targ_poses=[targ_ball_pose[i]],
                          replace=True,
                          rotations=True, metric='pose', params=None, step_max_reward=1 / 4 / 2,
                          language_goal=f'place the {block_colors[i]} ball on {block_colors[i]} block tower')
