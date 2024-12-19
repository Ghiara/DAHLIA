import numpy as np
import os
import pybullet as p
import random
from cliport.tasks.task import Task
from cliport.utils import utils


class AssembleWord(Task):
    """Pick letters to form an English word."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.lang_template = "Pick letters to form English word {word}"
        self.task_completed_desc = "done forming word."
        self.obj = {}
        self.goal = """Choose letters to form an English word at the front of the line, the word should also be placed centered."""
        self.additional_reset()

    def reset(self, env):
        super().reset(env)

        lib = ['A', 'E', 'G', 'L', 'M', 'R', 'O', 'T', 'V']

        # Add line
        line_size = (0.01, 0.8, 0.01)
        line_template = 'line/line-template.urdf'
        line_pose = ((0.65, 0, 0.001), (0, 0, 0, 1))
        replace = {'DIM': line_size, 'COLOR': utils.get_random_color()[0]}
        line_urdf = self.fill_template(line_template, replace)
        line = env.add_object(line_urdf, line_pose, 'fixed')
        self.obj[line] = {'catagory': 'fixed', 'shape': 'line'}

        # Add the target letter
        letters = {}
        for letter in lib:
            letters[letter] = []
            for _ in range(2):
                shape = os.path.join(self.assets_root, 'kitting', f'{self.kit[letter]:02d}.obj')
                letter_pose = self.get_random_pose(env, (0.02, 0.02, 0.01),
                                                   bound=np.array(
                                                       [[0.25, 0.6], [-0.5, 0.5], [0, 0.3]]))
                scale = [0.003, 0.003, 0.0005]
                replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': utils.get_random_color()[0]}
                template = 'kitting/object-template.urdf'
                urdf = self.fill_template(template, replace)
                letter_id = env.add_object(urdf, letter_pose)
                letters[letter].append(letter_id)
                self.obj[letter_id] = {'shape': letter}

        word_list = ['EAGLE', 'MOVE', 'MOTEL', 'MEGA', 'MARMOT', 'RAGE',
                     'LARGE', 'LOVE', 'GROVE', 'ROME', 'GREAT', 'MOTOR',
                     'VOTE', 'MART', 'MORTAL', 'ROT', 'GALA', 'GAMMA',
                     'METRO', 'VOLTAGE', 'MARVEL', 'VAG', 'GLOAM']

        word = random.sample(word_list, 1)[0]
        ids = []
        for letter in word:
            ids.append(letters[letter][-1])
            letters[letter].pop()

        # Add goal
        self.add_goal(objs=ids, matches=np.eye(len(word)), targ_poses=self.targ_word(word),
                      replace=False,
                      rotations=False, metric='pose', params=None, step_max_reward=2,
                      language_goal=self.lang_template.format(word=word))

    def targ_word(self, word=''):
        l = len(word)
        if l == 0:
            return None
        elif l == 1:
            targs = [((0.7, 0, 0.01), (0, 0, 0, 1))]
            return targs
        else:
            gap = 0.08
            targs = [((0.7, -l // 2 * gap + i * gap, 0.01), (0, 0, 0, 1)) for i in range(l)]

        return targs
