import ast
import copy
import itertools
import os
import random
import re
import traceback
from time import sleep

import astunparse
import google.generativeai as genai
import hydra
import numpy as np
import openai
import pybullet as p
import shapely
# imports for LMPs
import torch
import base64
from PIL import Image
from io import BytesIO
from openai.error import RateLimitError, APIConnectionError
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from scipy.spatial import distance
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#from vllm import LLM, SamplingParams

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
from cliport.utils import utils


# llm_from_vllm = LLM(
#     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
#     gpu_memory_utilization=0.9,
#     tensor_parallel_size=8,
#     # pipeline_parallel_size=4
# )


class LMP:
    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars,
                 offline_model=None, offline_tokenizer=None, use_vllm: bool = False):
        self._name = name
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._base_prompt = self._cfg['prompt_text']

        self._stop_tokens = list(self._cfg['stop'])

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self.exec_hist = ''

        self.mem = update_memory()

        self.offline_model = offline_model
        self.offline_tokenizer = offline_tokenizer
        self.use_vllm = use_vllm

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = (f"from utils import "
                                         f"{', '.join(self._variable_vars.keys())}")
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(
            self,
            query,
            context='',
            **kwargs
    ):
        if self.use_vllm:
            from vllm import LLM, SamplingParams
        prompt, use_query = self.build_prompt(query, context=context)
        while True:
            try:
                if 'gpt3' in self._llm:
                    code_str = openai.ChatCompletion.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                elif 'gpt4' in self._llm:
                    code_str = openai.ChatCompletion.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                elif 'gemini' in self._llm:
                    gen_model = genai.GenerativeModel(self._cfg['engine']())
                    code_str = gen_model.generate_content(prompt)
                    code_str = code_str.text
                    break
                elif 'llama' in self._llm:
                    prompt += ('\n# Refer to the example tasks, '
                               'now answer this last task question. '
                               'Avoid defining new methods as much as possible.')
                    # llm_model = llama_model[0]
                    # tokenizer = llama_model[1]

                    llm_model_inputs = self.offline_tokenizer.apply_chat_template(
                        [
                            {"role": "system",
                             "content": "You are a task planning assistant "
                                        "who only answers with python code"},
                            {"role": "user",
                             "content": prompt}
                        ],
                        add_generation_prompt=True,
                        # return_tensors="pt"
                    )
                    terminators = [
                        self.offline_tokenizer.eos_token_id,
                        self.offline_tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    if not self.use_vllm:
                        generated_ids = self.offline_model.generate(
                            llm_model_inputs,
                            max_new_tokens=self._cfg['max_tokens'](),
                            # stop_strings=self._stop_tokens,
                            top_p=1.0,
                            top_k=50,
                            temperature=0.7 + self._cfg['temperature'],
                            do_sample=True,
                            eos_token_id=terminators,
                            # tokenizer=tokenizer,
                        )
                        response = generated_ids[0][llm_model_inputs.shape[-1]:]
                        code_str = self.offline_tokenizer.decode(
                            response,
                            skip_special_tokens=True
                        )
                        break
                    else:
                        # Create a sampling params object.
                        sampling_params = SamplingParams(
                            temperature=0.8,
                            top_p=0.95,
                            top_k=50,
                            max_tokens=self._cfg['max_tokens'](),
                            stop_token_ids=terminators,
                        )
                        # Create an LLM.

                        # Generate texts from the prompts.
                        # The output is a list of RequestOutput objects
                        # that contain the prompt, generated text, and other information.
                        outputs = self.offline_model.generate(
                            prompt_token_ids=llm_model_inputs,
                            sampling_params=sampling_params
                        )
                        # Print the outputs.
                        assert len(outputs) == 1, "Size of outputs is not 1."
                        output = outputs[0]
                        generated_text = output.outputs[0].text
                        code_str = generated_text
                        # for output in outputs:
                        #     prompt = output.prompt
                        #     generated_text = output.outputs[0].text
                        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                        break

            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = code_str

        if '```' in code_str:
            code_str = extract_code(code_str)
        
        if self._cfg['include_context'] and context != '' and context not in code_str:
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP {self._name} exec:\n\n{to_log}\n'

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            # print(lvars)
            if self._cfg['return_val_name'] == 'whole_answer':
                return to_exec
            else:
                return lvars[self._cfg['return_val_name']]
        
class LMPV:
    def __init__(self, name, cfg):
        self._name = name
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._base_prompt = self._cfg['prompt_text']

        self.exec_hist = ''

        self.mem = update_memory()

    def clear_exec_hist(self):
        self.exec_hist = ''

    def encode_image(self, image_sources):
        images = []
        for image_source in image_sources:
            image = Image.fromarray(image_source[0])
            if image.mode == 'F':
                if image_source[1] == 'd':
                    image = image.convert('L')
                elif image_source[1] == 'c':
                    image = image.convert('RGB')
            buffered = BytesIO()
            image.save(buffered, format='JPEG')
            img_str = buffered.getvalue()
            images.append(base64.b64encode(img_str).decode('utf-8'))
        return images

    def build_prompt(self, query, context=''):
        variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(
            self,
            query,
            context=None,
            **kwargs
    ):
        prompt, use_query = self.build_prompt(query)
        images = self.encode_image(context)
        while True:
            try:
                if 'gpt4' in self._llm:
                    code_str = openai.ChatCompletion.create(
                        messages=[{"role": "system",
                                   "content": "You are a task completion checking assistant "
                                              "who compares the final observation with initial observation "
                                              "and gives the judge in python code format like "
                                              "'judge = True (or False)'"
                                              },
                                  {"role": "user",
                                   "content": [
                                       {"type": "text", "text": prompt},
                                       {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[0]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[1]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[2]}",
                                            },
                                        },
                                        {
                                           "type": "image_url",
                                            "image_url": {
                                            "url": f"data:image/jpeg;base64,{images[3]}",
                                            },
                                        },
                                    ],
                                },
                            ],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                else:
                    break

            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = code_str

        if '```' in code_str:
            code_str = extract_code(code_str)

        to_exec = code_str
        to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP {self._name} exec:\n\n{to_log}\n'

        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['has_return']:
            # print(lvars)
            try:
                return lvars[self._cfg['return_val_name']]
            except:
                if 'True' in to_exec:
                    return True
                else:
                    return False


class LMPFGen:

    def __init__(
            self,
            cfg,
            fixed_vars,
            variable_vars,
            offline_model=None,
            offline_tokenizer=None,
            use_vllm: bool = False
    ):
        self._cfg = cfg[0]
        self._llm = cfg[1]

        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']

        self.mem = update_memory()

        self.offline_model = offline_model
        self.offline_tokenizer = offline_tokenizer
        self.use_vllm = use_vllm

    def create_f_from_sig(
            self,
            f_name,
            f_sig,
            other_vars=None,
            fix_bugs=False,
            return_src=False,
    ):
        print(f'Creating function: {f_sig}')
        if self.use_vllm:
            from vllm import LLM, SamplingParams

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'
        while True:
            try:
                if 'gpt3' in self._llm:
                    f_src = openai.ChatCompletion.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        stop=self._stop_tokens,
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                elif 'gpt4' in self._llm:
                    f_src = openai.ChatCompletion.create(
                        messages=[{"role": "system",
                                   "content": "You are a task planning assistant "
                                              "who only answers with python code"},
                                  {"role": "user",
                                   "content": prompt}],
                        temperature=self._cfg['temperature'],
                        model=self._cfg['engine'](),
                        max_tokens=self._cfg['max_tokens'](),
                    )['choices'][0]['message']['content'].strip()
                    break
                elif 'gemini' in self._llm:
                    gen_model = genai.GenerativeModel(model)
                    f_src = gen_model.generate_content(prompt)
                    f_src = f_src.text
                    break
                elif 'llama' in self._llm:
                    prompt += ('\n# Refer to the example python method implementing tasks, '
                               'now answer this last method implementing task question.')
                    device = 'cuda'
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_flash_sdp(False)
                    # bitsandbytesconfig = BitsAndBytesConfig(load_in_8bit=True)
                    # tokenizer = AutoTokenizer.from_pretrained(
                    #     self._cfg['engine'](),
                    #     local_files_only=False,
                    #     device_map="auto",
                    #     trust_remote_code=True
                    # )
                    tokenizer = self.offline_tokenizer
                    llm_model_inputs = tokenizer.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": "You are a task planning assistant "
                                           "who only answers with python code."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        add_generation_prompt=True,
                        # return_tensors="pt"
                    )
                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    llm_model = self.offline_model
                    if not self.use_vllm:
                        # llm_model = AutoModelForCausalLM.from_pretrained(
                        #     self._cfg['engine'](),
                        #     local_files_only=False,
                        #     device_map="auto",
                        #     max_memory=self.mem,
                        #     trust_remote_code=True,
                        #     torch_dtype=torch.bfloat16,
                        #     # quantization_config=bitsandbytesconfig,
                        #     # rope_scaling={'type':'dynamic', 'factor':2},
                        # )
                        generated_ids = llm_model.generate(
                            llm_model_inputs,
                            max_new_tokens=self._cfg['max_tokens'](),
                            # stop_strings=self._stop_tokens,
                            top_p=1.0,
                            top_k=50,
                            temperature=0.7 + self._cfg['temperature'],
                            do_sample=True,
                            eos_token_id=terminators,
                            # tokenizer=tokenizer,
                        )
                        response = generated_ids[0][llm_model_inputs.shape[-1]:]
                        f_src = tokenizer.decode(response, skip_special_tokens=True)
                        break
                    else:
                        # Create a sampling params object.
                        sampling_params = SamplingParams(
                            temperature=0.8,
                            top_p=0.95,
                            top_k=50,
                            max_tokens=self._cfg['max_tokens'](),
                            stop_token_ids=terminators,
                        )

                        # Generate texts from the prompts.
                        # The output is a list of RequestOutput objects
                        # that contain the prompt, generated text, and other information.
                        outputs = llm_model.generate(
                            prompt_token_ids=llm_model_inputs,
                            sampling_params=sampling_params
                        )
                        # Print the outputs.
                        assert len(outputs) == 1, "Size of outputs is not 1."
                        output = outputs[0]
                        generated_text = output.outputs[0].text
                        f_src = generated_text
                        # for output in outputs:
                        #     prompt = output.prompt
                        #     generated_text = output.outputs[0].text
                        #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                        break

            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 10s.')
                sleep(10)

        res = f_src

        if '```' in f_src:
            f_src = extract_code(f_src)

        if fix_bugs:
            f_src = openai.Edit.create(
                model='code-davinci-edit-001',
                input='# ' + f_src,
                temperature=0,
                instruction='Fix the bug if there is one. '
                            'Improve readability. '
                            'Keep same inputs and outputs. '
                            'Only small changes. No comments.',
            )['choices'][0]['text'].strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')
        global answer
        answer += f'LLM answer:\n{res}\nLMP FGEN created:\n\n{use_query}\n{f_src}\n'

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        if '```' in code_str:
            code_str = extract_code(code_str)

        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name,
                    f_sig,
                    new_fs,
                    fix_bugs=fix_bugs,
                    return_src=True
                )

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k: v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    if '```' in code_str:
        code_str = extract_code(code_str)

    if 'import' in code_str:
        import_pattern = re.compile(r'^\s*(import .*|from .* import .*)$', re.MULTILINE)
        code_str = import_pattern.sub('', code_str).strip()
    assert '__' not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    # print(f'THE CODE STRING IS\n{code_str}\nEND')
    exec(code_str, custom_gvars, lvars)


def extract_code(res):
    if '```python' in res:
        pattern = r'```python\n(.*?)```'
    elif '```Python' in res:
        pattern = r'```Python\n(.*?)```'
    elif '```' in res:
        pattern = r'```\n(.*?)```'
    else:
        pattern = r'.*'
    code_string = re.search(pattern, res, re.DOTALL)
    if not code_string:
        print('input: ', res)
        raise ValueError('extract failed')
    if pattern == r'.*':
        code_string = code_string.group(0).strip()
    else:
        code_string = code_string.group(1).strip()

    lines = code_string.splitlines()
    if '```' in code_string:
        lines = lines[1:]
    lines = [line for line in lines if line.strip() != '']
    code_string = "\n".join(lines)

    return code_string


class LMP_wrapper():

    def __init__(self, env, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = self._cfg['env']['init_objs']

        self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
        self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['env']['coords']['table_z']
        self.render = render

    def is_obj_visible(self, obj_name):

        return obj_name in self.object_names
    
    def reset(self):
        np.random.seed(self.env._seed)
        random.seed(self.env._seed)
        self.env.reset()

    def get_obj_names(self, id=None):
        if not id:
            return self.object_names[::]
        elif isinstance(id, int):
            for s in self.object_names[::]:
                parts = s.split()  # Split the string by spaces
                if parts and parts[-1].isdigit():  # Check if the last part is a number
                    if int(parts[-1]) == id:
                        return [s]  # Return the matching string        
            raise ValueError(f'no matching obj with id {id}')
        elif isinstance(id, (str, np.str_)):
            return [id]
        elif isinstance(id, (list, tuple, np.ndarray)):
            return list(id)
        else:
            raise ValueError('input should be either int or str')

    def denormalize_xy(self, pos_normalized, size=None):
        pos_normalized = np.array([pos_normalized[1], pos_normalized[0]])
        if not size:
            return pos_normalized * self._range_xy + self._min_xy
        else:
            x = size[0]
            y = size[1]
            min_xy = np.array([-x / 2, -y / 2])
            max_xy = np.array([x / 2, y / 2])
            range_xy = max_xy - min_xy
            return pos_normalized * range_xy + min_xy
        
    def denormalize_bbox(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        left = self.denormalize_xy([x_min,y_max])
        right = self.denormalize_xy([x_max,y_min])
        return [left[0], left[1], right[0], right[1]]

    def get_corner_positions(self):
        unit_square = shapely.geometry.box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
        return corners

    def get_side_positions(self):
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(
            ([self.denormalize_xy(corner) for corner in normalized_side_positions]))
        return side_positions

    def get_obj_pos(self, obj_name, count=1):
        # return the xy position of the object in robot base frame. YHH: Why only xy position?
        return self.env.get_obj_pos(obj_name, count)  # [:2]

    def get_obj_rot(self, obj_name, count=1):

        return self.env.get_obj_rot(obj_name, count)

    def get_obj_positions_np(self, objects):
        if all(type(obj) == int for obj in objects) or all(
                'with obj_id' in obj for obj in objects):
            return [self.get_obj_pos(object)[0] for object in objects]

        positions_dict = {}

        # Retrieve positions for each object type
        for obj in objects:
            if obj not in positions_dict:
                positions_dict[obj] = self.get_obj_pos(obj, -1)

        # Initialize index counters for each object type
        counters = {obj: 0 for obj in positions_dict}

        # List to store the final positions in the required order
        position_list = []

        # Populate the position_list with the correct positions
        for obj in objects:
            pos_index = counters[obj]  # Get the current index for this object type
            position_list.append(positions_dict[obj][pos_index])  # Add the position to the list
            counters[obj] += 1  # Increment the index for this object type

        return position_list

    def get_obj_rotations_np(self, objects):
        if all(type(obj) == int for obj in objects) or all(
                'with obj_id' in obj for obj in objects):
            return [self.get_obj_rot(object)[0] for object in objects]

        rotations_dict = {}

        # Retrieve positions for each object type
        for obj in objects:
            if obj not in rotations_dict:
                rotations_dict[obj] = self.get_obj_rot(obj, -1)

        # Initialize index counters for each object type
        counters = {obj: 0 for obj in rotations_dict}

        # List to store the final positions in the required order
        rotation_list = []

        # Populate the position_list with the correct positions
        for obj in objects:
            pos_index = counters[obj]  # Get the current index for this object type
            rotation_list.append(rotations_dict[obj][pos_index])  # Add the position to the list
            counters[obj] += 1  # Increment the index for this object type

        return rotation_list

    def get_obj_pos_dict(self):
        catalog = {}
        for obj in self.get_obj_names():
            catalog[obj] = [self.get_obj_pos(obj)[0],
                            utils.quatXYZW_to_eulerXYZ(self.get_obj_rot(obj)[0])]
        return catalog

    def get_bbox(self, obj_name):
        # return the axis-aligned object bounding box in robot base frame (not in pixels)
        # the format is (min_x, min_y, max_x, max_y)
        is_zone = False
        is_pallet = False
        scale = 1
        if isinstance(obj_name, (list, np.ndarray, tuple)):
            obj_name = obj_name[0]
        if 'scaled' in self.get_obj_names(obj_name)[0]:
            pattern = r'(\d+(\.\d+)?)x'
            match = re.search(pattern, obj_name)
            scale = float(match.group(1)) if match else 1
        if 'zone' in self.get_obj_names(obj_name)[0]:
            is_zone = True
        if 'pallet' in self.get_obj_names(obj_name)[0]:
            is_pallet = True
        bbox = self.env.get_bounding_box(obj_name)
        size_x = bbox[3] - bbox[0]
        size_y = bbox[4] - bbox[1]
        size_z = bbox[5] - bbox[2]
        size = 50 * scale * np.array([size_x, size_y, size_z]) if is_zone else scale * np.array(
            [size_x, size_y, size_z])
        if is_pallet:
            size *= 0.5
        return tuple(size)

    def get_two_bbox(self, obj_name):
        if isinstance(obj_name, (list, np.ndarray, tuple)):
            obj_name = obj_name[0]
        bbox = self.env.get_bounding_box(obj_name)
        return bbox

    def get_color(self, obj_name):
        for color, rgb in utils.COLORS.items():
            if color in obj_name:
                return rgb

    def pick_place(self, obj, place):
        pick_pos = self.get_obj_pos(obj)[0]
        pick_rot = self.get_obj_rot(obj)[0]
        self.env.step(action={'pose0': (pick_pos, pick_rot), 'pose1': place})

    def put_first_on_second(self, arg1, arg2):
        # put the object with obj_name on top of target
        # target can either be another object name, or it can be an x-y position in robot base frame
        if not arg1 or not arg2:
            print('missing argument')
            return
        if isinstance(arg1, (str, int, np.str_)):
            pick_pos = (self.get_obj_pos(arg1)[0], self.get_obj_rot(arg1)[0])
        else:
            pick = list(arg1)
            if not isinstance(pick[0], (list, tuple, np.ndarray)):
                if len(pick) == 2:
                    pick_pos = ((pick[0], pick[1], 0), (0, 0, 0, 1))
                elif len(pick) == 3:
                    pick_pos = ((pick[0], pick[1], pick[2]), (0, 0, 0, 1))
            else:
                pick_pos = arg1

        if isinstance(arg2, (str, int, np.str_)):
            place_pos = (self.get_obj_pos(arg2)[0], self.get_obj_rot(arg2)[0])
        else:
            place = list(arg2)
            if not isinstance(place[0], (list, tuple, np.ndarray)):
                if len(place) == 2:
                    place_pos = ((place[0], place[1], 0), (0, 0, 0, 1))
                elif len(place) == 3:
                    place_pos = ((place[0], place[1], place[2]), (0, 0, 0, 1))
            else:
                place_pos = arg2
        self.env.step(action={'pose0': pick_pos, 'pose1': place_pos})

    def stack_objects_in_order(self, object_names, targ=None):
        if not object_names:
            return
        if not targ:
            if not isinstance(object_names, (list, tuple, np.ndarray)):
                object_names = [object_names]
            for i in range(len(object_names) - 1):
                self.put_first_on_second(object_names[i + 1], object_names[i])
        else:
            if not isinstance(object_names, (list, tuple, np.ndarray)):
                object_names = [object_names]
            self.put_first_on_second(object_names[0], targ)
            for i in range(len(object_names) - 1):
                self.put_first_on_second(object_names[i + 1], object_names[i])

    def is_target_occupied(self, targ, r=0.02):

        search_range = self.get_obj_names()
        if isinstance(targ, (list, tuple, np.ndarray)):
            if isinstance(targ[0], (list, tuple, np.ndarray)):
                targ = targ[0]
            targ = np.array(targ)[:2]  # force to check in 2d
            dim = len(targ)
            if dim not in [2, 3]:
                raise ValueError("Target position must be either 2D or 3D.")
        elif isinstance(targ, (int, str, np.str_)):
            targ_obj = self.get_obj_names(targ)[0]
            if targ_obj in search_range:
                search_range.remove(targ_obj)
            x, y, _ = self.get_bbox(targ)
            r = 0.5 * np.linalg.norm([x, y])
            targ = self.get_obj_pos(targ)[0]
            targ = np.array(targ)[:2]  # force to check in 2d
            dim = len(targ)
        else:
            raise ValueError("Target must be only one position, id, or name.")

        # Define the center of the circle/sphere
        center = np.array(targ)
        occupied_objects = []

        for obj in search_range:
            # Get the bounding box of the object
            bbox = self.get_two_bbox(obj)

            # Extract bounding box coordinates
            if dim == 2:
                x_min, y_min, _, x_max, y_max, _ = bbox
                closest_point = np.array([
                    np.clip(center[0], x_min, x_max),
                    np.clip(center[1], y_min, y_max)
                ])
            elif dim == 3:
                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                closest_point = np.array([
                    np.clip(center[0], x_min, x_max),
                    np.clip(center[1], y_min, y_max),
                    np.clip(center[2], z_min, z_max)
                ])

            # Calculate the distance from the center to the closest point on the bounding box
            distance_to_bbox = distance.euclidean(center, closest_point)

            # Check if this distance is less than or equal to the radius
            if distance_to_bbox <= r:
                occupied_objects.append(obj)

        return occupied_objects

    def get_random_free_pos(self, targ=None, r=0.02, search_area=None, grid_size=0.002):
        # local function to convert item of listed-targ 
        def convert_to_3d_pos(item):
            if isinstance(item, (list, tuple, np.ndarray)):
                # Handling pos
                if len(item) == 2 and all(isinstance(coord, (int, float)) for coord in item):
                    return [item[0], item[1], 0]
                elif len(item) == 3 and all(isinstance(coord, (int, float)) for coord in item):
                    return list(item)
                # Handling pos_rot
                elif len(item) == 2 and all(
                        isinstance(coord, (list, tuple, np.ndarray)) for coord in item) and len(
                    item[1]) == 4:
                    pos = item[0]
                    if len(pos) == 2:
                        return [pos[0], pos[1], 0]
                    elif len(pos) == 3:
                        return list(pos)
                    else:
                        raise ValueError("pos in pos_rot must be 2D or 3D")
            elif isinstance(item, (str, int, np.str_)):
                # Use get_obj_pos for str or int
                x, y, _ = self.get_bbox(item)
                radius = 0.5 * np.linalg.norm([x, y])
                return (self.get_obj_pos(item)[0], radius)
            else:
                raise ValueError("Unsupported type for conversion to 3D position")

        def convert_targ_to_3d_pos_list(targ):
            # Check if targ is a list that is not a nested list (indicating a single pos)
            if isinstance(targ, (list, tuple, np.ndarray)):
                if all(isinstance(coord, float) for coord in targ) and (
                        len(targ) == 2 or len(targ) == 3):
                    # It's a single position
                    return [convert_to_3d_pos(targ)]
                elif all(isinstance(coord, (list, tuple, np.ndarray)) for coord in targ) and len(
                        targ) == 2 and len(targ[1]) == 4:
                    # It's a single pos_rot
                    return [convert_to_3d_pos(targ)]
                elif all(isinstance(coord, float) for coord in targ) and len(targ) == 4:
                    raise ValueError("Search area")
                else:
                    # It's a list of elements
                    pos_list = [convert_to_3d_pos(item) for item in targ]
                    return pos_list
            else:
                # targ is a single item (str, int)
                return [convert_to_3d_pos(targ)]

        if search_area:
            x_min, y_min, x_max, y_max = search_area
        else:
            x_min, y_min = self._cfg['env']['coords']['top_left']
            x_max, y_max = self._cfg['env']['coords']['bottom_right']

        if targ:
            try:
                targ = convert_targ_to_3d_pos_list(targ)
            except ValueError as e:
                if str(e) == "Search area":
                    x_min, y_min, x_max, y_max = targ
                    targ = []
                else:
                    raise
        else:
            targ = []

        # Generate a grid of potential positions
        x_coords = np.arange(x_min, x_max, grid_size)
        y_coords = np.arange(y_min, y_max, grid_size)
        potential_positions = [(x, y, 0.001) for x in x_coords for y in y_coords]

        # Remove positions that are too close to the target
        for sub_targ in targ:
            if isinstance(sub_targ, tuple):
                potential_positions = [
                    pos for pos in potential_positions
                    if np.linalg.norm(np.array(pos) - np.array(sub_targ[0])) > sub_targ[1]
                ]
            else:
                potential_positions = [
                    pos for pos in potential_positions
                    if np.linalg.norm(np.array(pos) - np.array(sub_targ)) > r
                ]

        # Remove positions that are occupied by objects
        free_positions = [
            pos for pos in potential_positions
            if not self.is_target_occupied(pos, r)
        ]

        if not free_positions:
            print('no suitable position')
            return None  # No free position found

        # Randomly select a free position
        pos = random.choice(free_positions)
        return [list(pos), [0, 0, 0, 1]]

    def get_robot_pos(self):
        # return robot end-effector xy position in robot base frame
        return self.env.get_ee_pos()

    def goto_pos(self, position):
        # move the robot end-effector to the desired xy position while maintaining same z
        ee_xyz = self.env.get_ee_pos()
        if len(position) == 2:
            if not isinstance(position[0], (tuple, list, np.ndarray)):
                position_xyz = [np.concatenate([position, ee_xyz[-1]]), [0, 0, 0, 1]]
        elif len(position) == 3:
            position_xyz = [list(position), [0, 0, 0, 1]]
        else:
            position_xyz = position
        while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
            self.env.movep(position_xyz)
            self.env.step_simulation()
            ee_xyz = self.env.get_ee_pos()

    def follow_traj(self, traj):
        for pos in traj:
            self.goto_pos(pos)

    def get_corner_positions(self):
        # TODO: repetitive name.
        normalized_corners = np.array([
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0]
        ])
        return np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))

    def get_side_positions(self):
        # TODO: repetitive name.
        normalized_sides = np.array([
            [0.5, 1],
            [1, 0.5],
            [0.5, 0],
            [0, 0.5]
        ])
        return np.array(([self.denormalize_xy(side) for side in normalized_sides]))

    def get_corner_name(self, pos):
        corner_positions = self.get_corner_positions()
        if len(pos) == 2:
            corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        elif len(pos) == 3:
            pos = pos[:2]
            corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        return ['top left corner', 'top right corner', 'bottom left corner', 'botom right corner'][
            corner_idx]

    def get_side_name(self, pos):
        side_positions = self.get_side_positions()
        side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
        return ['top side', 'right side', 'bottom side', 'left side'][side_idx]


def set_llm_model(llm_model_name):
    """ globally set llm-model"""
    global model
    model = llm_model_name


def set_max_token(token):
    global max_token
    max_token = token


def update_model():
    global model
    return model


def update_max_token():
    global max_token
    return max_token


def update_memory():
    global mem
    memory = {}
    if not mem[0]:
        for id in mem[1]:
            memory[id] = f'{mem[2]}GB'
    else:
        for id, memo in mem[1]:
            memory[id] = f'{memo}GB'
    return memory


def say(msg):
    global answer
    msg = f'robot says: {msg}'
    answer += (msg + '\n')
    print(msg)


def check_obj():
    object_ids = []
    num_bodies = p.getNumBodies()
    for i in range(num_bodies):
        object_id = p.getBodyUniqueId(i)
        object_ids.append(object_id)
    return object_ids


def setup_LMP(env, cfg_tabletop, llm='gpt4',
              offline_model=None, offline_tokenizer=None, use_vllm=False):
    # LMP env wrapper
    cfg_tabletop = copy.deepcopy(cfg_tabletop)
    cfg_tabletop['env'] = dict()
    cfg_tabletop['env']['init_objs'] = env.object_list
    cfg_tabletop['env']['coords'] = lmp_tabletop_coords
    cfg_tabletop['llm'] = llm
    LMP_env = LMP_wrapper(env, cfg_tabletop)
    # creating APIs that the LMPs can interact with
    fixed_vars = {
        'np': np,
        'utils': utils,
        'itertools': itertools
    }
    fixed_vars.update({
        name: getattr(shapely.geometry, name)
        for name in shapely.geometry.__all__
    })

    fixed_vars.update({
        name: getattr(shapely.affinity, name)
        for name in shapely.affinity.__all__
    })
    variable_vars = {
        k: getattr(LMP_env, k)
        for k in [
            'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible', 'denormalize_xy',
            'put_first_on_second', 'get_obj_names', 'get_obj_rot', 'get_obj_positions_np',
            'get_corner_name', 'get_side_name', 'get_obj_rotations_np', 'goto_pos',
            'is_target_occupied', 'get_random_free_pos', 'stack_objects_in_order',
            'get_obj_pos_dict', 'denormalize_bbox', 'reset',
        ]
    }
    variable_vars['say'] = say

    # creating the function-generating LMP
    lmp_fgen = LMPFGen(
        (cfg_tabletop['lmps']['fgen'], cfg_tabletop['llm']),
        fixed_vars,
        variable_vars,
        offline_model=offline_model,
        offline_tokenizer=offline_tokenizer,
        use_vllm=use_vllm,
    )

    # creating other low-level LMPs
    variable_vars.update({
        k: LMP(
            k,
            (cfg_tabletop['lmps'][k], cfg_tabletop['llm']),
            lmp_fgen,
            fixed_vars,
            variable_vars,
            offline_model=offline_model,
            offline_tokenizer=offline_tokenizer,
            use_vllm=use_vllm,
        )
        for k in [
            'parse_obj_name', 'parse_position',
            'parse_question', 'transform_shape_pts', 'parse_completion'
        ]
    })

    # creating the LMP that deals w/ high-level language commands
    lmp_tabletop_ui = LMP(
        'tabletop_ui',
        (cfg_tabletop['lmps']['tabletop_ui'], cfg_tabletop['llm']),
        lmp_fgen,
        fixed_vars,
        variable_vars,
        offline_model=offline_model,
        offline_tokenizer=offline_tokenizer,
        use_vllm=use_vllm,
    )

    return lmp_tabletop_ui


@hydra.main(config_path='./cfg', config_name='cap')
def main(cfg):
    """
    Use the same LLM model for both LMP and LMPFGen.
    If using different LLMs for them,
    two offline loaded models should be passed to the function setup_LMP.
    """
    llm_model, llm_tokenizer = None, None
    use_vllm = cfg["use_vllm"]
    if use_vllm:
        from vllm import LLM, SamplingParams
    n_gpus = cfg["n_gpus"]

    if 'gpt' in cfg['gpt_model'] and '3' in cfg['gpt_model']:
        openai.api_key = cfg['openai_key']
        llm = 'gpt3'
        llm_model_name = "gpt-3.5-turbo-16k"
        set_llm_model(llm_model_name)
    elif 'gpt' in cfg['gpt_model'] and '4' in cfg['gpt_model']:
        openai.api_key = cfg['openai_key']
        llm = 'gpt4'
        llm_model_name = "gpt-4o-mini"
        set_llm_model(llm_model_name)
    elif 'gemini' in cfg['gpt_model']:
        genai.configure(api_key=cfg['genai_key'])
        llm = 'gemini'
        llm_model_name = "gemini-1.5-pro"
        set_llm_model(llm_model_name)
    elif 'llama' in cfg['gpt_model']:
        llm = 'llama'
        llm_model_name = cfg['llama_model_name']
        set_llm_model(llm_model_name)
        # set_llm_model("meta-llama/Meta-Llama-3.1-8B-Instruct")  # ../CodeLlama-7b-Python-hf/
        # set_llm_model(f"meta-llama/{llama_model_name}")  # ../CodeLlama-7b-Python-hf/
        # set_llm_model("../Meta-Llama-3-8B-Instruct/")  # ../CodeLlama-7b-Python-hf/
        # global llama_model
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(False)
        # bitsandbytesconfig = BitsAndBytesConfig(load_in_8bit=True)
        llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            local_files_only=False,
            device_map="auto",
            trust_remote_code=True,
        )
        if use_vllm:
            llm_model = LLM(
                # model="meta-llama/Meta-Llama-3.1-70B-Instruct",
                model=llm_model_name,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=n_gpus,
                # pipeline_parallel_size=4
            )
        else:
            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                local_files_only=False,
                device_map="auto",
                max_memory=update_memory(),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                # quantization_config=bitsandbytesconfig,
                rope_scaling={'type': 'dynamic', 'factor': 2},
            )
    else:
        llm = cfg['gpt_model']
        raise ValueError(f'Unknown LLM {llm}')
    set_max_token(2048)

    print(f"Use the model: {llm_model_name}")
    print(f"If use offline model, whether use vLLM to load offline model: {use_vllm}")

    # initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )

    global answer

    cfg['task'] = cfg['task'].replace("_", "-")
    mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']
    check = cfg['check']
    use_VLM = cfg['check_using_VLM']
    manual_eval = cfg['manual_eval']
    note = (" Write code to complete the task")
    if record:
        print('Recording...')
        # Collect training data from oracle demonstrations.
    if mode == 'cap':
        data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], mode))
        dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
        print(f"Saving to: {data_path} if save_data")
        print(f"Mode: {mode}")

        seed = dataset.max_seed
        max_eps = 1 * cfg['n']
        if seed < 0:
            seed = -1

        if 'regenerate_data' in cfg:
            dataset.n_episodes = 0

        curr_run_eps = 0

        while dataset.n_episodes < cfg['n'] and curr_run_eps < max_eps:
            answer = ''
            
            # for epi_idx in range(cfg['n']):
            episode = []
            seed += 2

            # Set seeds.
            np.random.seed(seed)
            random.seed(seed)
            task = tasks.names[cfg['task']]()

            env.seed(seed)
            print('CAP run: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))
            try:
                curr_run_eps += 1  # make sure exits the loop
                # env.seed(seed)
                env.set_task(task)
                env.reset()
                assert env.object_list
                # assert len(env.object_list) == len(check_obj()) - 5

                # Start video recording
                if record:
                    env.start_rec(
                        f'{dataset.n_episodes + 1:06d}_CAP') if not check else env.start_rec(
                        f'{dataset.n_episodes + 1:06d}_CAP_check')

                # Rollout LLM policy
                goal = task.goal + ' Finally check the completion of task.' if check else task.goal
                goal = goal + note

                lmp_tabletop_ui = setup_LMP(env, cfg_tabletop, llm,
                                            offline_model=llm_model,
                                            offline_tokenizer=llm_tokenizer,
                                            use_vllm=use_vllm)
                lmp_tabletop_ui(goal, f'objects = {env.object_list}')

                if record:
                    env.end_rec()

                obs = env._get_obs()
                info = env.info

                file_path = (
                    os.path.join(
                        data_path,
                        'answers',
                        f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}.txt'
                    )
                    if not check
                    else os.path.join(
                        data_path,
                        'answers',
                        f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}_check.txt'
                    )
                )
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(answer)
                episode.append((obs, None, 1, info))

            except:
                err = str(traceback.format_exc())
                answer += (err + '\n')
                to_print = highlight(f"{err}", PythonLexer(), TerminalFormatter())
                print(to_print)
                if record:
                    env.end_rec()

                obs = env._get_obs()
                info = env.info

                file_path = (
                    os.path.join(
                        data_path,
                        'answers',
                        f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}.txt'
                    )
                    if not check
                    else os.path.join(
                        data_path,
                        'answers',
                        f'{dataset.n_episodes + 1:06d}_{cfg["gpt_model"]}_check.txt'
                    )
                )
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(answer)
                episode.append((obs, None, 0, info))
                continue

                # Only save completed demonstrations.
            if save_data:
                dataset.add(seed, episode)

            if hasattr(env, 'blender_recorder'):
                print("blender pickle saved to ",
                      '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))
                env.blender_recorder.save(
                    '{}/blender_demo_{}.pkl'.format(data_path, dataset.n_episodes))

    elif mode == 'test':
        check_note = (' Finally check the completion of task using API parse_completion()'
                      if llm == 'llama'
                      else ' Finally check the completion of task.')
        success = 0
        n_demos = cfg['n']
        data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], mode))
        env.video_path = os.path.join(data_path, 'videos')
        if not cfg['random']:
            dataset = RavensDataset(data_path, cfg, n_demos=n_demos, augment=False, random=False)
            print(f"Testing on: {data_path}")
            print(f"Mode: {mode} Random: False")
            files = os.listdir(os.path.join(data_path, 'action'))
            files.sort()
            file_path = None
            for i in range(0, n_demos):
                answer = ''
                print(f'Test: {i + 1}/{n_demos} on {files[i]}')
                try:
                    episode, seed = dataset.load(i)
                except:
                    print(f"skip bad example {files[i]}")
                    continue
                np.random.seed(seed)
                random.seed(seed)
                task = tasks.names[cfg['task']]()

                env.seed(seed)
                env.set_task(task)
                try:
                    env.reset()  # TODO: think whether it conflicts with the loaded scene.
                    assert env.object_list
                    # assert len(env.object_list) == len(check_obj()) - 5

                    imagec = (env._get_obs()['color'][0], 'c')
                    imaged = (env._get_obs()['depth'][0], 'd')

                    # Start video recording
                    if record:
                        env.start_rec(f'{seed:06d}_CAP_test') if not check else env.start_rec(
                            f'{seed:06d}_CAP_test_check')

                    # Rollout LLM policy
                    if check and not use_VLM:
                        goal = task.goal + check_note 
                    else:
                        goal = task.goal + note

                    lmp_tabletop_ui = setup_LMP(
                        env,
                        cfg_tabletop,
                        llm,
                        offline_model=llm_model,
                        offline_tokenizer=llm_tokenizer,
                        use_vllm=use_vllm,
                    )
                    plan = lmp_tabletop_ui(goal, f'objects = {env.object_list}')

                    if check and use_VLM:
                        imagece = (env._get_obs()['color'][0], 'c')
                        imagede = (env._get_obs()['depth'][0], 'd')
                        lmp_check = LMPV(
                            'VLM_ui',
                            (cfg_tabletop['lmps']['VLM'], 'gpt4')
                            )
                        complete = lmp_check('Here are the inital and final observations in RGB and depth, '
                                             f'judge whether the robot has completed the task "{task.goal}"', 
                                             [imagec,imaged,imagece,imagede])
                        if not complete:
                            np.random.seed(env._seed)
                            random.seed(env._seed)
                            env.reset()
                            lmp_tabletop_ui(f'The code you have generated just now was judged to fail completing the task, please try again. '
                                            'You do not need to check completion this time', plan)

                    if manual_eval:
                        imagece = env._get_obs()['color'][0]
                        image = Image.fromarray(imagece)
                        if image.mode == 'F':
                            image = image.convert('RGB')
                        pic_path = (os.path.join(
                            data_path,
                            'finalsnapshot',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.jpg'
                            )
                        if not check
                        else os.path.join(
                            data_path,
                            'finalsnapshot',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.jpg'
                            )
                        )
                        os.makedirs(os.path.dirname(pic_path), exist_ok=True)
                        image.save(pic_path, format='JPEG')

                    if record:
                        env.end_rec()

                    reward = task._rewards
                    done = task.done()
                    if done:
                        success += 1
                    print(f'Total Reward: {reward:.3f} | Done: {done}\n')
                    answer += f'Total Reward: {reward:.3f} | Done: {done}\n'

                    file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(answer)
                except:
                    answer += f'\n **Task Failed** \n'
                    err = str(traceback.format_exc())
                    answer += (err + '\n')
                    to_print = highlight(f"{err}", PythonLexer(), TerminalFormatter())
                    print(to_print)
                    if record:
                        env.end_rec()
                    file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(answer)
            file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'summary_{n_demos}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'summary_{n_demos}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
            result = f'{success} successes in {n_demos} tests'
            print(result)
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(result)

        else:
            dataset = RavensDataset(data_path, cfg, n_demos=n_demos, augment=False, random=True)
            print(f"Testing on: {data_path}")
            print(f"Mode: {mode} Random: True")
            files = os.listdir(os.path.join(data_path, 'action'))
            files.sort()
            file_path = None
            for i, idx in enumerate(dataset.sample_set):
                answer = ''
                print(f'Test: {i + 1}/{n_demos} on {files[idx]}')
                try:
                    episode, seed = dataset.load(idx)
                except:
                    print(f"skip bad example {files[idx]}")
                    continue
                np.random.seed(seed)
                random.seed(seed)
                task = tasks.names[cfg['task']]()

                env.seed(seed)
                env.set_task(task)
                try:
                    env.reset()
                    assert env.object_list
                    # assert len(env.object_list) == len(check_obj()) - 5

                    imagec = (env._get_obs()['color'][0], 'c')
                    imaged = (env._get_obs()['depth'][0], 'd')

                    # Start video recording
                    if record:
                        env.start_rec(f'{seed:06d}_CAP_test') if not check else env.start_rec(
                            f'{seed:06d}_CAP_test_check')

                    # Rollout LLM policy
                    if check and not use_VLM:
                        goal = task.goal + check_note 
                    else:
                        goal = task.goal + note

                    lmp_tabletop_ui = setup_LMP(
                        env,
                        cfg_tabletop,
                        llm,
                        offline_model=llm_model,
                        offline_tokenizer=llm_tokenizer,
                        use_vllm=use_vllm,
                    )
                    plan = lmp_tabletop_ui(goal, f'objects = {env.object_list}')

                    if check and use_VLM:
                        imagece = (env._get_obs()['color'][0], 'c')
                        imagede = (env._get_obs()['depth'][0], 'd')
                        lmp_check = LMPV(
                            'VLM_ui',
                            (cfg_tabletop['lmps']['VLM'], 'gpt4')
                            )
                        complete = lmp_check('Here are the inital and final observations in RGB and depth, '
                                             f'judge whether the robot has completed the task "{task.goal}"', 
                                             [imagec,imaged,imagece,imagede])
                        if not complete:
                            np.random.seed(env._seed)
                            random.seed(env._seed)
                            env.reset()
                            lmp_tabletop_ui(f'The code you have generated just now was judged to fail completing the task, please try again. '
                                            'You do not need to check completion this time', plan) 

                    if manual_eval:
                        imagece = env._get_obs()['color'][0]
                        image = Image.fromarray(imagece)
                        if image.mode == 'F':
                            image = image.convert('RGB')
                        pic_path = (os.path.join(
                            data_path,
                            'finalsnapshot',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.jpg'
                            ) 
                        if not check
                        else os.path.join(
                            data_path,
                            'finalsnapshot',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.jpg'
                            )
                        )
                        os.makedirs(os.path.dirname(pic_path), exist_ok=True)
                        image.save(pic_path, format='JPEG')    

                    if record:
                        env.end_rec()

                    reward = task._rewards
                    done = task.done()
                    if done:
                        success += 1
                    print(f'Total Reward: {reward:.3f} | Done: {done}\n')
                    answer += f'Total Reward: {reward:.3f} | Done: {done}\n'

                    file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(answer)

                except:
                    answer += f'\n **Task Failed** \n'
                    err = str(traceback.format_exc())
                    answer += (err + '\n')
                    to_print = highlight(f"{err}", PythonLexer(), TerminalFormatter())
                    print(to_print)
                    if record:
                        env.end_rec()
                    file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'{seed:06d}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(answer)
            file_path = (
                        os.path.join(
                            data_path,
                            'answers',
                            f'summary_{n_demos}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test.txt'
                        )
                        if not check
                        else os.path.join(
                            data_path,
                            'answers',
                            f'summary_{n_demos}_{cfg["gpt_model"]}_{cfg["llama_model_name"]}_test_check.txt'
                        )
                    )
            result = f'{success} successes in {n_demos} tests'
            print(result)
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(result)

    elif mode == 'debug':
        # ---------------------------------------------------------------------------
        # TODO: Set Seed and Method
        # ---------------------------------------------------------------------------

        seed = 0

        # ---------------------------------------------------------------------------
        # End of Setting
        # ---------------------------------------------------------------------------

        data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], 'test'))
        env.video_path = os.path.join(data_path, 'videos')
        np.random.seed(seed)
        random.seed(seed)
        task = tasks.names[cfg['task']]()

        env.seed(seed)
        env.set_task(task)
        try:
            env.reset()
            assert env.object_list
            # Start video recording
            if record:
                env.start_rec(f'{seed:06d}_CAP_debug')

            dummy_cfg = {}
            dummy_cfg['env'] = dict()
            dummy_cfg['env']['init_objs'] = env.object_list
            dummy_cfg['env']['coords'] = lmp_tabletop_coords
            dummy_cfg['llm'] = 'gpt'
            dummy = LMP_wrapper(env, dummy_cfg)
            
            # Rollout LLM policy
            # ---------------------------------------------------------------------------
            # TODO: Set Execution Code
            # ---------------------------------------------------------------------------
            
            # ---------------------------------------------------------------------------
            # End of Execution Code
            # ---------------------------------------------------------------------------

            if record:
                env.end_rec()
            reward = task._rewards
            done = task.done()
            print(f'Total Reward: {reward:.3f} | Done: {done}\n')

        except:
            err = str(traceback.format_exc())
            to_print = highlight(f"{err}", PythonLexer(), TerminalFormatter())
            print(to_print)
            if record:
                env.end_rec()

    else:
        raise ValueError('mode must be in [cap, test, debug]')


model = "gpt-3.5-turbo-16k"
max_token = 2048
answer = ''
mem = (0, [0], 32)
# llama_model = (None, None)


cfg_tabletop = {
    'lmps': {
        'tabletop_ui': {
            'prompt_text': open(f"prompts/capravens/prompt_tabletop_ui.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': True,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'whole_answer',
        },
        'parse_obj_name': {
            'prompt_text': open(f"prompts/capravens/prompt_parse_obj_name.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'parse_position': {
            'prompt_text': open(f"prompts/capravens/prompt_parse_position.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'parse_question': {
            'prompt_text': open(f"prompts/capravens/prompt_parse_question.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'objects = ['],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'ret_val',
        },
        'transform_shape_pts': {
            'prompt_text': open(f"prompts/capravens/prompt_transform_shape_pts.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'new_shape_pts',
        },
        'fgen': {
            'prompt_text': open(f"prompts/capravens/prompt_fgen.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# define function: ',
            'query_suffix': '.',
            'stop': ['# define', '# example'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
        },
        'parse_completion': {
            'prompt_text': open(f"prompts/capravens/prompt_parse_completion.txt").read(),
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0,
            'query_prefix': '# ',
            'query_suffix': '.',
            'stop': ['#', 'object_positions = {'],
            'maintain_session': False,
            'debug_mode': False,
            'include_context': True,
            'has_return': True,
            'return_val_name': 'judge',
        },
        'VLM': {
            'prompt_text': '',
            'engine': update_model,
            'max_tokens': update_max_token,
            'temperature': 0.7,
            'query_prefix': '# ',
            'query_suffix': '.',
            'maintain_session': True,
            'debug_mode': False,
            'has_return': True,
            'return_val_name': 'judge',
        },
    }
}

lmp_tabletop_coords = {
    'top_left': (0.25, -0.5),
    'top_side': (0.25, 0),
    'top_right': (0.25, 0.5),
    'left_side': (0.5, -0.5),
    'middle': (0.5, 0),
    'right_side': (0.5, 0.5),
    'bottom_left': (0.75, -0.5),
    'bottom_side': (0.75, 0),
    'bottom_right': (0.75, 0.5),
    'table_z': 0.0,
}

if __name__ == '__main__':
    main()
