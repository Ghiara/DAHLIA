import os
import random
import traceback

from gensim.utils import (
    save_text,
    add_to_txt,
    extract_code,
    extract_dict,
    extract_list,
    extract_assets,
    format_dict_prompt,
    generate_feedback,
)


class Agent:
    """
    class that design new tasks and codes for simulation environments
    """

    def __init__(self, cfg, memory):
        self.cfg = cfg
        self.model_output_dir = cfg["model_output_dir"]
        self.prompt_folder = f"prompts/{cfg['prompt_folder']}"
        self.memory = memory
        self.chat_log = memory.chat_log
        self.use_template = cfg['use_template']

    def propose_task(self, proposed_task_names):
        """Language descriptions for the task"""
        add_to_txt(self.chat_log, "================= Task and Asset Design!", with_print=True)

        if self.use_template:
            task_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_task.txt").read()
            task_asset_replacement_str = format_dict_prompt(self.memory.online_asset_buffer,
                                                            self.cfg['task_asset_candidate_num'])
            task_prompt_text = task_prompt_text.replace("TASK_ASSET_PROMPT",
                                                        task_asset_replacement_str)

            task_desc_replacement_str = format_dict_prompt(self.memory.online_task_buffer,
                                                           self.cfg[
                                                               'task_description_candidate_num'])
            print("prompt task description candidates:")
            print(task_desc_replacement_str)
            task_prompt_text = task_prompt_text.replace("TASK_DESCRIPTION_PROMPT",
                                                        task_desc_replacement_str)

            if len(self.cfg['target_task_name']) > 0:
                task_prompt_text = task_prompt_text.replace("TARGET_TASK_NAME",
                                                            self.cfg['target_task_name'])

            if len(self.cfg['target_task_description']) > 0:
                task_prompt_text = task_prompt_text.replace("TARGET_TASK_DISP",
                                                            self.cfg['target_task_description'])
            else:
                task_prompt_text = task_prompt_text.replace("The aim is to TARGET_TASK_DISP.", "")

            # print("Template Task PROMPT: ", task_prompt_text)
        else:
            task_prompt_text = open(f"{self.prompt_folder}/cliport_prompt_task.txt").read()

        # maximum number
        print("online_task_buffer size:", len(self.memory.online_task_buffer))
        total_tasks = self.memory.online_task_buffer

        MAX_NUM = 10
        if len(total_tasks) > MAX_NUM:
            total_tasks = dict(random.sample(total_tasks.items(), MAX_NUM))

        task_prompt_text = task_prompt_text.replace("PAST_TASKNAME_TEMPLATE",
                                                    format_dict_prompt(total_tasks))

        res = generate_feedback(
            task_prompt_text,
            temperature=self.cfg["gpt_temperature"],
            interaction_txt=self.chat_log,
        )

        # Extract dictionary for task name, descriptions, and assets
        task_def = extract_dict(res, prefix="new_task")
        try:
            exec(task_def, globals())
            self.new_task = new_task
            return new_task
        except:
            self.new_task = {"task-name": "dummy", "assets-used": [], "task_descriptions": ""}
            print(str(traceback.format_exc()))
            return self.new_task

    def propose_assets(self):
        """Asset Generation. Not used for now."""
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_asset_template.txt"):
            add_to_txt(self.chat_log, "================= Asset Generation!", with_print=True)
            asset_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_asset_template.txt").read()

            if self.use_template:
                asset_prompt_text = asset_prompt_text.replace("TASK_NAME_TEMPLATE",
                                                              self.new_task["task-name"])
                asset_prompt_text = asset_prompt_text.replace("ASSET_STRING_TEMPLATE",
                                                              str(self.new_task["assets-used"]))
                print("Template Asset PROMPT: ", asset_prompt_text)

            res = generate_feedback(asset_prompt_text, temperature=0,
                                    interaction_txt=self.chat_log)
            print("Save asset to:", self.model_output_dir, task_name + "_asset_output")
            save_text(self.model_output_dir, f'{self.new_task["task-name"]}_asset_output', res)
            asset_list = extract_assets(res)
            # save_urdf(asset_list)
        else:
            asset_list = {}
        return asset_list

    def api_review(self):
        """review the task api"""
        if os.path.exists(f"{self.prompt_folder}/cliport_prompt_api_template.txt"):
            add_to_txt(
                self.chat_log, "================= API Preview!", with_print=True)
            api_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_api_template.txt").read()
            if "task-name" in self.new_task:
                api_prompt_text = api_prompt_text.replace("TASK_NAME_TEMPLATE",
                                                          self.new_task["task-name"])
            api_prompt_text = api_prompt_text.replace("TASK_STRING_TEMPLATE", str(self.new_task))
            print('API REVIEW new_task: ', str(self.new_task))

            res = generate_feedback(
                api_prompt_text, temperature=0, interaction_txt=self.chat_log)

    def template_reference_prompt(self):
        """ select which code reference to reference """
        if os.path.exists(
                f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            self.chat_log = add_to_txt(self.chat_log, "================= Code Reference!",
                                       with_print=True)
            code_reference_question = open(
                f'{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt').read()
            code_reference_question = code_reference_question.replace("TASK_NAME_TEMPLATE",
                                                                      self.new_task["task-name"])
            code_reference_question = code_reference_question.replace("TASK_CODE_LIST_TEMPLATE",
                                                                      str(list(
                                                                          self.memory.online_code_buffer.keys())))
            code_reference_question = code_reference_question.replace("TASK_STRING_TEMPLATE",
                                                                      str(self.new_task))

            res = generate_feedback(code_reference_question, temperature=0.,
                                    interaction_txt=self.chat_log)
            code_reference_cmd = extract_list(res, prefix='code_reference')
            exec(code_reference_cmd, globals())
            task_code_reference_replace_prompt = ''
            for key in code_reference:
                if key in self.memory.online_code_buffer:
                    task_code_reference_replace_prompt += f'```\n{self.memory.online_code_buffer[key]}\n```\n\n'
                else:
                    print("missing task reference code:", key)

        return task_code_reference_replace_prompt

    def implement_task(self):
        """Generate Code for the task"""

        if self.use_template or os.path.exists(
                f"{self.prompt_folder}/cliport_prompt_code_reference_selection_template.txt"):
            code_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_code_candidate_template.txt").read()
            code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE",
                                                        self.new_task["task-name"])
            task_code_reference_replace_prompt = self.template_reference_prompt()
            code_prompt_text = code_prompt_text.replace("TASK_CODE_REFERENCE_TEMPLATE",
                                                        task_code_reference_replace_prompt)
            code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE",
                                                        str(self.new_task))  ########

        elif os.path.exists(f"{self.prompt_folder}/cliport_prompt_code_split_template.txt"):
            code_prompt_text = open(
                f"{self.prompt_folder}/cliport_prompt_code_split_template.txt").read()
            code_prompt_text = code_prompt_text.replace("TASK_NAME_TEMPLATE",
                                                        self.new_task["task-name"])
            self.chat_log = add_to_txt(self.chat_log, "================= Code Generation!",
                                       with_print=True)
            code_prompt_text = code_prompt_text.replace("TASK_STRING_TEMPLATE", str(self.new_task))
            # code_prompt_text = code_prompt_text.replace("TASK_CODE_REFERENCE_TEMPLATE", task_code_reference_replace_prompt) ########

        res = generate_feedback(
            code_prompt_text, temperature=0, interaction_txt=self.chat_log)

        code, task_name = extract_code(res)
        print("Save code to:", self.model_output_dir, task_name + "_code_output")
        save_text(self.model_output_dir, task_name + "_code_output", code)

        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None

        return code, task_name

    def syntax_task(self, code, err):
        """Final syntax correction of Code for the task"""
        code_prompt_text = open(r"prompts/code_syntax_correction_template.txt").read()

        self.chat_log = add_to_txt(self.chat_log, "================= Code Syntax!",
                                   with_print=True)
        code_prompt_text = code_prompt_text.replace("TASK_CODE", code)
        code_prompt_text = code_prompt_text.replace("TASK_CODE_ERROR", err)

        res = generate_feedback(
            code_prompt_text, temperature=0, interaction_txt=self.chat_log)

        code, task_name = extract_code(res)
        print("Save syntaxed code to:", self.model_output_dir, task_name + "_code_output")
        save_text(self.model_output_dir, task_name + "_syntaxed_code_output", code)

        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None

        return code

    def runtime_task(self, count, err):
        """Final syntax correction of Code for the task"""
        code_prompt_text = open(r"prompts/code_runtime_correction_template.txt").read()

        self.chat_log = add_to_txt(self.chat_log, "================= Code Runtime!",
                                   with_print=True)
        # code_prompt_text = code_prompt_text.replace("TASK_CODE", code)
        code_prompt_text = code_prompt_text.replace("TASK_CODE_ERROR", err)

        res = generate_feedback(
            code_prompt_text, temperature=0, interaction_txt=self.chat_log)

        code, task_name = extract_code(res)
        print("Save runtimeed code to:", self.model_output_dir, task_name + "_code_output")
        save_text(self.model_output_dir, task_name + f"_runtimed_code_output_{count}", code)

        if len(task_name) == 0:
            print("empty task name:", task_name)
            return None

        return code
