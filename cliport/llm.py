import logging
import re
from time import sleep
from typing import Union, List, Dict, Any

import openai
import torch
from openai.error import RateLimitError, APIConnectionError, InvalidRequestError, APIError
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

OPEN_SOURCE_LLMs = ["meta-llama/Llama-2-70b-hf",
                    "upstage/Llama-2-70b-instruct-v2",
                    "meta-llama/Llama-2-13b-chat-hf",
                    "meta-llama/Llama-2-7b-chat-hf"]


def get_llm_general_prompt() -> tuple[str, str, str]:
    please_help = ("Please help to plan it into lower-level step-by-step action steps " +
                   "based on the observation descriptions. " +
                   "Note that your answer should follow the format of " +
                   "'Pick up the [object1] and place it on [object2]'. " +
                   "You should plan it with step-by-step action steps. " +
                   "Now your plan of first step is: ")
    examples = ("### User:\n"
                "There are bowls with yellow, green, grey, white, and orange colors. " +
                "There are blue, white, cyan, orange, and green blocks on the table. " +
                f"The instruction is 'Please put all the blocks into bowls with matching color', {please_help}"
                "### Assistant:\n"
                "1. Pick up the white block and place it on a white bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "2. Pick up the orange block and place it on the orange bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "3. Pick up the green blocks and place it on the green bowl. " +
                "<Done>\n" +
                "### User:\n"
                "There are bowls with yellow, green, grey, white, and orange colors. " +
                "There are brown, orange, red and green blocks on the table. " +
                f"The instruction is 'Please put all the blocks into bowls of the same color', {please_help}" +
                "### Assistant:\n"
                "1. Pick up the orange blocks and place it on the orange bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "2. Pick up the green blocks and place it on the green bowl. " +
                "<Done>\n" +
                "### User:\n"
                "There are bowls with green, red, blue, grey, white, and orange colors. " +
                "There are green, red, yellow, blue, and orange blocks on the table. " +
                f"The instruction is 'Please put all the blocks into bowls with matching color', {please_help}" +
                "### Assistant:\n"
                "1. Pick up the green blocks and place it on the green bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "2. Pick up the red blocks and place it on the red bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "3. Pick up the blue blocks and place it on the blue bowl. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                "4. Pick up the orange blocks and place it on the orange bowl. " +
                "<Done>\n" +
                "### User:\n"
                "On the table, there are 3 blocks. Their colors are [green, red, yellow]. " +
                "There are 2 zones. Their colors are [white, red]. " +
                "There are 2 bowls. Their colors are [blue, gray]. " +
                f"The instruction is 'Put the red block in the white zone', {please_help}" +
                "### Assistant:\n"
                f"1. Pick up the red block and place it on the white zone. " +
                f"<Done>\n" +
                "### User:\n"
                f"On the table, there are 2 blocks. Their colors are [red, green]. " +
                f"The instruction is 'Stack all blocks', {please_help} " +
                "### Assistant:\n"
                f"1. Pick up the red block and place it on the table. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                f"2. Pick up the green block and place it on the red block. " +
                f"<Done>\n" +
                "### User:\n"
                f"On the table, there are 2 blocks. Their colors are [grey, white]. " +
                "There is a zone. The color is green. " +
                f"The instruction is 'Put all the blocks in the green zone', {please_help}" +
                "### Assistant:\n"
                f"1. Pick up the white block and place it on the green zone. " +
                "### User:\n"
                "Your plan for next step is: " +
                "### Assistant:\n"
                f"2. Pick up the gray block and+place it on the green zone. " +
                f"<Done>\n")
    system_prompt = ("You are a helpful assistant that can help the embodied agent " +
                     "make plans for a high-level human instruction and " +
                     "decompose the high-level human instruction into low-level executable instructions.")

    return please_help, examples, system_prompt


def get_task_llm_prompt_for_next_step(
        task_instruction: str,
        feedback: Dict,
        is_at_start: bool,
        message_history: Union[List[Dict[str, str]], List[None]],
        llm_name: str,
):
    # observation, example = get_observation_and_example(task)
    please_help, examples, system_prompt_content = get_llm_general_prompt()
    observation_state = feedback["observation_state"] if "observation_state" in feedback else None
    success_state = feedback["success_state"] if "success_state" in feedback else None

    if is_at_start:
        complete_task_instruction = f"The instruction is '{task_instruction}'. {please_help}"
        user_prompt_content = ("Please make plans and decompose the high-level instruction into " +
                               "lower-level step-by-step action steps. Here are some examples: " +
                               f"{examples}")
        if observation_state:
            user_prompt_content += f"Now, given the observation '{observation_state}'."
        user_prompt_content += f"{complete_task_instruction}"

        if llm_name.lower() in ["chatgpt", "gpt-3.5-turbo"]:
            system_prompt = {
                "role": "system",
                "content": system_prompt_content
            }
            user_prompt = {
                "role": "user",
                "content": user_prompt_content
            }
            prompt = [system_prompt, user_prompt]
        elif llm_name in OPEN_SOURCE_LLMs:
            prompt = ("### System:\n"
                      f"{system_prompt_content}\n\n"
                      "### User:\n"
                      f"{user_prompt_content}\n\n"
                      "### Assistant:\n")
        else:
            raise ValueError("llm_name is wrong!")

    else:
        if not message_history:
            raise ValueError("Message history should not be None!")

        user_prompt_content = ""
        if not observation_state and not success_state:  # Without VLM's feedback
            user_prompt_content += "What's your next step plan?"
        else:
            if observation_state:
                user_prompt_content = f"Current observation state is: {observation_state}. "
            if success_state:
                user_prompt_content += f"Last step action execution state is: {success_state}. "
            user_prompt_content += (
                    f"The final task instruction is {task_instruction}. " +
                    "Based on the above information, you should plan the next step instruction for the robot. " +
                    "If the last step instruction is executed unsuccessfully, " +
                    "you should repeat the last step instruction." +
                    "Note that your answer can only contain one sentence and should also follow the format of " +
                    "'Pick up the [object1] and place it on the [object2]'. " +
                    "You should clearly indicate the color of each object like " +
                    "'Pick up the yellow block and place it in the yellow bowl'." +
                    "If the task is finished, just output <Done>. " +
                    "So, your plan for next step is: "
            )

        # user_prompt_content = (
        #         f"Current observation state is: {observation_state}. " +
        #         f"Last step action execution state is: {success_state}. " +
        #         f"The final low-level task instruction is {task_instruction}. " +
        #         "Based on the above information, you should plan the next step instruction for the robot. " +
        #         "If the last step instruction is executed unsuccessfully, " +
        #         "you should repeat the last step instruction." +
        #         "Note that your answer can only contain one sentence and should also follow the format of " +
        #         "'Pick up the [object1] and place it in [object2]'. " +
        #         "You should clearly indicate the color of each object like " +
        #         "'Pick up the yellow block and place it in the yellow bowl'." +
        #         "If the task is finished, just output <Done>. " +
        #         "So, what's your next step plan?"
        # )

        if llm_name.lower() in ["chatgpt", "gpt-3.5-turbo"]:
            user_prompt = {
                "role": "user",
                "content": user_prompt_content
            }
            message_history.append(user_prompt)
        elif llm_name in OPEN_SOURCE_LLMs:
            user_prompt = ("### User:\n"
                           f"{user_prompt_content}\n\n"
                           "### Assistant:\n")
            message_history += user_prompt
        else:
            raise ValueError("llm_name is wrong!")
        prompt = message_history
        # print(f"Prompt of GPT: {prompt}")

    return prompt


def load_llm(llm_name: str):
    logging.info("-" * 10 + f"Loading the LLM {llm_name}" + "-" * 10)
    if llm_name in OPEN_SOURCE_LLMs:
        tokenizer = AutoTokenizer.from_pretrained(llm_name)
        model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            rope_scaling={"type": "dynamic", "factor": 2}  # allows handling of longer inputs
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        raise ValueError("llm_name is wrong!")

    return tokenizer, model, streamer


def parse_gpt_output(plan: Dict[str, str]) -> List[str]:
    """
    ChatGPT's output example:
    {
      "content": "1. Put the blue blocks in a blue bowl.\n2. Put the orange blocks in an orange bowl.\n3. Put the purple blocks in a purple bowl.",
      "role": "assistant"
    }
    """
    plans = plan["content"].split("\n")

    # remove the prefix of each step and suffix
    new_plans = []
    for p in plans:
        new_plan = re.sub(r'[0-9]+\.', '', p).lstrip()
        new_plan = new_plan.split("<Done>")[0].strip()
        new_plans.append(new_plan)

    return new_plans


def parse_opensource_llm_output(llm_output: str) -> List[str]:
    """Remove the prompts from llm output

    LLM output example:
    ### System:
    {System}

    ### User:
    {User}

    ### Assistant:
    1. Pick up the white block and place it on the white bowl.
    2. Pick up the orange block and place it on the orange bowl.
    3. Pick up the pink block and place it on the pink bowl.
    4. Pick up the brown block and place it on the brown bowl.
    5. Pick up the cyan block and place it on the cyan bowl.
    6. Pick up the gray block and place it on the gray bowl.
    7. Pick up the blue block and place it on the blue bowl.
    8. Pick up the red block and place it on the red bowl.
    9. Pick up the green block and place it on the green bowl.
    10. Pick up the purple block and place it on the purple bowl.
    11. Pick up the yellow block and place it on the yellow bowl. <Done>
    """
    llm_response = llm_output.split("### Assistant:\n")[-1]
    plans = llm_response.split("\n")
    # remove the prefix of each step and suffix
    new_plans = []
    for p in plans:
        new_plan = re.sub(r'[0-9]+\.', '', p).lstrip()
        new_plan = new_plan.split("<Done>")[0].strip()
        new_plans.append(new_plan)

    return new_plans


def get_next_step_gpt_planner_config(
        task_instruction: str,
        feedback: Dict,
        is_at_start: bool,
        message_history: List[Dict[str, str]],
        model: str
):
    task_prompt = get_task_llm_prompt_for_next_step(
        task_instruction,
        feedback,
        is_at_start,
        message_history,
        model
    )

    if model in ["chatgpt", "gpt-3.5-turbo"]:
        engine = "gpt-3.5-turbo"
    else:
        raise ValueError("Model name should be chatgpt or gpt-3.5-turbo!")

    planner_cfg = {
        'prompt_text': task_prompt,
        'engine': engine,
        'max_tokens': 256,
        'temperature': 0,
        'query_prefix': '# ',
        'query_suffix': '.',
        'stop': ['#', 'objects = ['],
        'maintain_session': True,
        'debug_mode': False,
        'include_context': True,
        'has_return': False,
        'return_val_name': 'ret_val',
    }
    return planner_cfg


def get_next_step_plan_from_gpt_planner(
        task_name: str,
        planner_cfg: Dict[str, Any],
        planner_name: str = "chatgpt"
) -> List[str]:
    while True:
        try:
            # Use ChatGPT
            plan = openai.ChatCompletion.create(
                messages=planner_cfg['prompt_text'],
                stop=list(planner_cfg['stop']),
                temperature=planner_cfg['temperature'],
                model=planner_cfg['engine'],
                max_tokens=planner_cfg['max_tokens']
            ).choices[0].message
            break
        except (RateLimitError, APIConnectionError, APIError) as e:
            print(f'OpenAI API got err {e}')
            print('Retrying after 10s.')
            sleep(10)
        except InvalidRequestError as e:
            print(f'OpenAI API got err {e}')
            print('Skip this test case.')
            plan = {"role": "system", "content": "<Done>."}
            break

    plan = parse_gpt_output(plan)
    return plan


def get_next_step_plan_from_llm(
        task_instruction: str,
        feedback: Dict,
        is_at_start: bool,
        message_history: Union[List[Dict[str, str]], List[None]],
        llm_name: str,
        return_prompt: bool = False,
        llm_args: Dict = None,
) -> tuple[
    Union[str, List[str]],
    Union[List[Dict[str, str]], str, None]
]:
    if llm_name.lower() in ["chatgpt", "gpt-3.5-turbo"]:
        gpt_cfg = get_next_step_gpt_planner_config(
            task_instruction,
            feedback,
            is_at_start,
            message_history,
            llm_name
        )
        next_plans = get_next_step_plan_from_gpt_planner(task_instruction, gpt_cfg)
        next_plan = next_plans[0]
        prompt = gpt_cfg["prompt_text"]
    elif llm_name in OPEN_SOURCE_LLMs:
        tokenizer, model, streamer = llm_args["tokenizer"], llm_args["model"], llm_args["streamer"]

        prompt = get_task_llm_prompt_for_next_step(
            task_instruction,
            feedback,
            is_at_start,
            message_history,
            llm_name
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # del inputs["token_type_ids"]

        output = model.generate(
            **inputs,
            streamer=streamer,
            use_cache=True,
            max_new_tokens=float('inf')
        )
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Original decoded output is: ", decoded_output)
        next_plans = parse_opensource_llm_output(decoded_output)
        print("After parsing: ", next_plans)
        next_plan = next_plans[0]
    else:
        raise ValueError("llm_name is wrong!")

    if return_prompt:
        return next_plan, prompt
    else:
        return next_plan


def update_message_history_with_next_plan(message_history, next_plan, llm_name):
    if llm_name.lower() in ["chatgpt", "gpt-3.5-turbo"]:
        message_history.append({"role": "system", "content": next_plan})
    elif llm_name in OPEN_SOURCE_LLMs:
        message_history += f"{next_plan}\n\n"
    else:
        raise ValueError("llm_name is wrong!")

    return message_history
