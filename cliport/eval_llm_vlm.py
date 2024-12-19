"""Ravens main training script."""

import json
import logging
import os

import hydra
import numpy as np
import openai
import torch
from matplotlib import pyplot as plt

from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.environments.environment import Environment
from cliport.utils import utils
from llm import get_next_step_plan_from_llm, update_message_history_with_next_plan, load_llm
from vlm import load_vlm, get_description_from_vlm

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


@hydra.main(config_path='./cfg', config_name='eval_reporter')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(
            vcfg['data_dir'],
            tcfg,
            group=eval_task,
            mode=mode,
            n_demos=vcfg['n_demos'],
            augment=False
        )
    else:
        ds = dataset.RavensDataset(
            os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
            tcfg,
            n_demos=vcfg['n_demos'],
            augment=False
        )

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg[
        'model_path'] else f"results-{mode}-openflamingo.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # LLM
    use_llm = vcfg["use_llm"]
    gpt_name_list = ["gpt-3.5-turbo", "chatgpt"]
    open_source_llm_name_list = ["meta-llama/Llama-2-70b-hf",
                                 "upstage/Llama-2-70b-instruct-v2",
                                 "meta-llama/Llama-2-13b-chat-hf",
                                 "meta-llama/Llama-2-7b-chat-hf"]
    llm_name = open_source_llm_name_list[3]
    OPENAI_API_KEY = "your_openai_key"
    openai.api_key = OPENAI_API_KEY

    # Load LLM
    if use_llm:
        if llm_name.lower() not in gpt_name_list:
            tokenizer, model, streamer = load_llm(llm_name)
            llm_args = {
                "tokenizer": tokenizer,
                "model": model,
                "streamer": streamer
            }
        else:
            llm_args = {}

    # VLM
    use_vlm = vcfg["use_vlm"]

    # Load VLM
    if use_vlm:
        vlm, vis_processors, tokenizer = load_vlm("OpenFlamingo-9B-vitl-mpt7b", device)
        vlm_args = {
            "tokenizer": tokenizer,
            "model": vlm,
            "vis_processors": vis_processors
        }
        # example_image_path = f"{save_path}/example_images/"
        example_image_path = ("/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/"
                              "pick-and-place-primitive-cliport-n20000-train/downstream_eval/example_images/")
    else:
        vlm_args = {}

    # Evaluation loop
    logging.info(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):
            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            logging.info(f"Loaded: {model_file}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']

            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos):
                logging.info(f'Test: {i + 1}/{n_demos}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)

                # set task
                if 'multi' in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f'Evaluating on {task_name}')
                else:
                    task_name = vcfg['eval_task']
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(
                    task)  # SQ: I think this should be a key of interaction between env and task
                obs = env.reset()
                info = env.info  # SQ: Final language goal and env info with object poses, rotation, and colors
                reward = 0

                scene_description = env.task.scene_description
                print("Scene description:", scene_description)

                # Start recording video (NOTE: super slow)
                if record:
                    logging.info("Start recording video ......")
                    video_name = f'{task_name}-{i + 1:06d}'
                    if 'multi' in vcfg['model_task']:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                final_goal = info["lang_goal"]
                print(f'THE FINAL GOAL: {final_goal}')
                done = False
                if use_llm:
                    feedback = {"observation_state": scene_description}
                    next_plan, messages = get_next_step_plan_from_llm(
                        task_instruction=final_goal,
                        feedback=feedback,
                        is_at_start=True,
                        message_history=[],
                        llm_name=llm_name,
                        return_prompt=True,
                        llm_args=llm_args
                    )

                n_steps = 0
                raw_images = []

                if use_vlm:
                    # Save an image before execution.
                    color = env.render()
                    save_name = f"{save_path}/{task_name}_{n_steps}.png"
                    plt.imsave(save_name, color)
                    raw_images.append(save_name)

                while not done and n_steps < env.task.max_steps:
                    if use_llm:
                        messages = update_message_history_with_next_plan(messages, next_plan,
                                                                         llm_name)
                        logging.info(f"Next_plan is: {next_plan}")
                        print("This is the end of the next plan output.")
                        info['lang_goal'] = next_plan
                    act = agent.act(obs, info,
                                    goal)  # SQ: goal is not used in the implementation of act.
                    lang_goal = info['lang_goal']
                    obs, reward, done, info = env.step(act)
                    n_steps += 1
                    if use_vlm:
                        color = env.render()
                        save_name = f"{save_path}/{task_name}_{n_steps}.png"
                        plt.imsave(save_name, color)
                        raw_images.append(save_name)

                        observation_description = get_description_from_vlm(
                            vlm_args=vlm_args,
                            example_img_directory=f"{example_image_path}/obs/",
                            test_imgs=raw_images[-1:],
                            is_success_description=False,
                            device=device
                        )
                        logging.info(f"observation_description: {observation_description}")

                        success_description = get_description_from_vlm(
                            vlm_args=vlm_args,
                            example_img_directory=f"{example_image_path}/success/",
                            test_imgs=raw_images[-2:],
                            is_success_description=True,
                            last_instruction=lang_goal,
                            device=device
                        )
                        logging.info(f"success_description: {success_description}")
                        feedback = {
                            "observation_description": observation_description,
                            "success_description": success_description
                        }
                    else:
                        feedback = {}

                    if use_llm:
                        next_plan = get_next_step_plan_from_llm(
                            task_instruction=final_goal,
                            feedback=feedback,
                            is_at_start=False,
                            message_history=messages,
                            llm_name=llm_name,
                            return_prompt=False,
                            llm_args=llm_args
                        )

                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                    if done:
                        break

                results.append((total_reward, info))
                mean_reward = np.mean([r for r, i in results])
                print(f'Mean: {mean_reward} | Task: {task_name} | Ckpt: {ckpt}')

                # End recording video
                if record:
                    env.end_rec()

            all_results[ckpt] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }

        # Save results in a json file.
        if vcfg['save_results']:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


if __name__ == '__main__':
    main()
