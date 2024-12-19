import csv
import os

import hydra
import openai

from gensim.memory import Memory
from gensim.topdown_sim_runner import TopDownSimulationRunner
from gensim.utils import (
    save_text)
from gensim.utils import set_gpt_model


@hydra.main(config_path='../cliport/cfg', config_name='data', version_base="1.2")
def main(cfg):
    openai.api_key = cfg['openai_key']

    cfg['model_output_dir'] = os.path.join(cfg['output_folder'], cfg['model_output_dir'])
    if 'seed' in cfg:
        cfg['model_output_dir'] = cfg['model_output_dir'] + f"_{cfg['seed']}"

    eval_model_output_dir = "eval_" + cfg['model_output_dir']
    set_gpt_model(cfg['gpt_model'])
    memory = Memory(cfg)
    simulation_runner = TopDownSimulationRunner(cfg, memory)

    # take in a folder and output a csv
    results = {}
    folders = sorted(os.listdir(cfg['model_output_dir']))
    for file in folders:
        file = os.path.join(cfg['model_output_dir'], file)
        res = open(file).read()

        simulation_runner.task_creation(res, offline_mode=True)
        simulation_runner.simulate_task()
        simulation_runner.print_current_stats()
        try:
            task_name = simulation_runner.curr_task_name
            save_text(eval_model_output_dir,
                      task_name + "_code_output", simulation_runner.generated_code)
            cnt = (simulation_runner.curr_trials)
            results['task_name'] = [simulation_runner.syntax_pass_rate / cnt,
                                    simulation_runner.runtime_pass_rate / cnt,
                                    simulation_runner.env_pass_rate / cnt]

        except Exception as e:
            print(e)
            print("save stats failed")

    print(results)

    with open(os.path.join(eval_model_output_dir, "topdown_eval_results.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['task name', 'syntax', 'runtime', 'env success'])
        for k, (s, v, t) in results.items():
            writer.writerow([k, s, v, t])

    # simulation_runner.save_stats()


# load few shot prompts


if __name__ == "__main__":
    main()
