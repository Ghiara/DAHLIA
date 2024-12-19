# DAHLIA
Official implementation of paper "Data-Agnostic Robotic Long-Horizon Manipulation with Language-Conditioned Closed-Loop Feedback"



## Installation

### Cloning
Go to the GitHub repository website and select 'Code' to get an HTTPS or SSH link to the repository.
Clone the repository to your device, e.g.
```bash
git clone https://github.com/Ghiara/DAHLIA
```
Enter the root directory of this project on your device. The root directory contains this README-file.

### Environment

We recommend to manage the python environment with **conda** and suggest [Miniconda](https://docs.conda.io/en/latest/miniconda.html) as a light-weight installation.

> You may also use different environment tools such as python's *venv*. Please refer
to *requirements.txt* in this case. In the following, we will proceed with conda.

Install the environment using the ``conda`` command:
```bash
conda create -n dahlia python=3.9
```
This might take some time because it needs to download and install all required packages.

Activate the new environment by running (Make sure that no other environment was active before.):
```bash
conda activate dahlia
```

### Install Other Packages
**NOTE:** DAHLIA principly does not need training. Install Pytorch in case you want to 
fine-tune the LLM or use CLIPort. 
```bash
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.9.5
python setup.py develop
```

----------------------------------------------------------------------------

## Run code

### Generate Task File
Generate more new tasks if needed.

**reference number** determines the number of existing task candidates being
prompted to LLM.

For top-down generation, fill in **task name** and **task instruction** (optional)
to specify the desired task.

```bash
# bottom-up template generation
python gensim/run_simulation.py \ 
    prompt_folder=bottomup_task_generation_prompt \ 
    save_memory=True load_memory=True \ 
    task_description_candidate_num=[reference number] \ 
    use_template=True

# top-down task generation
python gensim/run_simulation.py \
    prompt_folder=topdown_task_generation_prompt \ 
    save_memory=True load_memory=True \
    task_description_candidate_num=[reference number] \ 
    use_template=True target_task_name=[task name] \
    target_task_description=[task instruction]

# task-conditioned chain-of-thought generation
python gensim/run_simulation.py \
    prompt_folder=topdown_chain_of_thought_prompt \ 
    save_memory=True load_memory=True \ 
    task_description_candidate_num=[reference number] \ 
    use_template=True target_task_name=[task name] \ 
    target_task_description=[task instruction] 
```
### Execute a Task
To directly try to complete a task (based on existing task file).

Fill in **task name** to locate the task file.

```bash
python cliport/cap.py task=[task name] mode=cap check=False
```

### Generate Test Task Dataset
DAHLIA uses the same dataset style of CLIPort.

Fill in **number of samples** to set number of episodes one task dataset should have.

```bash
python cliport/demo.py n=[number of samples] \
    task=[task name] mode=test all_result=False\
```
This will save the test dataset in the folder data.

### Test the Execution
Randomly pick *n* episodes form test dataset and execute and evaluate, 
finally show success rate.

```bash
python cliport/cap.py task=[task name] mode=test check=False n=1
```

----------------------------------------------------------------------------

## Acknowledgements

This project uses code or idea from open-source projects and datasets including:

#### GenSim

Origin: [https://github.com/liruiw/GenSim](https://github.com/liruiw/GenSim)  
License: [MIT](https://github.com/liruiw/GenSim/blob/main/LICENSE) 

#### LoHoRavens

Origin: [https://github.com/Shengqiang-Zhang/lohoravens](https://github.com/Shengqiang-Zhang/lohoravens)  
License: [Apache 2.0](https://github.com/Shengqiang-Zhang/lohoravens/blob/main/LICENSE)

#### Code as Policies

Origin: [https://github.com/google-research/google-research/tree/master/code_as_policies](https://github.com/google-research/google-research/tree/master/code_as_policies)

#### CLIPort-batchify

Origin: [https://github.com/ChenWu98/cliport-batchify](https://github.com/ChenWu98/cliport-batchify)

#### Google Ravens (TransporterNets)

Origin:  [https://github.com/google-research/ravens](https://github.com/google-research/ravens)  
License: [Apache 2.0](https://github.com/google-research/ravens/blob/master/LICENSE)    

#### OpenAI CLIP

Origin: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)  
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)  

#### Google Scanned Objects

Origin: [Dataset](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects)  
License: [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/)  