# DAHLIA

Official implementation of paper "Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback"

### [[Project Website]](https://ghiara.github.io/DAHLIA/) 

[Yuan Meng](https://github.com/Ghiara)<sup>1,</sup>, 
[Xiangtong Yao](https://www.ce.cit.tum.de/air/people/xiangtong-yao/)<sup>1</sup>, 
[Haihui Ye]()<sup>1</sup>,
[Yirui Zhou]()<sup>1</sup>,
[Shengqiang Zhang]()<sup>2</sup>,
[Achim Lilienthal](https://kifabrik.mirmi.tum.de/team/)<sup>1</sup>,
[Zhenshan Bing](https://github.com/zhenshan-bing)<sup>3,4</sup>, 
[Alois Knoll](https://www.ce.cit.tum.de/air/people/prof-dr-ing-habil-alois-knoll/)<sup>1</sup>,

<sup>1</sup>The School of Computation, Information and Technology, Technical University of Munich, Germany

<sup>2</sup>Center for Information and Language Processing, Ludwig Maximilian University of Munich, Germany

<sup>3</sup>State Key Laboratory for Novel Software Technology, Nanjing University, China

<small><sup>4</sup>Corresponding author: zhenshan.bing@tum.de</small>








## 1. Installation

### 1.1 Cloning
Go to the GitHub repository website and select 'Code' to get an HTTPS or SSH link to the repository.
Clone the repository to your device, e.g.
```bash
git clone https://github.com/Ghiara/DAHLIA
```
Enter the root directory of this project on your device. The root directory contains this README-file.

### 1.2 Build Environment

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

### 1.3 Install Other Packages (option)
**NOTE:** DAHLIA principly does not need training. Install Pytorch in case you want to 
fine-tune the LLM or use CLIPort. 
```bash
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install pytorch-lightning==1.9.5
python setup.py develop
```

----------------------------------------------------------------------------

## 2. Run code

### 2.1 Generate Task File (option)
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
### 2.2 Execute a Task
To directly try to complete a task (based on existing task file).

Fill in **task name** to locate the task file.

```bash
python cliport/cap.py task=[task name] mode=cap check=False
```

### 2.3 Generate Test Task Dataset
DAHLIA uses the same dataset style of CLIPort.

Fill in **number of samples** to set number of episodes one task dataset should have.

```bash
python cliport/demo.py n=[number of samples] \
    task=[task name] mode=test all_result=False\
```
This will save the test dataset in the folder data.

### 2.4 Test the Execution
Randomly pick *n* episodes form test dataset and execute and evaluate, 
finally show success rate.

```bash
python cliport/dahlia_run.py task=[task name] mode=test check=False n=1
```



## 3. Prompting

The prompting of Task generation can visit at `/prompts/bottomup_task_generation_prompt_new/*` or `/prompts/topdown_task_generation_prompt/*`.

The prompting of DAHLIA role definition can visit at `/prompts/dahlia/*`

As described in the paper, the prompting of each LMP planner contains basically following parts:

1. The libraries import, including our predefined APIs and widely used third party libraries(e.g., numpy), for example:
```python

import numpy as np
from env_utils import get_obj_pos, get_obj_rot, parse_position
from utils import get_obj_positions_np, get_obj_rotations_np
from cliport.utils import utils
```

2. Methods explanation, which briefly introduce how our customized APIs can be used, for example:
```python

# ---------------------------------------------------------------------------
# Existing Method Explanations
# ---------------------------------------------------------------------------
'''
get_obj_pos(obj) -> [list] # return a list of len(obj) of 3d position-vectors of obj, even when obj is just one object not a list of objects
get_obj_rot(obj) -> [list] # return a list of len(obj) of 4d quaternion orientation-vectors of obj, even when obj is just one object not a list of objects
get_obj_positions_np([obj]) -> [list] # return a list of len([obj]) of 3d position-vectors of obj in [obj]
...etc.
'''

```

3. Third part define how our Coordinate system defined related to the robot view, for example:
```python

# ---------------------------------------------------------------------------
# Orientations in Coordinate System
# ---------------------------------------------------------------------------
'''
left: y-
right: y+
front: x+
rear: x-
top: z+
bottom: z-
top left: x-y-
bottom left: x+y-
top right: x-y+
bottom right: x+y+
'''
```

4. This part define the general role how the LLM agent can act and response with each other, for example:
```python

# ---------------------------------------------------------------------------
# General Requirements
# ---------------------------------------------------------------------------
'''
You are writing python code for object parsing, refer to the code style in examples below.
You can use the existing APIs above, you must NOT import other packages.
Our coordinate system is 3D cartesian system, but still pay attention to the orientations. 
Also pay attention to the return format requirements in descriptions for some tasks.
When you are not sure about positions, you had better use parse_position(), and clarify your return format demand.
'''
```

5. We may introduce task plan examples to help agent adapt the task planning in a few-shot manner, the code for example:
```python

# ---------------------------------------------------------------------------
# Task Examples
# ---------------------------------------------------------------------------

objects = ['blue block', 'cyan block', 'purple bowl', 'gray bowl', 'brown bowl', 'pink block', 'purple block']
# the block closest to the purple bowl.
block_names = ['blue block', 'cyan block', 'purple block']
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=get_obj_pos('purple bowl')[0])
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name

objects = ['brown bowl', 'banana with obj_id 1', 'brown block with obj_id 9', 'apple', 'blue bowl with obj_id 8', 'blue block with obj_id 3']
# the block, return result as list.
ret_val = ['brown block with obj_id 9', 'blue block with obj_id 3']

objects = ['brown bowl', 'banana with obj_id 1', 'brown block with obj_id 9', 'apple', 'blue bowl with obj_id 8', 'blue block with obj_id 3']
# the block color, return result as tuple.
ret_val = ('brown', 'blue')
...
```

Based on above mention promptings, we can help the agent to build a systematical planning mechanism based on the idea of chain-of-thought.

----------------------------------------------------------------------------

## 4. Acknowledgements

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