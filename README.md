# DAHLIA

<div align="center">
Official implementation of paper "Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback"

### [[Project Website]](https://ghiara.github.io/DAHLIA/) 

[Yuan Meng](https://github.com/Ghiara)<sup>1,</sup>, [Xiangtong Yao](https://www.ce.cit.tum.de/air/people/xiangtong-yao/)<sup>1</sup>, [Haihui Ye]()<sup>1</sup>, [Yirui Zhou]()<sup>1</sup>, [Shengqiang Zhang]()<sup>2</sup>,

[Zhenshan Bing](https://github.com/zhenshan-bing)<sup>3,&dagger;</sup>, [Alois Knoll](https://www.ce.cit.tum.de/air/people/prof-dr-ing-habil-alois-knoll/)<sup>1</sup>,
</div>

<p align="center">
<small><sup>1</sup>The School of Computation, Information and Technology, Technical University of Munich, Germany</small>
<br><small><sup>2</sup>Center for Information and Language Processing, Ludwig Maximilian University of Munich, Germany</small>
<br><small><sup>3</sup>State Key Laboratory for Novel Software Technology, Nanjing University, China</small>
<br><small><sup>&dagger;</sup>Corresponding author: zhenshan.bing@tum.de</small>
</p>


## Abstract

Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning 
to enable robots to execute tasks from human commands. However, these approaches often suffer from limited generalization, 
adaptability, and the scarcity of large-scale specialized datasets—unlike data-rich fields such as computer vision—leading 
to challenges in handling complex long-horizon tasks.
In this work, We introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation 
that leverages large language models (LLMs) for real-time task planning and execution.
Our framework features a dual-tunnel architecture, where a planner LLM decomposes tasks and generates executable plans, 
while a reporter LLM provides closed-loop feedback, ensuring adaptive re-planning and robust execution.
Additionally, we incorporate temporal abstraction and chain-of-thought (CoT) reasoning to enhance inference efficiency 
and traceability. DAHLIA achieves superior generalization and adaptability across diverse, unstructured environments, 
demonstrating state-of-the-art performance in both simulated and real-world long-horizon tasks.

![framework](/docs/static/images/framework.jpg "Framework of DAHLIA")



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
python cliport/dahlia_run.py task=[task name] mode=dahlia check=False
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

### 2.5 Dahlia task names and task goal Instructions
<div align="center">


| Task ID | Task Name                                           | Task Goal                                                                                                                                     |
|--------:|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| A.      | stack-all-blocks-on-a-zone                          | "Stack all blocks in the [COLOR] zone."                                                                                                       |
| B.      | stack-blocks-of-same-size                          | "Stack blocks of the same size in the [COLOR1] zone and [COLOR2] zone respectively."                                                         |
| C.      | stack-blocks-of-same-color                         | "Stack all the blocks of the same color together in the same colored zone."                                                                  |
| D.      | stack-blocks-by-color-and-size                     | "Stack only the [SIZE] blocks of [COLOR_TYPE] color in the [COLOR] zone."                                                                    |
| E.      | stack-blocks-by-relative-position-and-color        | "Stack all the blocks, which are to the [REL_POS] of the [COLOR1] block with [POS_TYPE] distance larger than 0.05 unit, in the [COLOR2] zone." |
| F.      | move-blocks-between-absolute-positions             | "Move all the blocks in the [POS1] area to [POS2] area."                                                                                      |
| G.      | move-blocks-between-absolute-positions-by-size     | "Move all the [SIZE] blocks in the [POS1] area to [POS2] area."                                                                              |
| H.      | put-block-into-matching-bowl                       | "Put the blocks in the bowls with matching colors."                                                                                           |
| I.      | put-block-into-mismatching-bowl                    | "Put the blocks in the bowls with mismatching colors."                                                                                        |
| J.      | stack-blocks-with-alternate-color                  | "Stack blocks with alternate colors on the [COLOR1] zone, starting with the [COLOR2] color."                                                  |
| G1      | build-rectangular-pyramid                 | "Construct a 9-4-1 rectangular pyramid structure in the zone using 14 blocks of the same color."                                                                                                          |
| G2      | build-cube                                | "Construct a 2*2*2 cube structure in the zone using 8 blocks of the same color."                                                                                                                          |
| G3      | construct-circle-with-blocks              | "Construct a circle with suitable radius with alternating [COLOR1] and [COLOR2] blocks in the zone."                                                                                                     |
| G4      | construct-circle-ball-middle              | "Construct a circle with suitable radius with alternating [COLOR1] and [COLOR2] blocks around the ball."                                                                                                  |
| G5      | build-concentric-circles                  | "Construct two concentric circles in the zone using [NUM] [COLOR1] and [NUM + 4] [COLOR2] blocks."                                                                                                        |
| G6      | divide-blocks                             | "Divide the blocks into groups of [NUM] and stack each group (also including the group with block number less than [NUM]) in a different zone."                                                          |
| G7      | max-odd-number-blocks-in-same-color-zone  | "Place the maximal odd number of blocks of the same color in each correspondingly colored zone."                                                                                                          |
| G8      | stack-most-color-block                    | "Stack blocks of the same color that has the largest quantity in the zone."                                                                                                                               |
| G9      | zone-bisector                             | "Arrange all blocks on the zone bisector line between two symmetrically placed zones evenly on the tabletop, and the gap between two adjacent blocks' edges should be near the block size, and the line connecting the center of the zones also bisects these blocks." |
| G10     | insert-blocks-in-fixture                  | "Each L-shaped fixture can hold three blocks, suppose the block size is (a,a,a), then in fixture's local coordinate system, the three places that can hold blocks are [(0,0,0),(a,0,0),(0,a,0)]. Fill in all the fixtures which have random position and rotation with blocks, and make sure in the end in every fixture there are three blocks with different colors." |

</div>


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

## 4. Citation

if you use this work, please cite:

```bibtex
    @misc{meng2025dataagnosticroboticlonghorizonmanipulation,
        title={Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback}, 
        author={Yuan Meng and Xiangtong Yao and Haihui Ye and Yirui Zhou and Shengqiang Zhang and Zhenshan Bing and Alois Knoll},
        year={2025},
        eprint={2503.21969},
        archivePrefix={arXiv},
        primaryClass={cs.RO},
        url={https://arxiv.org/abs/2503.21969}, 
    }
```


## 5. Acknowledgements

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

