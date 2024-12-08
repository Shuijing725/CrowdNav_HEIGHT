# Height_CrowdNav

This repository contains the codes for our paper titled "HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments".   

[[Website]](https://sites.google.com/view/crowdnav-height/home) [[arXiv]](https://arxiv.org/abs/2307.06924) [[Videos]](https://www.youtube.com/playlist?list=PLL4IPhbfiY3YkITpyLjeroak_wBn151pn)  

------

**[News]**
- Please check out my curated paper list for robot social navigation [here](https://github.com/Shuijing725/awesome-robot-social-navigation) (It is under active development) 

## Abstract
We study the problem of safe and intention-aware robot navigation in dense and interactive crowds. 
Most previous reinforcement learning (RL) based methods fail to consider different types of interactions among all agents or ignore the intentions of people, which results in performance degradation. 
In this paper, we propose a novel recurrent graph neural network with attention mechanisms to capture heterogeneous interactions among agents through space and time. 
To encourage longsighted robot behaviors, we infer the intentions of dynamic agents by predicting their future trajectories for several timesteps. 
The predictions are incorporated into a model-free RL framework to prevent the robot from intruding into the intended paths of other agents. 
We demonstrate that our method enables the robot to achieve good navigation performance and non-invasiveness in challenging crowd navigation scenarios. We successfully transfer the policy learned in simulation to a real-world TurtleBot 2i.

<p align="center">
<img src="/figures/open.png" width="450" />
</p>

## Setup
1. In a conda environment or virtual environment with Python 3.x, install the required python package
```
pip install -r requirements.txt
```

2. Install Pytorch 1.12.1 and torchvision following the instructions [here](https://pytorch.org/get-started/previous-versions/#v1121)

3. Install [OpenAI Baselines](https://github.com/openai/baselines#installation) 
```
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library


## Overview
This repository is organized in five parts: 
- `crowd_nav/` folder contains configurations and policies used in the simulator.
- `crowd_sim/` folder contains the simulation environment.
- `training/` contains the code for the RL policy networks and ppo algorithm. 
- `trained_models/` contains some pretrained models provided by us. 

## Run the code
### Training
- Modify the configurations in `crowd_nav/configs/config.py`. Especially,
  - Gym environment: 
    - Set `env.env_name = 'CrowdSim3DTbObsHie-v0'` for A^*^+CNN baseline. Set `env.env_name = 'CrowdSim3DTbObs-v0'` for all other methods.
    - Different environment layouts:
      - For random environment in pure simulation, 
        - Set `env.scenario = 'circle_crossing'`
        - Set `env.mode = 'sim'`
      - For sim2real environments (Hallway, Lounge),
        - Set `env.scenario = 'csl_workspace'`
        - Set `env.csl_workspace_type = 'lounge'` or `'hallway'`
        - Set `env.mode = 'sim2real'`
    - Number of humans: 
      - Dynamic humans: change `sim.human_num` and `sim.human_num_range`
      - Static humans: change `sim.static_human_num`, `sim.static_human_range`
    - Number of obstacles:
      - Set `sim.static_obs = True`
      - Change `sim.static_obs_num` and `sim.static_obs_num_range`
  - Robot policy: set `robot.policy` to
    - `selfAttn_merge_srnn_lidar` for HEIGHT (ours) and its ablations
      - No attn: `SRNN.use_hr_attn = False`, `SRNN.use_self_attn = False`
      - RH: `SRNN.use_hr_attn = True`, `SRNN.use_self_attn = False`
      - HH: `SRNN.use_hr_attn = False`, `SRNN.use_self_attn = True`
      - RH + HH (ours): `SRNN.use_hr_attn = True`, `SRNN.use_self_attn = True`
    - `lidar_gru` for A^*^+CNN
    - `dsrnn_obs_vertex` for DS-RNN
    - `homo_transformer_obs` for HomoGAT
  - Logging and saving:
    - All logs and checkpoints will be saved in `training.output_dir`
    - To resume training from a previous checkpoint,
      - Set `training.resume = 'rl'`
      - Set `training.load_path` to the path of the previous checkpoint

- After you change the configurations, run
  ```
  python train.py 
  ```

### Testing 
#### In simulation
- Please modify the test arguments in line 20-33 of `test.py` (**Don't set the argument values in terminal!**)
  - Set `--model_dir` to the path of the saved folder from training
  - Robot policy:
    - To test RL-based methods (all methods in the paper except DWA),
      - Set `--dwa` to False, `--cpu` to False, `--test_model` to the name of the checkpoint to be tested
    - To test DWA,
      - Set `--dwa` to True, `--cpu` to True (DWA also needs a dummy `--model_dir`! The directory can be anything with a `configs/` folder.)
  - To save the gif and pictures of each episode, set `--save_slides` to True (Note: Saving the visuals will significantly slow down testing)
- Run   
  ```
  python test.py 
  ```
- To test policies in OOD environments, run `test_less_human.py`/`test_less_obs.py`/`test_more_human.py`/`test_more_obs.py`  
Note that the `config.py` in the `--model_dir` folder will be loaded, instead of those in the root directory.  
The testing results are logged in `trained_models/your_output_dir/test/` folder, and are also printed on terminal.  
If you set `--save_slides` to True in `test.py`, you will be able to see visualizations like this:  
<img src="/figures/output1.gif" width="420" /> <img src="/figures/output2.gif" width="420" />

#### In real-world
- In the folder of a trained checkpoint, in `config.py`,
  - Set `env.env_name = 'rosTurtlebot2iEnv-v0'` 
  - Change `sim2real` configurations if needed
- Set up the sensors, perception modules, and the robot following our sim2real tutorial [here](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)
- Run   
  ```
  python test.py 
  ```
- From terminal, input the robot goal position and press enter, the robot will move if everything is up and running

#### Test pre-trained models provided by us
| Method                    | `--model_dir` in test.py                             | `--test_model` in test.py |
|---------------------------|------------------------------------------------------|---------------------------|
| Ours, Random environment  | `trained_models/ours_HH_RH_randEnv`                  | `237800.pt`               |
| Ours, Lounge environment  | `trained_models/ours_RH_HH_loungeEnv_resumeFromRand` | `137400.pt`               |
| Ours, Hallway environment | `trained_models/ours_RH_HH_hallwayEnv`               | `208200.pt`               |


### Plot the training curves
```
python plot.py
```
Here are example learning curves of our proposed method.

<img src="/figures/rewards.png" width="370" /> <img src="/figures/losses.png" width="370" />

## Sim2Real
We are happy to announce that our sim2real tutorial and code are released [here](https://github.com/Shuijing725/CrowdNav_Sim2Real_Turtlebot)!  
**Note:** This repo only serves as a reference point for the sim2real transfer of crowd navigation. Since there are lots of uncertainties in real-world experiments that may affect performance, we cannot guarantee that it is reproducible on all cases. 

## Disclaimer
1. We only tested our code in Ubuntu 20.04 with Python 3.8. The code may work on other OS or other versions of Python, but we do not have any guarantee.  

2. The performance of our code can vary depending on the choice of hyperparameters and random seeds (see [this reddit post](https://www.reddit.com/r/MachineLearning/comments/rkewa3/d_what_are_your_machine_learning_superstitions/)). 
Unfortunately, we do not have time or resources for a thorough hyperparameter search. Thus, if your results are slightly worse than what is claimed in the paper, it is normal. 
To achieve the best performance, we recommend some manual hyperparameter tuning.

## Citation
If you find the code or the paper useful for your research, please cite the following papers:
```
@inproceedings{liu2022intention,
  title={Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph},
  author={Liu, Shuijing and Chang, Peixin and Huang, Zhe and Chakraborty, Neeloy and Hong, Kaiwen and Liang, Weihang and Livingston McPherson, D. and Geng, Junyi and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023},
  pages={12015-12021}
}

@inproceedings{liu2020decentralized,
  title={Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning},
  author={Liu, Shuijing and Chang, Peixin and Liang, Weihang and Chakraborty, Neeloy and Driggs-Campbell, Katherine},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
  pages={3517-3524}
}
```

## Credits
Other contributors:  

[Haochen Xia](https://www.linkedin.com/in/haochen-xia-614bb0251/)

Part of the code is based on the following repositories:  

[1] S. Liu, P. Chang, Z. Huang, N. Chakraborty, K. Hong, W. Liang, D. L. McPherson, J. Geng, and K. Driggs-Campbell, “Intention aware robot crowd navigation with attention-based interaction graph,” in ICRA 2023. (Github: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph)

[2] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, "Decentralized Structural-RNN for Robot Crowd Navigation with Deep Reinforcement Learning," in ICRA 2019. (Github: https://github.com/Shuijing725/CrowdNav_DSRNN)  

[3] Z. Huang, H. Chen, J. Pohovey, and K. Driggs-Campbell. "Neural Informed RRT*: Learning-based Path Planning with Point Cloud State Representations under Admissible Ellipsoidal Constraints," in ICRA 2024. (Github: https://github.com/tedhuang96/nirrt_star)  

## Contact
If you have any questions or find any bugs, please feel free to open an issue or pull request.