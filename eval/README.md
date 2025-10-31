This repository contains the code for evaluating trained SAIR agents in the Distracting Safety Gymnasium environment corresponding with the paper "Safety-Aware Invariant Representations for Reinforcement Learning from Pixels". This code is based on the Deep Bisim4Control repository: https://github.com/facebookresearch/deep_bisim4control.

We provide a Dockerfile for running the evaluation in a Docker container. After startin the container, to run the evaluation, you can use the following command:
```bash
python evaluate.py --model_dir <path_to_model> --domain_name <SafetyPointGoal1-v0, SafetyCarGoal1-v0> --num_episodes <number_of_episodes> --trained_model <none, static, dynamic> --video_distractions <none, static, dynamic> --color_distractions <none, static, dynamic> --seed <random_seed> --video_background_path <path_to_video_backgrounds>
```
Example command for evaluating a SAIR agent on the SafetyPointGoal1-v0 environment (trained with no distractions):
```bash
python evaluate.py --model_dir ./weights/sair/SafetyPointGoal1-v0/ --domain_name SafetyPointGoal1-v0 --num_episodes 20 --trained_model none --video_distractions none --color_distractions none
```
Project page: https://sites.google.com/view/safeinvariantrl