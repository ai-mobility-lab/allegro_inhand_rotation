#!/bin/bash
CACHE=$1
python train.py task=LeftAllegroHandDigitHora headless=False pipeline=gpu \
task.env.numEnvs=64 test=True \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=LeftAllegroHandDigitHora/"${CACHE}" \
checkpoint=outputs/LeftAllegroHandDigitHora/"${CACHE}"/stage2_nn/best.pth \
wandb.enabled=False \
