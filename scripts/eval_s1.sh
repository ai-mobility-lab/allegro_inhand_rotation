#!/bin/bash
GPUS=$1
CACHE=$2
C=outputs/LeftAllegroHandHora/"${CACHE}"/stage1_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=LeftAllegroHandHora headless=True \
task.env.numEnvs=10240 test=True task.on_evaluation=True \
task.env.object.type=cylinder_default \
task.env.object.sampleProb=[1.0] \
train.algo=PPO \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeCOM=True \
task.env.randomization.randomizeFriction=True \
task.env.randomization.randomizePDGains=True \
task.env.randomization.randomizeScale=True \
task.env.randomization.jointNoiseScale=0.005 \
task.env.reset_height_threshold=0.6 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.ppo.priv_info=True \
train.ppo.output_name=LeftAllegroHandHora/"${CACHE}" \
checkpoint="${C}" \
wandb.enabled=False
