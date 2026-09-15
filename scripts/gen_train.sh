#!/bin/bash

GPUS=$1
SEED=$2
CACHE=$3

bash scripts/gen_grasp.sh ${GPUS}
bash scripts/train_s1.sh ${GPUS} ${SEED} ${CACHE}