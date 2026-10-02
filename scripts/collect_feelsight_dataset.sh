#!/bin/bash

Isaac_Lab=~/lib/IsaacLab/isaaclab.sh
OBJECT_TYPE=ycb_016_pear

# bash $Isaac_Lab -p scripts/collect_stage2_feelsight_dataset.py --enable_cameras --headless \
#     --checkpoint outputs/LeftAllegroHandDigitHora/baseline_sphere/stage2_nn/best.pth \
#     --num_episodes 2 --episode_steps 300 --output_dir data/feelsight_sim \
#     --object_name ${OBJECT_TYPE}_default

bash $Isaac_Lab -p scripts/collect_stage1_feelsight_dataset.py --enable_cameras --headless \
    --task LeftAllegroHandDigitContactHora \
    --checkpoint outputs/LeftAllegroHandDigitContactHora/${OBJECT_TYPE}/stage1_nn/best.pth \
    --num_episodes 20 --episode_steps 400 --output_dir data/feelsight_sim \
    --object_name ${OBJECT_TYPE}