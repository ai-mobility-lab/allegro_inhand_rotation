#!/bin/bash

Isaac_Lab=~/lib/IsaacLab/isaaclab.sh
OBJECT_TYPE=cuboid # sphere, cylinder, cuboid

bash $Isaac_Lab -p scripts/collect_stage2_feelsight_dataset.py --enable_cameras --headless \
    --checkpoint outputs/LeftAllegroHandDigitHora/baseline_sphere/stage2_nn/best.pth \
    --num_episodes 2 --episode_steps 300 --output_dir data/feelsight_sim \
    --object_name ${OBJECT_TYPE}_default