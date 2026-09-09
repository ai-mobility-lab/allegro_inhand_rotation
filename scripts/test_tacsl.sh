#!/bin/bash

Isaac_Lab=~/lib/IsaacLab/isaaclab.sh
OBJECT_TYPE=dextouch_pear

# bash $Isaac_Lab -p scripts/tools/tacsl_sensor_demo.py --enable_cameras --steps 100 --save_every 20
bash $Isaac_Lab -p scripts/tools/tacsl_nut_push_demo.py --enable_cameras --steps 100 --save_every 20
