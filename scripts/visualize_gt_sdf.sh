#!/bin/bash

DATA_DIR=data/feelsight_sim/dextouch_pear/00

python scripts/tools/visualize_gt_sdf.py \
  ${DATA_DIR}/gt_sdf_voxel=0.0005.npz
