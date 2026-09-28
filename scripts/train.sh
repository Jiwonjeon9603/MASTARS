#!/bin/bash
# Train the MASTARS diffusion model (+ inverse-dynamics / transition / reward /
# value models) on an offline dataset. Checkpoints are written under logs/.
set -e

GPU=${GPU:-0}
CONFIG=${1:-configs/mpe/simple_spread_medium.yaml}

python train.py --config "$CONFIG" --gpu "$GPU"
