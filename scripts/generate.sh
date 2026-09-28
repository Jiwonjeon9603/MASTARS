#!/bin/bash
# MASTARS data augmentation with a trained checkpoint. Accepted episodes are
# written under generated_data/.
set -e

GPU=${GPU:-0}
CONFIG=${1:-configs/mpe/simple_spread_medium.yaml}
CHECKPOINT=${2:-logs/mpe/simple_spread-medium/dseed_2000_h8/seed_100/checkpoint/state_100000.pt}

python generate.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --gpu "$GPU" \
    --adapt_threshold 0.01 --generate_episode_nums 1000
