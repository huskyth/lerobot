#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH="/home/husky/miniconda3/envs/hilserl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

/home/husky/miniconda3/envs/hilserl/bin/python -u -m lerobot.rl.learner \
  --config_path rl/train_config.json \
  2>&1 | tee "logs/hilserl_learner_$(date +%Y%m%d_%H%M%S).txt"
