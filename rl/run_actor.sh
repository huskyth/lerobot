#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export LD_LIBRARY_PATH="/home/husky/miniconda3/envs/hilserl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

actor_run_dir="outputs/hilserl/so101_push_cube_formal20_actor_$(date +%Y%m%d_%H%M%S)"
checkpoint_policy="outputs/hilserl/so101_push_cube_hardcube15_20261112/checkpoints/last/pretrained_model"
policy_args=()

if [[ -f "$checkpoint_policy/model.safetensors" ]]; then
  policy_args+=(--policy.pretrained_path "$checkpoint_policy")
fi

/home/husky/miniconda3/envs/hilserl/bin/python -u -m lerobot.rl.actor \
  --config_path rl/train_config.json \
  --output_dir "$actor_run_dir" \
  "${policy_args[@]}" \
  2>&1 | tee "logs/hilserl_actor_$(date +%Y%m%d_%H%M%S).txt"
