#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
config_path="${1:-rl/train_config_hilserl_so101_push.json}"

# Use the cached ResNet10 weights; actor does not need Hugging Face network access.
export HF_HUB_OFFLINE=1
unset ALL_PROXY all_proxy

exec python -m lerobot.rl.actor \
  --config_path "$config_path"
