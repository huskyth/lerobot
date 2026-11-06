#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
config_path="${1:-rl/train_config_hilserl_so101_push.json}"

# ResNet10 is already cached locally. Offline mode avoids the incompatible
# ALL_PROXY="socks://..." path while SwanLab still uses HTTPS_PROXY for cloud sync.
export HF_HUB_OFFLINE=1
unset ALL_PROXY all_proxy

if [[ -z "${SWANLAB_API_KEY:-}" ]]; then
  read -r -s -p "SwanLab API key: " SWANLAB_API_KEY
  echo
  export SWANLAB_API_KEY
fi

exec python -m lerobot.rl.learner \
  --config_path "$config_path"
