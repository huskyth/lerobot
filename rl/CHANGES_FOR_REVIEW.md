# SO101 HIL-SERL changes for review

This branch is based on the upstream LeRobot `v0.6.1` source tree.

## Main changes

- Added `SO101LeaderFollower` so the leader can follow the policy-controlled
  follower and release torque for human takeover.
- Added keyboard controls for leader takeover: Space takes over, `s` marks
  success, Esc marks failure, and `r` requests rerecord.
- Added `LeaderFollowerProcessor` to map leader end-effector motion into the
  same 4D action space used by the SAC policy.
- Added takeover deadband, bounded IK motion, hold behavior, and safe leader
  re-engagement after reset.
- Finalized RL datasets correctly even when `push_to_hub=false`.
- Fixed replay-buffer image serialization at checkpoint time.
- Added JSON-safe training-config serialization for HIL-SERL's `Any` fields.
- Added SwanLab Cloud logging and disabled it from storing the API key in config.
- Added SO101 push-cube recording/training configurations and helper scripts.

## Important files

- `src/lerobot/teleoperators/so_leader/so_leader_follower.py`
- `src/lerobot/processor/leader_follower_processor.py`
- `src/lerobot/rl/gym_manipulator.py`
- `src/lerobot/rl/learner.py`
- `src/lerobot/rl/actor.py`
- `src/lerobot/common/swanlab_utils.py`
- `rl/README_SO101_HIL_SERL.md`
- `rl/env_config_so101_push_record.json`
- `rl/train_config_hilserl_so101_push.json`

## Runtime artifacts intentionally excluded

- `data/`
- `outputs/`
- `rl/camera_probe/`
- `rl/current_pose.json`
- Python caches
