#!/home/husky/miniconda3/envs/hilserl/bin/python
"""Run a trained HIL-SERL policy without learner, replay, or training writes.

This follows the actor's observation/action processors and safety bounds, but
uses only the selected checkpoint for inference.  Space toggles human takeover,
s marks success, Esc ends an episode as failure, and Ctrl+C exits.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

# Checkpoints are local. Prevent Transformers/Hugging Face from consulting the
# network, which also avoids incompatible system proxy settings on the robot PC.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch

# These imports populate the draccus registries used by train_config.json.
import lerobot.cameras.opencv  # noqa: F401
import lerobot.robots.so_follower  # noqa: F401
import lerobot.teleoperators.gamepad  # noqa: F401
import lerobot.teleoperators.keyboard  # noqa: F401
import lerobot.teleoperators.so_leader  # noqa: F401
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import (
    TransitionKey,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.rl.gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_follower_position,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
from lerobot.teleoperators.so_leader.so_leader_follower import SO101LeaderFollower
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "rl" / "train_config.json"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "hilserl"
    / "so101_push_cube_hardcube15_20261112"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"No model.safetensors found in checkpoint: {path}")
    return path


def running_training_processes() -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8").split()
            if len(stat) > 2 and stat[2] == "Z":
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "lerobot.rl.actor" in command or "lerobot.rl.learner" in command:
            matches.append((int(entry.name), command.strip()))
    return matches


def policy_action(
    cfg: TrainRLServerPipelineConfig,
    policy: torch.nn.Module,
    preprocessor: Any,
    postprocessor: Any,
    observation: dict[str, Any],
) -> torch.Tensor:
    filtered = {key: value for key, value in observation.items() if key in cfg.policy.input_features}
    normalized_observation = preprocessor.process_observation(filtered)
    action = transition_to_policy_action(
        policy_output_to_transition(policy.select_action(batch=normalized_observation))
    )

    if cfg.policy.num_discrete_actions is not None:
        continuous_action = postprocessor.process_action(action[..., :-1])
        discrete_action = action[..., -1:].to(
            device=continuous_action.device,
            dtype=continuous_action.dtype,
        )
        action = torch.cat([continuous_action, discrete_action], dim=-1)
        gripper = cfg.env.processor.gripper
        if gripper is not None and gripper.lock_gripper:
            action = action.clone()
            action[..., -1] = 1.0
    else:
        action = postprocessor.process_action(action)

    return action


def reset_episode(env: Any, env_processor: Any, action_processor: Any, teleop: Any):
    manual_leader = isinstance(teleop, SO101LeaderFollower)
    if manual_leader:
        teleop.release_for_episode_reset()
    try:
        return reset_and_build_transition(env, env_processor, action_processor)
    finally:
        if manual_leader:
            teleop.resume_after_episode_reset()


def return_follower_to_initial(env: Any, cfg: TrainRLServerPipelineConfig) -> None:
    robot = getattr(env, "robot", None)
    reset_cfg = cfg.env.processor.reset
    if robot is None or not getattr(robot, "is_connected", False) or reset_cfg is None:
        return
    positions = reset_cfg.fixed_reset_joint_positions
    if positions is None:
        return
    print("[INFER] Returning follower to its initial joint position.")
    reset_follower_position(robot, np.asarray(positions, dtype=np.float32))


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    checkpoint = resolve_checkpoint(args.checkpoint)
    active = running_training_processes()
    if active:
        details = "\n".join(f"  PID {pid}: {cmd}" for pid, cmd in active)
        raise RuntimeError(f"Stop learner/actor before inference:\n{details}")

    cfg = TrainRLServerPipelineConfig.from_pretrained(str(args.config.expanduser().resolve()))
    if args.device is not None:
        cfg.policy.device = args.device

    print(f"Checkpoint: {checkpoint}")
    print(f"Policy: {cfg.policy.type}")
    print(f"Action bounds: {cfg.env.processor.inverse_kinematics.end_effector_bounds}")
    print("Inference only: no learner, no replay buffer, no training data will be written.")

    device = get_safe_torch_device(cfg.policy.device, log=True)
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        pretrained_path=checkpoint,
    )
    policy = policy.to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=checkpoint,
        dataset_stats=cfg.policy.dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    if args.dry_run:
        print(f"Dry run OK: policy and processors loaded on {device}.")
        return 0

    env, teleop = make_robot_env(cfg=cfg.env, play_sounds=False)
    if isinstance(teleop, SO101LeaderFollower):
        teleop.enable_follower_tracking = False
        print("[INFER] Leader-following is disabled; Space still enables manual takeover.")
    env_processor, action_processor = make_processors(
        env,
        teleop,
        cfg.env,
        cfg.policy.device,
    )

    print("Policy loaded. Space=takeover/release, s=success, Esc=failure, Ctrl+C=exit.")
    started = time.perf_counter()
    completed = 0
    try:
        for episode in range(args.episodes):
            if args.max_seconds is not None and time.perf_counter() - started >= args.max_seconds:
                break

            transition = reset_episode(env, env_processor, action_processor, teleop)
            episode_reward = 0.0
            episode_steps = 0
            print(f"\n[INFER] Episode {episode + 1}/{args.episodes}")

            while True:
                if args.max_seconds is not None and time.perf_counter() - started >= args.max_seconds:
                    return 0

                loop_started = time.perf_counter()
                observation = transition[TransitionKey.OBSERVATION]
                with torch.inference_mode():
                    action = policy_action(
                        cfg,
                        policy,
                        preprocessor,
                        postprocessor,
                        observation,
                    )

                next_transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )
                episode_reward += float(next_transition[TransitionKey.REWARD])
                episode_steps += 1
                transition = next_transition

                if transition.get(TransitionKey.DONE) or transition.get(TransitionKey.TRUNCATED):
                    completed += 1
                    info = transition.get(TransitionKey.INFO) or {}
                    success = bool(info.get(TeleopEvents.SUCCESS, False))
                    print(
                        f"[INFER] Episode {episode + 1}: steps={episode_steps}, "
                        f"reward={episode_reward:.2f}, success={success}"
                    )
                    break

                elapsed = time.perf_counter() - loop_started
                precise_sleep(max(1.0 / cfg.env.fps - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\n[INFER] Interrupted.")
    finally:
        env_available = "env" in locals()
        if env_available:
            try:
                return_follower_to_initial(env, cfg)
            except Exception:
                print("[INFER] Failed to return follower to its initial position.")
        if hasattr(teleop, "disconnect"):
            teleop.disconnect()
        if env_available and hasattr(env, "close"):
            env.close()

    print(f"[INFER] Finished {completed} episode(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
