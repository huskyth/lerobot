#!/home/husky/miniconda3/envs/hilserl/bin/python
"""Read the current SO-101 end-effector pose without commanding the robot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "rl" / "train_config.json"
MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read the current SO-101 end-effector pose. This script never sends actions."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="HIL-SERL JSON config.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_training_processes() -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            proc_stat = (entry / "stat").read_text(encoding="utf-8").split()
            if len(proc_stat) > 2 and proc_stat[2] == "Z":
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "lerobot.rl.actor" in command or "lerobot.rl.learner" in command:
            matches.append((int(entry.name), command.strip()))
    return matches


def make_robot_config(config: dict) -> SOFollowerRobotConfig:
    robot_config = config["env"]["robot"]
    if robot_config.get("type") not in {"so101_follower", "so100_follower"}:
        raise ValueError(f"Unsupported robot type: {robot_config.get('type')}")
    calibration_dir = robot_config.get("calibration_dir")
    return SOFollowerRobotConfig(
        id=robot_config.get("id"),
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        port=robot_config["port"],
        disable_torque_on_disconnect=False,
        max_relative_target=None,
        cameras={},
        use_degrees=robot_config.get("use_degrees", True),
        position_p_coefficient=robot_config.get("position_p_coefficient", 16),
        position_i_coefficient=robot_config.get("position_i_coefficient", 0),
        position_d_coefficient=robot_config.get("position_d_coefficient", 32),
        num_read_retries=robot_config.get("num_read_retries", 2),
    )


def main() -> int:
    args = parse_args()
    config = load_json(args.config.expanduser().resolve())
    ik_config = config["env"]["processor"]["inverse_kinematics"]
    urdf_path = Path(ik_config["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = REPO_ROOT / urdf_path

    training_processes = find_training_processes()
    if training_processes:
        details = "\n".join(f"  PID {pid}: {command}" for pid, command in training_processes)
        raise RuntimeError(
            "Stop the HIL-SERL actor and learner before reading the arm serial bus.\n"
            f"{details}"
        )

    robot = make_robot_from_config(make_robot_config(config))
    robot.bus.connect()
    try:
        if not robot.is_calibrated:
            raise RuntimeError("Robot calibration is unavailable; refusing to report an invalid pose.")

        positions = robot.bus.sync_read(
            "Present_Position",
            num_retry=config["env"]["robot"].get("num_read_retries", 2),
        )
        joints_deg = np.array([float(positions[motor]) for motor in MOTOR_NAMES], dtype=float)
        kinematics = RobotKinematics(
            str(urdf_path),
            target_frame_name=ik_config["target_frame_name"],
        )
        pose = kinematics.forward_kinematics(joints_deg)
        xyz_mm = pose[:3, 3] * 1000.0

        print("Read-only mode: no actuation command was sent.")
        print(f"Current XYZ: {np.round(xyz_mm, 3).tolist()} mm")
        print(f"Current Z: {xyz_mm[2]:.3f} mm")
        print(f"Configured minimum Z: {float(ik_config['end_effector_bounds']['min'][2]) * 1000:.3f} mm")
        print("Current joints (deg):")
        for motor, value in zip(MOTOR_NAMES, joints_deg, strict=True):
            print(f"  {motor}: {value:.3f}")
    finally:
        # Do not change torque state; close the serial port only.
        robot.bus.disconnect(disable_torque=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
