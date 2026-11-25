#!/home/husky/miniconda3/envs/hilserl/bin/python
"""Interactively inspect and control the SO-101 end-effector Z height.

The configured end-effector bounds are treated as a hard safety limit. The script
preserves the current X/Y position and orientation, while moving Z gradually with
inverse kinematics. It does not start any camera devices and is intended to be run
only while the HIL-SERL actor is stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "rl" / "train_config.json"
ESCAPE = "\x1b"
JOINT_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely move the SO-101 to its configured minimum Z and inspect it by keyboard."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="HIL-SERL JSON config.")
    parser.add_argument(
        "--clearance-mm",
        type=float,
        default=0.0,
        help="Stop this many millimetres above configured min Z. Default: 0 (exact boundary).",
    )
    parser.add_argument(
        "--target-z-mm",
        type=float,
        default=None,
        help="Initial absolute end-effector Z target in millimetres.",
    )
    parser.add_argument(
        "--travel-step-mm",
        type=float,
        default=2.0,
        help="Maximum automatic Z movement per control step.",
    )
    parser.add_argument(
        "--manual-step-mm",
        type=float,
        default=1.0,
        help="Z movement per W/S key press.",
    )
    parser.add_argument(
        "--z-tolerance-mm",
        type=float,
        default=0.8,
        help="Consider the target reached within this height tolerance.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Joint command rate.")
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=3.0,
        help="Maximum IK joint change per command.",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=30,
        help="Maximum Placo IK solver iterations for each target.",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the final Enter confirmation before automatic movement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and IK using the configured reset pose without connecting hardware.",
    )
    parser.add_argument(
        "--start-current",
        action="store_true",
        help="Skip the automatic Z target and enter interactive control at the current pose.",
    )
    parser.add_argument(
        "--release-torque",
        action="store_true",
        help="Connect, release motor torque, and exit without sending a movement target.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_kinematics(config: dict) -> RobotKinematics:
    ik_config = config["env"]["processor"]["inverse_kinematics"]
    urdf_path = Path(ik_config["urdf_path"])
    if not urdf_path.is_absolute():
        urdf_path = REPO_ROOT / urdf_path
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    return RobotKinematics(
        str(urdf_path),
        target_frame_name=ik_config["target_frame_name"],
    )


def make_robot_config(config: dict, max_joint_step_deg: float) -> SOFollowerRobotConfig:
    robot_config = config["env"]["robot"]
    if robot_config.get("type") not in {"so101_follower", "so100_follower"}:
        raise ValueError(f"Unsupported robot type: {robot_config.get('type')}")

    calibration_dir = robot_config.get("calibration_dir")
    joint_limit = {
        motor: float(max_joint_step_deg if motor != "gripper" else 0.0)
        for motor in JOINT_MOTOR_NAMES
    }
    return SOFollowerRobotConfig(
        id=robot_config.get("id"),
        calibration_dir=Path(calibration_dir) if calibration_dir else None,
        port=robot_config["port"],
        disable_torque_on_disconnect=False,
        max_relative_target=joint_limit,
        cameras={},
        use_degrees=robot_config.get("use_degrees", True),
        position_p_coefficient=robot_config.get("position_p_coefficient", 16),
        position_i_coefficient=robot_config.get("position_i_coefficient", 0),
        position_d_coefficient=robot_config.get("position_d_coefficient", 32),
        num_read_retries=robot_config.get("num_read_retries", 2),
    )


def find_training_processes() -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            proc_stat = (entry / "stat").read_text(encoding="utf-8").split()
            # A zombie retains its old command line but no longer owns devices.
            if len(proc_stat) > 2 and proc_stat[2] == "Z":
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "lerobot.rl.actor" in command or "lerobot.rl.learner" in command:
            matches.append((int(entry.name), command.strip()))
    return matches


def configure_robot(robot) -> None:
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        raise RuntimeError(
            "Robot is not calibrated according to its calibration file. "
            "Run normal LeRobot calibration before using this script."
        )


def get_joint_positions(robot) -> np.ndarray:
    observation = robot.get_observation()
    return np.array(
        [float(observation[f"{motor}.pos"]) for motor in JOINT_MOTOR_NAMES],
        dtype=float,
    )


def send_joints(robot, joints_deg: np.ndarray) -> None:
    action = {
        f"{motor}.pos": float(joints_deg[index])
        for index, motor in enumerate(JOINT_MOTOR_NAMES)
    }
    robot.send_action(action)


def xyz_mm(pose: np.ndarray) -> np.ndarray:
    return pose[:3, 3] * 1000.0


def solve_target_joints(
    kinematics: RobotKinematics,
    current_joints: np.ndarray,
    target_pose: np.ndarray,
    max_joint_step_deg: float,
    ik_iters: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    solved = np.asarray(
        kinematics.inverse_kinematics(
            current_joints,
            target_pose,
            position_weight=1.0,
            orientation_weight=0.01,
            max_iters=ik_iters,
        ),
        dtype=float,
    )
    if solved.shape != current_joints.shape or not np.isfinite(solved).all():
        raise RuntimeError("Inverse kinematics returned an invalid joint target.")

    solved[-1] = current_joints[-1]
    delta = solved - current_joints
    max_delta = float(np.max(np.abs(delta[:-1])))
    if max_delta > max_joint_step_deg:
        solved[:-1] = current_joints[:-1] + delta[:-1] * (max_joint_step_deg / max_delta)

    achieved = kinematics.forward_kinematics(solved)
    if not np.isfinite(achieved).all():
        raise RuntimeError("Forward kinematics returned an invalid pose.")
    return solved, achieved


def validate_pose_in_x_y_bounds(
    pose: np.ndarray,
    bounds: dict,
    tolerance_m: float = 1e-4,
) -> None:
    low = np.asarray(bounds["min"], dtype=float)[:2]
    high = np.asarray(bounds["max"], dtype=float)[:2]
    current = pose[:2, 3]
    if np.any(current < low - tolerance_m) or np.any(current > high + tolerance_m):
        raise RuntimeError(
            "Current X/Y pose is outside the configured workspace. "
            f"Current XY={xyz_mm(pose)[:2].round(2).tolist()} mm, "
            f"bounds XY min={np.round(low * 1000, 2).tolist()} mm, "
            f"max={np.round(high * 1000, 2).tolist()} mm. "
            "Move the arm back into the workspace manually before retrying."
        )


@contextmanager
def keyboard_mode() -> Iterator[None]:
    if not sys.stdin.isatty():
        raise RuntimeError("Keyboard control requires an interactive terminal.")
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")


def print_status(current_pose: np.ndarray, target_z_m: float, floor_z_m: float) -> None:
    current = xyz_mm(current_pose)
    print(
        f"\rEE x={current[0]:7.2f} y={current[1]:7.2f} z={current[2]:7.2f} mm | "
        f"target z={target_z_m * 1000:7.2f} mm | min z={floor_z_m * 1000:7.2f} mm",
        end="",
        flush=True,
    )


def stop_after_key(key: str | None) -> str | None:
    if key == ESCAPE:
        return "release"
    if key in {"q", "Q"}:
        return "hold"
    return None


def move_to_z(
    robot,
    kinematics: RobotKinematics,
    base_pose: np.ndarray,
    target_z_m: float,
    floor_z_m: float,
    ceiling_z_m: float,
    args: argparse.Namespace,
) -> str:
    period = 1.0 / args.fps
    travel_step = args.travel_step_mm / 1000.0
    z_tolerance = args.z_tolerance_mm / 1000.0
    target_pose = base_pose.copy()
    last_z = float(base_pose[2, 3])
    stalled_since = time.monotonic()

    while True:
        try:
            current_joints = get_joint_positions(robot)
            current_pose = kinematics.forward_kinematics(current_joints)
        except KeyboardInterrupt:
            return "hold"

        current_z = float(current_pose[2, 3])
        if abs(current_z - target_z_m) <= z_tolerance:
            print()
            print(
                f"Reached target Z={current_z * 1000:.3f} mm "
                f"(within {args.z_tolerance_mm:.3f} mm of {target_z_m * 1000:.3f} mm)."
            )
            return "continue"

        direction = 1.0 if target_z_m > current_z else -1.0
        next_z = current_z + direction * min(travel_step, abs(target_z_m - current_z))
        next_z = max(next_z, floor_z_m)
        target_pose[2, 3] = next_z

        joints_target, achieved = solve_target_joints(
            kinematics,
            current_joints,
            target_pose,
            args.max_joint_step_deg,
            args.ik_iters,
        )
        if achieved[2, 3] < floor_z_m - 0.0005:
            raise RuntimeError(
                "IK solution would put the end effector below the configured Z minimum; stopping."
            )

        send_joints(robot, joints_target)
        print_status(achieved, next_z, floor_z_m)

        if abs(current_z - last_z) > 0.00005:
            stalled_since = time.monotonic()
            last_z = current_z
        elif time.monotonic() - stalled_since > 2.5:
            print()
            backoff_pose = current_pose.copy()
            if direction < 0:
                backoff_pose[2, 3] = min(current_z + 0.002, ceiling_z_m)
            for _ in range(5):
                try:
                    measured_joints = get_joint_positions(robot)
                    joints_target, _ = solve_target_joints(
                        kinematics,
                        measured_joints,
                        backoff_pose,
                        args.max_joint_step_deg,
                        args.ik_iters,
                    )
                    send_joints(robot, joints_target)
                except KeyboardInterrupt:
                    break
                time.sleep(period)
            raise RuntimeError(
                "No measurable Z progress for 2.5 s. The arm may be blocked by the table "
                "or the target may be unreachable. "
                f"Current Z={current_z * 1000:.3f} mm, requested Z={next_z * 1000:.3f} mm, "
                f"IK Z={achieved[2, 3] * 1000:.3f} mm. "
                "The controller retracted 2 mm when descending and is holding its final target."
            )

        action = stop_after_key(read_key(period))
        if action is not None:
            print()
            return action


def interactive_loop(
    robot,
    kinematics: RobotKinematics,
    target_pose: np.ndarray,
    floor_z_m: float,
    ceiling_z_m: float,
    args: argparse.Namespace,
) -> str:
    period = 1.0 / args.fps
    manual_step = args.manual_step_mm / 1000.0
    print("Controls: W=up 1mm, S=down 1mm, M=go to min Z, R=hold current pose, "
          "Q=quit with torque on, Esc=quit and release torque.")

    while True:
        key = read_key(0.0)
        action = stop_after_key(key)
        if action is not None:
            print()
            return action

        current_joints = get_joint_positions(robot)
        current_pose = kinematics.forward_kinematics(current_joints)

        if key in {"w", "W"}:
            target_pose[2, 3] = min(float(target_pose[2, 3]) + manual_step, ceiling_z_m)
        elif key in {"s", "S"}:
            target_pose[2, 3] = max(float(target_pose[2, 3]) - manual_step, floor_z_m)
        elif key in {"m", "M"}:
            target_pose = current_pose.copy()
            target_pose[2, 3] = floor_z_m
        elif key in {"r", "R"}:
            target_pose = current_pose.copy()

        joints_target, achieved = solve_target_joints(
            kinematics,
            current_joints,
            target_pose,
            args.max_joint_step_deg,
            args.ik_iters,
        )
        if achieved[2, 3] < floor_z_m - 0.0005:
            # Near the far workspace edge, a rate-limited joint step can initially
            # dip below the Cartesian floor even when the final target is valid.
            # Retry with progressively smaller moves from the measured pose.
            delta = joints_target - current_joints
            for scale in (0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
                candidate = current_joints + delta * scale
                candidate[-1] = current_joints[-1]
                candidate_pose = kinematics.forward_kinematics(candidate)
                if candidate_pose[2, 3] >= floor_z_m - 0.0005:
                    joints_target = candidate
                    achieved = candidate_pose
                    break
            else:
                raise RuntimeError("IK solution exceeded the safe Z minimum; stopping.")

        send_joints(robot, joints_target)
        print_status(achieved, float(target_pose[2, 3]), floor_z_m)
        time.sleep(period)


def dry_run(config: dict, kinematics: RobotKinematics, args: argparse.Namespace) -> None:
    bounds = config["env"]["processor"]["inverse_kinematics"]["end_effector_bounds"]
    reset_joints = np.asarray(
        config["env"]["processor"]["reset"]["fixed_reset_joint_positions"],
        dtype=float,
    )
    reset_pose = kinematics.forward_kinematics(reset_joints)
    floor_z = float(bounds["min"][2])
    initial_target_z = (
        args.target_z_mm / 1000.0
        if args.target_z_mm is not None
        else floor_z + args.clearance_mm / 1000.0
    )
    target_pose = reset_pose.copy()
    target_pose[2, 3] = initial_target_z
    solved, achieved = solve_target_joints(
        kinematics,
        reset_joints,
        target_pose,
        args.max_joint_step_deg,
        args.ik_iters,
    )
    position_error_mm = np.linalg.norm(achieved[:3, 3] - target_pose[:3, 3]) * 1000.0
    print(f"Configured reset XYZ: {np.round(xyz_mm(reset_pose), 3).tolist()} mm")
    print(f"Configured Z range: {bounds['min'][2] * 1000:.3f} .. "
          f"{bounds['max'][2] * 1000:.3f} mm")
    print(f"Initial test target Z: {initial_target_z * 1000:.3f} mm")
    print(f"Hard minimum Z: {floor_z * 1000:.3f} mm")
    print(f"IK joint error norm: {np.linalg.norm(solved - reset_joints):.3f} deg")
    print(f"IK position error: {position_error_mm:.3f} mm")


def main() -> int:
    args = parse_args()
    if (
        args.fps <= 0
        or args.travel_step_mm <= 0
        or args.manual_step_mm <= 0
        or args.z_tolerance_mm <= 0
        or args.ik_iters <= 0
    ):
        raise ValueError("fps and movement steps must be positive.")
    if args.clearance_mm < 0:
        raise ValueError("clearance-mm cannot be negative.")

    config = load_config(args.config.expanduser().resolve())
    kinematics = load_kinematics(config)
    if args.dry_run:
        dry_run(config, kinematics, args)
        return 0

    training_processes = find_training_processes()
    if training_processes:
        details = "\n".join(f"  PID {pid}: {command}" for pid, command in training_processes)
        raise RuntimeError(
            "Stop the HIL-SERL actor and learner before running this hardware controller.\n"
            f"{details}"
        )

    bounds = config["env"]["processor"]["inverse_kinematics"]["end_effector_bounds"]
    configured_min_z = float(bounds["min"][2])
    floor_z = configured_min_z
    initial_target_z = (
        args.target_z_mm / 1000.0
        if args.target_z_mm is not None
        else floor_z + args.clearance_mm / 1000.0
    )
    ceiling_z = float(bounds["max"][2])
    if initial_target_z < floor_z or initial_target_z > ceiling_z:
        raise ValueError(
            "Initial target Z must be within the configured Z range "
            f"({floor_z * 1000:.3f} .. {ceiling_z * 1000:.3f} mm)."
        )

    robot_config = make_robot_config(config, args.max_joint_step_deg)
    robot = make_robot_from_config(robot_config)
    release_torque = False

    try:
        configure_robot(robot)
        current_joints = get_joint_positions(robot)
        current_pose = kinematics.forward_kinematics(current_joints)
        validate_pose_in_x_y_bounds(current_pose, bounds)
        print(f"Connected to {robot_config.port}.")
        print(f"Current XYZ: {np.round(xyz_mm(current_pose), 3).tolist()} mm")
        print(f"Configured Z range: {configured_min_z * 1000:.3f} .. {ceiling_z * 1000:.3f} mm")
        print(f"Initial test target Z: {initial_target_z * 1000:.3f} mm")
        print(f"Hard minimum Z: {floor_z * 1000:.3f} mm")

        if args.release_torque:
            robot.bus.disable_torque(num_retry=5)
            print("Motor torque released. Support the arm before moving it manually.")
            return 0

        if not args.no_confirm:
            input(
                "Confirm the table, camera, cables and arm path are clear, then press Enter. "
                "Ctrl-C aborts: "
            )

        with keyboard_mode():
            if args.start_current:
                target_pose = current_pose.copy()
                action = interactive_loop(
                    robot,
                    kinematics,
                    target_pose,
                    floor_z,
                    ceiling_z,
                    args,
                )
            else:
                action = move_to_z(
                    robot,
                    kinematics,
                    current_pose,
                    initial_target_z,
                    floor_z,
                    ceiling_z,
                    args,
                )
                if action == "continue":
                    target_pose = current_pose.copy()
                    target_pose[2, 3] = initial_target_z
                    action = interactive_loop(
                        robot,
                        kinematics,
                        target_pose,
                        floor_z,
                        ceiling_z,
                        args,
                    )
            release_torque = action == "release"
    except KeyboardInterrupt:
        print("\nInterrupted. Keeping the last commanded position.")
    finally:
        if robot.is_connected:
            if release_torque:
                robot.bus.disable_torque(num_retry=5)
            robot.disconnect()
        print("\nHardware controller closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
