#!/usr/bin/env python

"""Read the follower's current joint pose without commanding motor motion."""

import json
from pathlib import Path

import draccus
import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.rl.gym_manipulator import GymManipulatorConfig


CONFIG = "rl/env_config_so101_push_record.json"
OUTPUT = Path("rl/current_pose.json")


def main() -> None:
    cfg = draccus.parse(
        config_class=GymManipulatorConfig,
        config_path=CONFIG,
        args=[],
    )
    follower = make_robot_from_config(cfg.env.robot)
    motor_names = list(follower.bus.motors)

    print("[POSE] Connecting read-only to the follower. The arm will not be commanded.")
    follower.bus.connect()
    try:
        input("Move the follower to the desired reset pose, then press Enter: ")
        samples = []
        for _ in range(10):
            samples.append(follower.bus.sync_read("Present_Position"))
        pose = {
            name: round(float(np.mean([sample[name] for sample in samples])), 3)
            for name in motor_names
        }
    finally:
        follower.bus.disconnect(disable_torque=False)

    OUTPUT.write_text(json.dumps(pose, indent=2) + "\n")
    print("[POSE] Current pose:")
    print(json.dumps(pose, indent=2))
    print(f"[POSE] Saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
