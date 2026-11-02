#!/usr/bin/env python

"""Read the follower's current calibrated joint pose without enabling torque."""

from __future__ import annotations

import json

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

from lerobot.utils.constants import HF_LEROBOT_CALIBRATION


PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79015851-if00"
CALIBRATION = (
    HF_LEROBOT_CALIBRATION
    / "robots"
    / "so_follower"
    / "follower_arm.json"
)
MOTORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
PRESENT_POSITION = 56


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text())
    port_handler = PortHandler(PORT)
    packet_handler = PacketHandler(0)
    if not port_handler.openPort() or not port_handler.setBaudRate(1_000_000):
        raise SystemExit(f"Failed to open {PORT}")

    try:
        normalized = []
        for motor_id, motor_name in enumerate(MOTORS, start=1):
            raw, comm, error = packet_handler.read2ByteTxRx(
                port_handler, motor_id, PRESENT_POSITION
            )
            if comm != COMM_SUCCESS or error != 0:
                raise RuntimeError(f"Failed to read {motor_name}: comm={comm}, error={error}")

            cal = calibration[motor_name]
            if motor_name == "gripper":
                value = (raw - cal["range_min"]) / (
                    cal["range_max"] - cal["range_min"]
                ) * 100
                value = min(100.0, max(0.0, value))
            else:
                midpoint = (cal["range_min"] + cal["range_max"]) / 2
                value = (raw - midpoint) * 360 / 4095
            normalized.append(round(float(value), 4))

        print("fixed_reset_joint_positions:")
        print(json.dumps(normalized))
    finally:
        port_handler.closePort()


if __name__ == "__main__":
    main()
