#!/usr/bin/env python

"""Read-only inspection of SO101 serial buses.

This script never enables torque, writes goals, calibrates, or otherwise commands a motor.
Use ``--observe-seconds`` to compare positions over time and identify which serial port
belongs to the arm being moved by hand.
"""

from __future__ import annotations

import argparse
import glob
import time

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler


BAUDRATE = 1_000_000
PRESENT_POSITION = 56
TORQUE_ENABLE = 40
MOTOR_IDS = range(1, 7)


def read_bus(port: str) -> dict | None:
    port_handler = PortHandler(port)
    packet_handler = PacketHandler(0)
    if not port_handler.openPort() or not port_handler.setBaudRate(BAUDRATE):
        return None

    try:
        positions = {}
        torque = {}
        for motor_id in MOTOR_IDS:
            position, comm, error = packet_handler.read2ByteTxRx(
                port_handler, motor_id, PRESENT_POSITION
            )
            torque_value, torque_comm, torque_error = packet_handler.read1ByteTxRx(
                port_handler, motor_id, TORQUE_ENABLE
            )
            if comm == COMM_SUCCESS and error == 0:
                positions[motor_id] = position
            if torque_comm == COMM_SUCCESS and torque_error == 0:
                torque[motor_id] = torque_value

        if not positions:
            return None
        return {"positions": positions, "torque": torque}
    finally:
        port_handler.closePort()


def discover_motor_ports() -> list[str]:
    ports = []
    for path in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        data = read_bus(path)
        if data is not None and len(data["positions"]) == 6:
            ports.append(path)
    return ports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=0.0,
        help="Compare positions after this many seconds. No robot command is sent.",
    )
    args = parser.parse_args()

    ports = discover_motor_ports()
    if not ports:
        raise SystemExit("No SO101 motor bus found.")

    before = {port: read_bus(port) for port in ports}
    for port, data in before.items():
        print(f"{port}: positions={data['positions']} torque={data['torque']}")

    if args.observe_seconds <= 0:
        return

    print(f"Observing for {args.observe_seconds:.1f}s...")
    time.sleep(args.observe_seconds)

    after = {port: read_bus(port) for port in ports}
    for port in ports:
        deltas = {
            motor_id: after[port]["positions"][motor_id] - before[port]["positions"][motor_id]
            for motor_id in before[port]["positions"].keys() & after[port]["positions"].keys()
        }
        print(f"{port}: deltas={deltas} total_abs_change={sum(abs(v) for v in deltas.values())}")


if __name__ == "__main__":
    main()
