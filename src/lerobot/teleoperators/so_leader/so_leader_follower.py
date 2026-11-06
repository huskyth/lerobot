#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import time
from threading import Event, Thread
from typing import Any

import numpy as np

from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.keyboard_input import pynput_can_capture

from .so_leader import SOLeader

logger = logging.getLogger(__name__)


class SO101LeaderFollower(SOLeader):
    """SO101 leader that follows the follower and supports keyboard takeover."""

    def __init__(self, config):
        super().__init__(config)
        self.is_intervening = False
        self.leader_torque_enabled = False
        self.keyboard_events = {
            "success": False,
            "failure": False,
            "rerecord": False,
        }
        self.keyboard_thread: Thread | None = None
        self.stop_event = Event()
        self._keys_down: set[Any] = set()

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def connect(self, calibrate: bool = True) -> None:
        self.stop_event.clear()
        super().connect(calibrate)
        try:
            self.bus.sync_write("Torque_Enable", 1)
            self.leader_torque_enabled = True
            self._start_keyboard_listener()
        except Exception:
            self.bus.disconnect(disable_torque=True)
            raise
        print("SO101 leader-follower mode:")
        print("  Space: toggle human intervention")
        print("  s: success, Esc: failure, r: rerecord")

    def _start_keyboard_listener(self) -> None:
        if not pynput_can_capture():
            self.bus.disconnect()
            raise RuntimeError(
                "Leader intervention requires keyboard capture. Run from an X11 desktop session "
                "(not Wayland/headless) with access to the active display."
            )

        from pynput import keyboard

        def on_press(key) -> None:
            # Key repeat otherwise toggles intervention many times while Space is held.
            if key in self._keys_down:
                return
            self._keys_down.add(key)
            if key == keyboard.Key.space:
                self.is_intervening = not self.is_intervening
                state = "ON - leader controls follower" if self.is_intervening else "OFF - leader follows follower"
                print(f"\n[LEADER] Intervention {state}\n", flush=True)
            elif key == keyboard.Key.esc:
                self.keyboard_events["failure"] = True
            elif getattr(key, "char", None) == "s":
                self.keyboard_events["success"] = True
            elif getattr(key, "char", None) == "r":
                self.keyboard_events["rerecord"] = True

        def on_release(key) -> None:
            self._keys_down.discard(key)

        def listen() -> None:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                while not self.stop_event.is_set():
                    time.sleep(0.05)
                listener.stop()

        self.keyboard_thread = Thread(target=listen, name="so101-leader-events", daemon=True)
        self.keyboard_thread.start()

    def send_action(self, action: dict[str, float]) -> None:
        """Command the leader to track the follower while not intervening."""
        if self.is_intervening:
            return
        current = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        max_step = float(self.config.leader_follow_max_step_deg)
        goals = {
            motor: float(
                np.clip(
                    float(action[f"{motor}.pos"]) - float(current[motor]),
                    -max_step,
                    max_step,
                )
                + float(current[motor])
            )
            for motor in self.bus.motors
        }
        # Write the limited target before enabling torque to avoid a startup jump.
        self.bus.sync_write("Goal_Position", goals)
        if not self.leader_torque_enabled:
            self.bus.sync_write("Torque_Enable", 1)
            self.leader_torque_enabled = True
            print(
                f"[LEADER] Re-engaging follow at max {max_step:.1f} deg/step.",
                flush=True,
            )

    def get_action(self) -> dict[str, float]:
        if self.is_intervening and self.leader_torque_enabled:
            self.bus.sync_write("Torque_Enable", 0)
            self.leader_torque_enabled = False
        positions = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        return {f"{motor}.pos": value for motor, value in positions.items()}

    def get_teleop_events(self) -> dict[TeleopEvents, Any]:
        events: dict[TeleopEvents, Any] = {
            TeleopEvents.IS_INTERVENTION: self.is_intervening,
        }
        if self.keyboard_events["success"]:
            events[TeleopEvents.SUCCESS] = True
            self.keyboard_events["success"] = False
        if self.keyboard_events["failure"]:
            events[TeleopEvents.FAILURE] = True
            events[TeleopEvents.TERMINATE_EPISODE] = True
            self.keyboard_events["failure"] = False
        if self.keyboard_events["rerecord"]:
            events[TeleopEvents.RERECORD_EPISODE] = True
            events[TeleopEvents.TERMINATE_EPISODE] = True
            self.keyboard_events["rerecord"] = False
        return events

    def reset(self) -> None:
        self.is_intervening = False
        self.keyboard_events = {"success": False, "failure": False, "rerecord": False}

    def begin_reset(self) -> None:
        """Release leader torque before the follower starts its reset trajectory."""
        self.is_intervening = False
        if self.leader_torque_enabled and self.is_connected:
            self.bus.sync_write("Torque_Enable", 0)
            self.leader_torque_enabled = False
            print(
                "[LEADER] Torque released. You may manually reposition the leader during reset.",
                flush=True,
            )

    def disconnect(self) -> None:
        self.stop_event.set()
        self._keys_down.clear()
        if self.keyboard_thread is not None:
            self.keyboard_thread.join(timeout=1.0)
            self.keyboard_thread = None
        super().disconnect()
