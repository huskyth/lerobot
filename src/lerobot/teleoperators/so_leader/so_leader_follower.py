#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from collections import deque
from threading import Event, Thread

import numpy as np

from lerobot.teleoperators.utils import TeleopEvents

from .so_leader import SOLeader

logger = logging.getLogger(__name__)


class SO101LeaderFollower(SOLeader):
    """
    Extended SO101 Leader that can both lead (human control) and follow (mimic follower).

    This class adds leader-follower functionality where:
    - In follow mode: The leader arm mimics the follower's position (torque enabled)
    - In lead mode: Human controls the leader (torque disabled) and provides actions
    """

    def __init__(self, config):
        super().__init__(config)

        # Leader-follower state
        self.is_intervening = False
        self.leader_torque_enabled = True
        # When disabled, the leader still supplies intervention actions but does
        # not continuously track the follower during normal policy execution.
        self.enable_follower_tracking = True

        # Tracking error for automatic intervention detection
        self.leader_tracking_error_queue = deque(maxlen=4)

        # Keyboard event handling
        self.keyboard_events = {
            "intervention": False,
            "success": False,
            "failure": False,
            "rerecord": False,
        }
        self.keyboard_thread = None
        self.stop_event = Event()
        self.toggle_sequence = 0
        self._space_held = False

        # Store last follower position for action computation
        self.last_follower_pos = None

    @property
    def action_features(self) -> dict:
        """Describe the 4-D end-effector action produced by HIL leader mode."""
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def connect(self, calibrate: bool = True) -> None:
        """Connect and configure for leader-follower mode."""
        super().connect(calibrate)

        # Configure for leader-follower mode with lower gains
        # Lower gains allow manual intervention without injury risk
        self.bus.sync_write("Torque_Enable", 1)
        for motor in self.bus.motors:
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 16)

        # Start keyboard listener
        self._start_keyboard_listener()

        print("[主臂] 操作提示:")
        print("  - Space：在“主臂接管”和“跟随模式”之间切换")
        print("  - 跟随模式：主臂会跟随从臂，从臂保持不动")
        print("  - 接管模式：可以用手扳动主臂，从臂跟随主臂末端")
        print("  - s：成功结束本回合并保存")
        print("  - ESC：失败结束本回合")
        print("  - r：丢弃本回合并重录")
        print(
            f"[主臂 进程={os.getpid()}] 初始状态=跟随中 扭矩=开启",
            flush=True,
        )

    def _start_keyboard_listener(self):
        """Start keyboard listener thread for intervention control."""
        # Import lazily so merely constructing/configuring a non-leader teleoperator
        # remains possible on headless systems.
        from pynput import keyboard

        space_held = False

        def on_press(key):
            nonlocal space_held
            try:
                if key == keyboard.Key.space:
                    # X11/Wayland key repeat emits repeated on_press events while Space
                    # is held. Only the initial press may change intervention state.
                    if space_held:
                        return
                    space_held = True
                    self.keyboard_events["intervention"] = not self.keyboard_events["intervention"]
                    self.is_intervening = self.keyboard_events["intervention"]
                    self.toggle_sequence += 1
                    state = "接管模式" if self.is_intervening else "跟随模式"
                    logger.info(f"切换到 {state}")
                    print(
                        f"[主臂 进程={os.getpid()}] "
                        f"[按键] 第 {self.toggle_sequence} 次按 Space，正在切换到{state}。",
                        flush=True,
                    )
                elif key == keyboard.Key.esc:
                    self.keyboard_events["failure"] = True
                    print("[按键] 已按 ESC：本回合将标记为失败并结束。", flush=True)
                elif hasattr(key, "char"):
                    if key.char == "s":
                        self.keyboard_events["success"] = True
                        print("[按键] 已按 s：本回合将标记为成功并保存。", flush=True)
                    elif key.char == "r":
                        self.keyboard_events["rerecord"] = True
                        print("[按键] 已按 r：本回合数据将丢弃并重录。", flush=True)
            except Exception as e:
                logger.error(f"Error handling key press: {e}")

        def on_release(key):
            nonlocal space_held
            if key == keyboard.Key.space:
                space_held = False

        def listen():
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                listener.stop()

        self.keyboard_thread = Thread(target=listen, daemon=True)
        self.keyboard_thread.start()

    def send_action(self, action: dict[str, float]) -> None:
        """
        Send position commands to leader arm (follow mode).

        Args:
            action: Dictionary of motor positions to command
        """
        # Store follower position for later use
        self.last_follower_pos = np.array([action.get(f"{motor}.pos", 0) for motor in self.bus.motors])

        if not self.is_intervening:
            if not self.enable_follower_tracking:
                return
            # Follow mode: enable torque and track follower
            if not self.leader_torque_enabled:
                self.bus.sync_write("Torque_Enable", 1)
                self.leader_torque_enabled = True
                print(
                    f"[主臂 进程={os.getpid()}] "
                    "[模式] 已进入跟随模式：主臂会跟随从臂，请松开主臂。",
                    flush=True,
                )

            # Send follower positions to leader
            goal_pos = {motor: action[f"{motor}.pos"] for motor in self.bus.motors}
            self.bus.sync_write("Goal_Position", goal_pos)

            # Track error for automatic intervention detection
            current_pos = self.bus.sync_read("Present_Position")
            current_array = np.array([current_pos[motor] for motor in self.bus.motors])
            error = np.linalg.norm(self.last_follower_pos[:-1] - current_array[:-1])
            self.leader_tracking_error_queue.append(error)

    def get_action(self) -> dict[str, float]:
        """
        Get action from leader arm.

        In follow mode: Returns neutral/current positions
        In lead mode: Returns actual leader positions for follower to track
        """
        start = time.perf_counter()

        if self.is_intervening:
            # Lead mode: disable torque if needed and return leader positions
            if self.leader_torque_enabled:
                self.bus.sync_write("Torque_Enable", 0)
                self.leader_torque_enabled = False
                print(
                    f"[主臂 进程={os.getpid()}] "
                    "[模式] 已进入主臂接管：现在可以扳动主臂；"
                    "再按一次 Space 返回跟随。",
                    flush=True,
                )

            # Get current leader position
            action = self.bus.sync_read("Present_Position")
            action = {f"{motor}.pos": val for motor, val in action.items()}

            # Track error
            if self.last_follower_pos is not None:
                current_array = np.array([action[f"{motor}.pos"] for motor in self.bus.motors])
                error = np.linalg.norm(self.last_follower_pos[:-1] - current_array[:-1])
                self.leader_tracking_error_queue.append(error)
        else:
            # Follow mode: return current/neutral positions
            action = self.bus.sync_read("Present_Position")
            action = {f"{motor}.pos": val for motor, val in action.items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """Get current keyboard events."""
        events = {}

        # Map keyboard events to TeleopEvents
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

        # Always report intervention state
        events[TeleopEvents.IS_INTERVENTION] = self.is_intervening

        return events

    def release_for_episode_reset(self) -> None:
        """Release torque so the operator can move the leader during episode reset."""
        if self.leader_torque_enabled:
            self.bus.sync_write("Torque_Enable", 0)
            self.leader_torque_enabled = False
        print(
            f"[主臂 进程={os.getpid()}] [复位] 主臂扭矩已关闭，"
            "请利用准备窗口将主臂扳回初始姿态附近。",
            flush=True,
        )

    def resume_after_episode_reset(self) -> bool:
        """Resume following after episode reset unless intervention is active."""
        if self.is_intervening:
            print(
                f"[主臂 进程={os.getpid()}] [复位] 当前仍在接管模式，"
                "主臂保持可扳动；按 Space 后恢复跟随。",
                flush=True,
            )
            return False

        if not self.leader_torque_enabled:
            self.bus.sync_write("Torque_Enable", 1)
            self.leader_torque_enabled = True
        print(
            f"[主臂 进程={os.getpid()}] [复位] 准备时间结束，主臂恢复跟随。",
            flush=True,
        )
        return True

    def disconnect(self) -> None:
        """Disconnect and cleanup."""
        self.stop_event.set()
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
        super().disconnect()

    def reset(self) -> None:
        """Reset leader-follower state."""
        self.is_intervening = False
        self.leader_torque_enabled = True
        self._space_held = False
        self.leader_tracking_error_queue.clear()
        self.keyboard_events = {
            "intervention": False,
            "success": False,
            "failure": False,
            "rerecord": False,
        }
