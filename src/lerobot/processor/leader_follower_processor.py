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

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.pipeline import (
    EnvTransition,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents


@ProcessorStepRegistry.register("leader_follower_processor")
@dataclass
class LeaderFollowerProcessor(ProcessorStep):
    """
    Processor for leader-follower teleoperation mode.

    This processor:
    1. Sends follower positions to leader arm when not intervening
    2. Computes EE delta actions from leader when intervening
    3. Handles teleop events from the leader device
    """

    leader_device: Teleoperator
    motor_names: list[str]
    robot: Robot
    kinematics: RobotKinematics
    end_effector_step_sizes: np.ndarray | None = None
    use_gripper: bool = True
    lock_gripper: bool = False
    prev_leader_gripper: float | None = None
    max_gripper_pos: float = 100.0
    _last_log_time: float = field(default=0.0, init=False, repr=False)
    _last_leader_ee: np.ndarray | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Process transition with leader-follower logic."""
        # Get current follower position from complementary data
        raw_joint_pos = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("raw_joint_positions")
        if raw_joint_pos is not None:
            # v0.7.0 stores raw joints with a ".pos" suffix.
            raw_joint_pos = {
                motor: raw_joint_pos.get(motor, raw_joint_pos.get(f"{motor}.pos", 0.0))
                for motor in self.motor_names
            }
            # Send follower position to leader (for follow mode)
            follower_action = {f"{motor}.pos": float(raw_joint_pos[motor]) for motor in self.motor_names}
            self.leader_device.send_action(follower_action)

        # Only compute EE action if intervention is active
        # (AddTeleopEventsAsInfo already added IS_INTERVENTION to info)
        info = transition.get(TransitionKey.INFO, {})
        if info.get(TeleopEvents.IS_INTERVENTION, False):
            # Get leader joint positions from teleop_action
            # (AddTeleopActionAsComplimentaryData already got the action)
            complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
            teleop_action = complementary.get("teleop_action", {})

            if isinstance(teleop_action, dict) and raw_joint_pos is not None:
                # Extract leader positions from teleop action dict
                leader_pos = np.array([teleop_action.get(f"{motor}.pos", 0) for motor in self.motor_names])
                follower_pos = np.array([raw_joint_pos[motor] for motor in self.motor_names])

                # Compute EE positions
                leader_ee = self.kinematics.forward_kinematics(leader_pos)[:3, 3]
                follower_ee = self.kinematics.forward_kinematics(follower_pos)[:3, 3]
                leader_motion_m = (
                    np.inf
                    if self._last_leader_ee is None
                    else float(np.linalg.norm(leader_ee - self._last_leader_ee))
                )
                target_error = leader_ee - follower_ee

                # Compute normalized EE delta
                if self.end_effector_step_sizes is not None:
                    ee_delta = np.clip(
                        leader_ee - follower_ee,
                        -self.end_effector_step_sizes,
                        self.end_effector_step_sizes,
                    )
                    ee_delta_normalized = ee_delta / self.end_effector_step_sizes
                else:
                    ee_delta_normalized = leader_ee - follower_ee

                # Handle gripper
                if self.use_gripper and len(leader_pos) > 3:
                    if self.prev_leader_gripper is None:
                        self.prev_leader_gripper = np.clip(leader_pos[-1], 0, self.max_gripper_pos)

                    leader_gripper = leader_pos[-1]
                    gripper_delta = leader_gripper - self.prev_leader_gripper
                    normalized_delta = gripper_delta / self.max_gripper_pos

                    # Quantize gripper action. A locked gripper always sends
                    # 1 ("stay") while preserving the 4-D action layout.
                    if self.lock_gripper:
                        gripper_action = 1
                    elif normalized_delta >= 0.3:
                        gripper_action = 2
                    elif normalized_delta <= -0.1:
                        gripper_action = 0
                    else:
                        gripper_action = 1

                    self.prev_leader_gripper = leader_gripper

                    # Create intervention action
                    intervention_action = np.append(ee_delta_normalized, gripper_action)
                else:
                    intervention_action = ee_delta_normalized

                now = time.monotonic()
                if now - self._last_log_time >= 0.5:
                    print(
                        "[主臂证据] "
                        f"主臂移动毫米={leader_motion_m * 1000:.3f} "
                        f"末端误差厘米={np.round(target_error * 100, 2).tolist()} "
                        f"输出动作={np.round(intervention_action, 3).tolist()}",
                        flush=True,
                    )
                    self._last_log_time = now
                self._last_leader_ee = leader_ee.copy()

                # Override teleop_action with computed EE action
                complementary["teleop_action"] = torch.from_numpy(intervention_action).float()
                transition[TransitionKey.COMPLEMENTARY_DATA] = complementary  # type: ignore[misc]

        return transition

    def reset(self) -> None:
        """Reset leader-follower state."""
        self.prev_leader_gripper = None
        self._last_log_time = 0.0
        self._last_leader_ee = None
        if hasattr(self.leader_device, "reset"):
            self.leader_device.reset()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
