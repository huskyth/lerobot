#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.model import RobotKinematics
from lerobot.processor.pipeline import EnvTransition, ProcessorStep, ProcessorStepRegistry, TransitionKey
from lerobot.teleoperators import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents


@ProcessorStepRegistry.register("leader_follower_processor")
@dataclass
class LeaderFollowerProcessor(ProcessorStep):
    """Make the leader track the follower and convert a takeover pose into EE deltas."""

    leader_device: Teleoperator
    motor_names: list[str]
    kinematics: RobotKinematics
    end_effector_step_sizes: np.ndarray
    use_gripper: bool = True
    prev_leader_gripper: float | None = None
    max_gripper_pos: float = 100.0
    position_deadband_m: float = 0.002
    leader_reference_ee: np.ndarray | None = None
    follower_reference_ee: np.ndarray | None = None
    was_intervening: bool = False

    @staticmethod
    def _joint_value(joints: dict[str, float], motor: str) -> float:
        return float(joints.get(f"{motor}.pos", joints.get(motor, 0.0)))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        raw_joints = complementary.get("raw_joint_positions") or {}
        teleop_action = complementary.get("teleop_action") or {}
        info = new_transition.get(TransitionKey.INFO) or {}
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))

        if raw_joints:
            follower_action = {
                f"{motor}.pos": self._joint_value(raw_joints, motor) for motor in self.motor_names
            }
            self.leader_device.send_action(follower_action)

        leader_pos = np.array(
            [self._joint_value(teleop_action, motor) for motor in self.motor_names], dtype=np.float64
        )
        if is_intervention and raw_joints:
            leader_ee = self.kinematics.forward_kinematics(leader_pos)[:3, 3]
            follower_pos = np.array(
                [self._joint_value(raw_joints, motor) for motor in self.motor_names],
                dtype=np.float64,
            )
            follower_ee = self.kinematics.forward_kinematics(follower_pos)[:3, 3]
            if not self.was_intervening or self.leader_reference_ee is None:
                self.leader_reference_ee = leader_ee.copy()
                self.follower_reference_ee = follower_ee.copy()
                self.prev_leader_gripper = float(leader_pos[-1])

            target_follower_ee = self.follower_reference_ee + (
                leader_ee - self.leader_reference_ee
            )
            leader_delta = target_follower_ee - follower_ee
            if np.linalg.norm(leader_delta) <= self.position_deadband_m:
                leader_delta = np.zeros(3, dtype=np.float64)
            ee_delta = np.clip(
                leader_delta,
                -self.end_effector_step_sizes,
                self.end_effector_step_sizes,
            )
            ee_delta_normalized = ee_delta / self.end_effector_step_sizes

            gripper_action = 1.0
            leader_gripper = float(leader_pos[-1])
            if self.use_gripper:
                if self.prev_leader_gripper is not None:
                    normalized_delta = (leader_gripper - self.prev_leader_gripper) / self.max_gripper_pos
                    # SO101 gripper position increases when closing.
                    if normalized_delta >= 0.1:
                        gripper_action = 0.0
                    elif normalized_delta <= -0.1:
                        gripper_action = 2.0
                self.prev_leader_gripper = leader_gripper
            policy_action = np.concatenate([ee_delta_normalized, [gripper_action]]).astype(np.float32)
            leader_hold = bool(np.linalg.norm(policy_action[:3]) < 1e-6)
        else:
            self.leader_reference_ee = None
            self.follower_reference_ee = None
            self.prev_leader_gripper = float(leader_pos[-1])
            policy_action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            if not self.use_gripper:
                policy_action = policy_action[:3]
            leader_hold = False

        complementary["teleop_action"] = torch.from_numpy(policy_action)
        complementary["leader_hold"] = leader_hold
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        self.was_intervening = is_intervention
        return new_transition

    def reset(self) -> None:
        self.prev_leader_gripper = None
        self.leader_reference_ee = None
        self.follower_reference_ee = None
        self.was_intervening = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
