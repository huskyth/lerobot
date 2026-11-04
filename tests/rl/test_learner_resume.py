#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest

pytest.importorskip("datasets", reason="datasets is required")

from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig  # noqa: E402
from lerobot.rl.learner import handle_resume_logic  # noqa: E402
from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
)


def test_resume_keeps_current_runtime_configuration(tmp_path):
    checkpoint_dir = tmp_path / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK
    pretrained_model_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_model_dir.mkdir(parents=True)
    (pretrained_model_dir / "train_config.json").write_text("{}")

    cfg = TrainRLServerPipelineConfig(output_dir=tmp_path, resume=True)
    cfg.policy = GaussianActorConfig()
    cfg.policy.policy_kwargs.std_min = 0.01
    cfg.policy.policy_kwargs.std_max = 0.25

    resumed_cfg = handle_resume_logic(cfg)

    assert resumed_cfg is cfg
    assert resumed_cfg.resume is True
    assert resumed_cfg.policy.policy_kwargs.std_min == pytest.approx(0.01)
    assert resumed_cfg.policy.policy_kwargs.std_max == pytest.approx(0.25)
