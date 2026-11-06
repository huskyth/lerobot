#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
import os
from numbers import Number
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig


def _swanlab_config(cfg: TrainPipelineConfig) -> dict:
    return {
        "job_name": cfg.job_name,
        "output_dir": str(cfg.output_dir),
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "save_freq": cfg.save_freq,
        "dataset_repo_id": cfg.dataset.repo_id if cfg.dataset is not None else None,
        "env_type": cfg.env.type if cfg.env is not None else None,
        "env_task": cfg.env.task if cfg.env is not None else None,
        "env_fps": cfg.env.fps if cfg.env is not None else None,
        "policy_type": cfg.policy.type if cfg.policy is not None else None,
        "policy_device": cfg.policy.device if cfg.policy is not None else None,
        "online_steps": cfg.policy.online_steps if cfg.policy is not None else None,
        "algorithm_type": cfg.algorithm.type if cfg.algorithm is not None else None,
    }


class SwanLabLogger:
    """Log HIL-SERL actor and learner metrics to SwanLab Cloud."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.swanlab
        if self.cfg.mode != "online":
            raise ValueError(
                "SwanLab is configured for cloud-only logging; set swanlab.mode='online'."
            )

        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"SwanLab API key is missing. Export it first, for example: "
                f"export {self.cfg.api_key_env}=your_api_key"
            )

        import swanlab

        if not swanlab.login(api_key=api_key, host=self.cfg.host, save=False):
            raise RuntimeError("SwanLab login failed. Check the API key and network connection.")

        self._swanlab = swanlab
        self._run = swanlab.init(
            reinit=True,
            mode="online",
            log_dir=str(Path(cfg.output_dir) / "swanlab"),
            workspace=self.cfg.workspace,
            project=self.cfg.project,
            public=self.cfg.public,
            name=self.cfg.experiment_name or cfg.job_name,
            description=self.cfg.description,
            job_type="train",
            group=cfg.job_name,
            tags=self.cfg.tags or None,
            id=self.cfg.run_id,
            resume=self.cfg.resume,
            config=_swanlab_config(cfg),
        )
        self._step = 0
        logging.info("Logs will be synced with SwanLab Cloud.")
        logging.info("Track this run --> %s", self._run.url)

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        payload = {}
        for key, value in d.items():
            if isinstance(value, Number):
                value = value.item() if hasattr(value, "item") else value
            elif not isinstance(value, str):
                logging.debug("SwanLab ignored non-scalar metric %s=%r", key, value)
                continue
            payload[f"{mode}/{key}"] = value

        if not payload:
            return
        self._step += 1
        self._swanlab.log(payload, step=self._step)

    def finish(self) -> None:
        self._swanlab.finish()
