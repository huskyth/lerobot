#!/usr/bin/env python

"""Check the software prerequisites for the SO101 HiL-SERL workflow."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

import torch
import draccus

from lerobot.rl.gym_manipulator import GymManipulatorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "hilserl" / "configs" / "so101_leader_env.json"
URDF_PATH = PROJECT_ROOT / "SO101" / "so101_new_calib.urdf"


def main() -> None:
    failures: list[str] = []

    if sys.version_info < (3, 12):
        failures.append(f"Python >= 3.12 is required, got {sys.version.split()[0]}")
    print(f"Python: {sys.version.split()[0]}")

    packages = {
        "lerobot": "lerobot",
        "gym-hil": "gym_hil",
        "placo": "placo",
        "grpcio": "grpc",
        "feetech-servo-sdk": "scservo_sdk",
        "pyserial": "serial",
        "transformers": "transformers",
    }
    for distribution, module in packages.items():
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(distribution)
            print(f"{distribution}: {version}")
        except Exception as exc:
            failures.append(f"{distribution}: {exc}")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        failures.append("CUDA is not available to PyTorch")

    if not CONFIG_PATH.is_file():
        failures.append(f"Missing config: {CONFIG_PATH}")
    else:
        try:
            draccus.parse(GymManipulatorConfig, config_path=str(CONFIG_PATH), args=[])
            print(f"Config schema: OK ({CONFIG_PATH})")
        except Exception as exc:
            failures.append(f"Invalid config {CONFIG_PATH}: {exc}")
    if not URDF_PATH.is_file():
        failures.append(f"Missing URDF: {URDF_PATH}")

    if failures:
        print("\nBlocking issues:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nHiL-SERL software prerequisites are ready.")
    print("Hardware ports and cameras are intentionally not probed by this check.")


if __name__ == "__main__":
    main()
