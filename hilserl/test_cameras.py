#!/usr/bin/env python

"""Test the two SO101 cameras simultaneously without touching the robot."""

from __future__ import annotations

import argparse
import time

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


TOP = (
    "/dev/v4l/by-id/"
    "usb-2MP_USB_Camera_2MP_USB_Camera_04434000_P120800_SN0002-video-index0"
)
WRIST = (
    "/dev/v4l/by-id/"
    "usb-JoyandAI_JYU2C-2083_JYU2C-2083-2609096-video-index0"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    cameras = {}
    try:
        for name, path in {"top": TOP, "wrist": WRIST}.items():
            config = OpenCVCameraConfig(
                index_or_path=path,
                width=640,
                height=480,
                fps=args.fps,
                fourcc=args.fourcc,
                warmup_s=1,
            )
            camera = OpenCVCamera(config)
            camera.connect()
            cameras[name] = camera

        counts = dict.fromkeys(cameras, 0)
        end_time = time.perf_counter() + args.seconds
        while time.perf_counter() < end_time:
            for name, camera in cameras.items():
                camera.read_latest()
                counts[name] += 1
            time.sleep(1 / args.fps)

        print(f"fourcc={args.fourcc} fps={args.fps} counts={counts}")
    finally:
        for camera in cameras.values():
            camera.disconnect()


if __name__ == "__main__":
    main()
