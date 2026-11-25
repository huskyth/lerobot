#!/usr/bin/env python
"""Interactively detect the SO101 leader port and update both RL configs."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIGS = [ROOT / "rl" / "train_config.json", ROOT / "rl" / "env_config.json"]
BY_ID = Path("/dev/serial/by-id")
FOLLOWER_SERIAL = "5B79015851"


def ports() -> set[Path]:
    return set(BY_ID.glob("usb-1a86_USB_Single_Serial_*-if00"))


def port_processes(port: Path) -> str:
    try:
        result = subprocess.run(
            ["fuser", "-v", str(port)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return ""
    return (result.stderr + result.stdout).strip()


def update_config(path: Path, leader_port: Path, follower_port: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    config["env"]["robot"]["port"] = str(follower_port)
    config["env"]["teleop"]["port"] = str(leader_port)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> None:
    if not BY_ID.exists():
        raise SystemExit(f"{BY_ID} does not exist; no USB serial devices are visible")

    before = ports()
    print("Detected serial ports:")
    for port in sorted(before):
        print(f"  {port}")

    input("\n拔掉主臂 USB，不要拔从臂；拔下后按 Enter：")
    time.sleep(1.0)
    after = ports()
    removed = sorted(before - after)
    if len(removed) != 1:
        raise SystemExit(
            "无法唯一识别被拔掉的主臂端口。"
            f"\n拔线前: {sorted(map(str, before))}"
            f"\n拔线后: {sorted(map(str, after))}"
        )

    leader_port = removed[0]
    follower_candidates = [port for port in after if FOLLOWER_SERIAL in port.name]
    if len(follower_candidates) != 1:
        raise SystemExit(
            f"无法识别从臂端口（期望包含 {FOLLOWER_SERIAL}）。当前端口："
            f"{sorted(map(str, after))}"
        )
    follower_port = follower_candidates[0]

    print(f"\n识别到主臂: {leader_port}")
    print(f"识别到从臂: {follower_port}")
    input("\n重新插回主臂 USB；连接稳定后按 Enter：")
    time.sleep(1.5)

    if leader_port not in ports():
        raise SystemExit(f"主臂重新插回后没有出现：{leader_port}")

    holder = port_processes(leader_port)
    if holder:
        raise SystemExit(
            f"主臂端口仍被占用：\n{holder}\n"
            "请关闭对应程序后重新运行本脚本。"
        )

    for config_path in CONFIGS:
        update_config(config_path, leader_port, follower_port)
        print(f"Updated {config_path.relative_to(ROOT)}")

    print("\n配置完成。端口空闲，可以重新启动 learner 和 actor。")


if __name__ == "__main__":
    main()
