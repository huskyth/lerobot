# SO101 HiL-SERL

This directory contains the local starting point for the LeRobot v0.6.2 HiL-SERL workflow.
The checked-in repository is version 0.6.2 and is newer than the linked v0.6.1 documentation.

## Installed environment

Use the existing Conda environment:

```bash
conda activate lerobot
cd /home/husky/Desktop/lerobot
```

The environment was installed with:

```bash
pip install -e ".[hilserl,core_scripts,feetech,gamepad]"
```

This includes HiL-SERL, Feetech/SO101 serial control, gamepad support, dataset tools,
WandB, and the kinematic solver.

## Before connecting the robot

1. Add your user to `dialout`, then sign out and back in. This step still needs your sudo
   password and was not applied automatically:

```bash
sudo usermod -aG dialout husky
```

2. Adjust ports, camera paths, robot IDs, task text, and dataset name in
   `hilserl/configs/so101_leader_env.json`.
3. Keep the follower powered off while checking `/dev/ttyACM*` and `/dev/video*`.
4. Calibrate both arms before recording:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm

lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm
```

5. Run teleoperation at low speed before enabling HIL-SERL:

```bash
lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm
```

## Record demonstrations

Review every hardware setting first. The reset pose is intentionally unset until it has been
validated on the physical arm.

```bash
python -m lerobot.rl.gym_manipulator \
  --config_path=hilserl/configs/so101_leader_env.json
```

The current robot IDs match the existing calibration files. If the calibration files are
recreated under different IDs, update both `robot.id` and `teleop.id` in the JSON config.

## Software smoke test

This does not access the robot, serial ports, or cameras:

```bash
python hilserl/check_env.py
```
