# SO101 HIL-SERL push-cube setup

All paths below assume:

```bash
conda activate hilserl
cd /home/husky/Desktop/lerobot-beta
```

## Devices

- Follower: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79015851-if00`
- Leader: `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116379-if00`
- Top camera: `/dev/video2`
- Wrist camera: `/dev/video0`
- Calibration IDs: `follower_arm` and `leader_arm`

If `Permission denied` is reported for the serial devices, run:

```bash
sudo chmod 666 /dev/ttyACM*
```

## 1. Check cameras

```bash
lerobot-find-cameras opencv --output-dir rl/camera_probe
```

Open the generated images and verify that the top view and wrist view are assigned
to the expected indices.

## 2. Test leader/follower teleoperation

Keep a clear workspace around both arms. This command uses normal direct leader
teleoperation, not HIL-SERL intervention mode:

```bash
lerobot-teleoperate \
  --config_path rl/teleoperate_so101_preflight.json
```

When prompted about an existing calibration, press Enter to use it.

## 3. Record demonstrations

```bash
python -m lerobot.rl.gym_manipulator \
  --config_path rl/env_config_so101_push_record.json
```

Initially the leader follows the follower and the follower stays still. Press Space
once to take over with the leader. The terminal prints
`Intervention ON - leader controls follower`. Press Space again to hand control
back; the terminal prints `Intervention OFF - leader follows follower`, and the
leader is intentionally torque-enabled so it cannot be manually moved in this mode.
Press `s` when the cube reaches the goal, Esc to mark failure, or `r` to rerecord
the episode.

Follower motion during takeover is intentionally speed-limited to
`end_effector_step_sizes` (about 1 cm per control step). A fast leader movement
therefore looks like the follower catching up over several frames rather than
teleporting to the leader pose.

The terminal now announces every reset:

```text
[EPISODE 1/10] END: success, frames=..., reward=1.0
[RESET] Next episode 2/10 will reset automatically.
[RESET] Starting automatic reset in 3.0s.
[RESET] Moving follower to the configured reset pose. Keep hands clear.
[RESET] Follower motion complete. Settling at the reset pose.
[READY] Robot is at the reset pose. Place the cube, then press Space to take over.
```

Do not reach into the workspace until `[READY]` is printed. The robot waits five
seconds at the reset pose before the next episode is available.

The leader torque is released before the follower reset starts:

```text
[LEADER] Torque released. You may manually reposition the leader during reset.
```

During this time the leader can be moved by hand. When following resumes it ramps
at no more than `leader_follow_max_step_deg` (2 degrees) per control step instead
of jumping back to the follower.

After the requested episodes finish, wait for:

```text
[DATASET] Finalizing parquet, videos, and episode metadata...
[DATASET] Saved N episodes, M frames at ...
```

Only then is the dataset valid for training. On an intentional `Ctrl+C`, the
current unfinished episode is discarded and all previously saved episodes are
finalized. Do not terminate the terminal with a forced kill while recording.

The two arms are mounted in different physical locations, so their absolute
end-effector positions are not expected to match. At takeover the controller
latches both starting end-effector poses and applies only relative leader motion
to the follower. This remains a 4D end-effector action and is executed through IK,
matching the policy action space.

Record 10 consistent successful episodes first. Each episode has a 60 second limit,
but pressing `s` ends a successful episode immediately.

`display_cameras` opens a live Rerun window with separate `top` and `wrist` views.
The OpenCV package in this environment is headless and cannot open `cv2.imshow`
windows, so Rerun is the live preview backend.
The dataset is saved under `data/so101_push_cube`.

## 4. Train

Start the learner first:

```bash
read -s -p "SwanLab API key: " SWANLAB_API_KEY
echo
export SWANLAB_API_KEY

python -m lerobot.rl.learner \
  --config_path rl/train_config_hilserl_so101_push.json
```

SwanLab is configured in cloud-only `online` mode. The API key is read from
`SWANLAB_API_KEY`, is not stored in the JSON file, and is not written to the
dataset. The learner prints the cloud run URL after initialization.

Start the actor in a second terminal:

```bash
conda activate hilserl
cd /home/husky/Desktop/lerobot-beta
python -m lerobot.rl.actor \
  --config_path rl/train_config_hilserl_so101_push.json
```

During online training, keep the same Space/s/Esc/r controls. The first run is set
to 30,000 actor steps, batch size 64, CPU replay storage, and no W&B account.

Use `lerobot-find-joint-limits` to refine workspace bounds after confirming that
teleoperation works. Update the bounds in both JSON configs before the main run.
