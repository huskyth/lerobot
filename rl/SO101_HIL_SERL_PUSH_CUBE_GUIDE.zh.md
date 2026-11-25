# SO-101 推箱子 HIL-SERL 实战教程与心得

本文基于 LeRobot v0.7.0，官方基线 commit：

```text
aa7ecdd7e8d7e94310f1b235483d55ed45b77d15
```

本地分支为 `so101-hilserl`，截至整理时最新提交为：

```text
ace5a81e 配置优化
```

这套代码针对一个具体任务：SO-101 follower 在固定桌面、顶部相机和腕部相机下，把方块从不同初始位置推到固定目标区域。夹爪保持闭合，策略只学习末端 XYZ 增量。

本文不是通用机器人学习理论介绍，而是一份从真实实验、故障和参数调整中整理出来的操作教程。

## 当前快照

整理本文时：

```text
最新 checkpoint：425000
在线 replay：20000 帧，103 episodes
离线 replay：5638 帧，36 episodes
训练频率：10 Hz
```

最新模型路径：

```text
outputs/hilserl/so101_push_cube_hardcube15_20261112/checkpoints/425000
```

本地提交从官方基线之后依次为：

| Commit | 主题 | 主要内容 |
|---|---|---|
| `aa7ecdd7` | 官方基线 | LeRobot v0.7.0 |
| `c8295bc4` | 暂时可以 | 首版训练、安全 Z、数据修复和运行脚本 |
| `bdf8ca06` | 限制低限 | 收紧安全检查、端口配置、buffer 恢复优化 |
| `9378440d` | 推理 | 独立推理脚本 |
| `7f66c4b5` | 修复 | 推理退出与运行细节修正 |
| `ccaafb35` | 推理修正 | 推理专用主臂和回位行为 |
| `7ed9fd47` | 修正 weight | IK 姿态权重和安全检查日志 |
| `ace5a81e` | 配置优化 | 最终工作空间参数 |

## 1. 最终系统结构

HIL-SERL 使用 actor-learner 分布式结构：

```text
actor（真实机械臂） ← gRPC → learner（GPU 训练）
```

actor 负责：

- 读取顶部和腕部相机
- 读取六个关节状态
- 执行策略的 XYZ 动作
- 支持人工接管
- 把 transition 发送给 learner

learner 负责：

- 维护在线 replay buffer
- 加载离线人工干预数据
- 执行 SAC 更新
- 每两秒向 actor 推送新策略参数
- 每 5000 个优化步保存 checkpoint

### 1.1 策略输入和输出

当前策略输入：

```text
observation.images.top    [3, 128, 128]
observation.images.wrist  [3, 128, 128]
observation.state         [6]
```

模型输出：

```text
连续动作 [3]：delta_x, delta_y, delta_z
离散夹爪类别 [3]：0=闭合, 1=保持, 2=张开
```

当前推箱任务不需要夹爪开合，因此 actor 会把离散夹爪动作强制设为 `1`，保持夹爪当前位置。现有四维动作布局和 checkpoint 仍保持兼容。

### 1.2 当前策略规模

```text
总参数：约 5.8M
可训练参数：约 891K
视觉编码器：lerobot/resnet10，冻结
latent_dim：64
actor hidden dims：[256, 256]
```

相机图像由冻结视觉编码器提取特征，actor、critic 和 discrete critic 训练策略。

## 2. 硬件与端口准备

当前硬件是 SO-101 leader + SO-101 follower、两部 USB 相机和一条末端总线。

### 2.1 固定串口

不要长期依赖 `/dev/ttyACM0` 或 `/dev/ttyACM1`，因为重新插拔后编号可能改变。使用设备序列号：

```bash
ls -l /dev/serial/by-id/
```

本项目使用的端口：

```text
follower:
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79015851-if00

leader:
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116379-if00
```

如果端口变化，可运行：

```bash
./rl/configure_ports.py
```

脚本会要求拔掉主臂 USB，自动识别主臂和从臂端口，并更新：

```text
rl/train_config.json
rl/env_config.json
```

注意：拔的是主臂，不是从臂或相机。

### 2.2 普通遥操作自检

在做强化学习前，先确认主臂可以稳定带动从臂：

```bash
cd /home/husky/Desktop/lerobot

/home/husky/miniconda3/envs/hilserl/bin/lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B79015851-if00 \
  --robot.id=follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E116379-if00 \
  --teleop.id=leader_arm \
  --fps=30 \
  --teleop_time_s=10
```

普通遥操作是直接关节映射，不经过 XYZ 逆运动学。因此它可以帮助区分：

- 机械臂硬件、校准和串口是否正常
- XYZ IK 是否在远端产生误差

如果普通遥操作稳定，但 HIL-SERL 在远端动作不准，优先检查 IK 和末端工作空间。

## 3. 当前配置

配置文件：

```text
rl/train_config.json
rl/env_config.json
```

训练数据：

```text
repo_id: local/so101_push_cube_hardcube15_20261112
root: data/so101_push_cube_hardcube15_20261112
```

### 3.1 核心训练参数

```text
算法：SAC
batch_size：64
discount：0.97
actor_lr：3e-4
critic_lr：3e-4
temperature_lr：3e-4
utd_ratio：1
fps：10
online_ratio：0.5
online_buffer_capacity：20000
offline_buffer_capacity：10000
```

在线和离线数据的混合比例是 50/50。

`train_config.json` 顶层的 `steps=145000` 不是当前 HIL-SERL learner 的主要停止条件。真机实验已经超过该值仍在继续，实际训练以 actor 的 `online_steps`、手动停止和 checkpoint 为准。

### 3.2 当前末端工作空间

```text
X: 139.672 ～ 326.500 mm
Y: -30.150 ～ +83.210 mm
Z:    6.718 ～ +43.000 mm
```

当前复位关节：

```json
[
  -1.099002,
  -95.947376,
  87.729507,
  76.172869,
  -9.978002,
  3.665
]
```

通过 URDF 计算得到的复位末端位置约为：

```text
X = 139.674 mm
Y =   3.291 mm
Z =   6.718 mm
```

当前动作步长：

```text
X：2 mm
Y：2 mm
Z：3 mm
```

当前方向约定：

```text
+Y：画面左侧
-Y：画面右侧
-X：靠近底座
+X：远离底座、进入工作区
-Z：向下
+Z：向上
```

### 3.3 IK 姿态权重

当前配置：

```json
"orientation_weight": 0.03
```

SO-101 是欠驱动五轴机械臂。姿态权重过高时：

- 夹爪更容易保持水平
- 但左右位置可能根本走不到

姿态权重过低时：

- XYZ 更容易到达
- 但夹爪可能发生较大倾转，尖端碰桌

`0.03` 是当前任务中位置可达性和夹爪姿态之间的折中值。

### 3.4 安全边界日志

当动作超出安全边界时，处理器会输出：

```text
EE target clipped:
requested=[...],
bounded=[...],
bounds=...
```

含义：

- `requested`：策略或人工接管想去的目标
- `bounded`：安全层最终允许执行的目标
- `bounds`：完整工作空间

看到 Z 被裁剪通常表示策略想压向桌面。看到 Y 或 X 被裁剪表示已到达工作空间边界。

边界裁剪是保护机制，不等于程序错误。连续出现“边界 + 一个动作步长”的裁剪，通常表示策略在该方向持续饱和。

## 4. 桌面高度和 Z 安全

Z 下限不能直接照搬 URDF 坐标。要区分：

```text
gripper_frame_link 的 Z
实际最低爪尖的 Z
桌面实际高度
机械柔顺造成的误差
```

### 4.1 只读检查当前位姿

先停止 actor 和 learner：

```bash
pkill -INT -f '[l]erobot.rl.actor' || true
pkill -INT -f '[l]erobot.rl.learner' || true
sleep 3
```

读取当前末端位姿，不发送动作：

```bash
./rl/read_ee_z.py
```

该脚本会检查是否还有 actor/learner 占用串口，并忽略僵尸进程。

### 4.2 手动标定安全 Z

使用：

```bash
./rl/safe_z_control.py --start-current --max-joint-step-deg 1
```

控制键：

```text
W    上升 1 mm
S    下降 1 mm
R    保持当前姿态
Q    保持扭矩并退出
Esc  释放扭矩并退出
```

操作规则：

- 每次只按一下 `S`，等待约一秒
- 观察实际爪尖，不只看参考坐标
- 找到爪尖刚好离桌的高度
- 记录 `EE z`
- 至少测试近端、中段和远端
- 最终下限取所有位置中最高的接触值再加 5～10 mm

如果机械臂已经压住桌面，先释放扭矩：

```bash
./rl/safe_z_control.py --release-torque
```

托住机械臂后再手动抬高，避免直接掉臂。

## 5. 数据采集与 buffer

### 5.1 离线数据

当前实际离线 replay：

```text
5638 帧
36 episodes
```

这些数据主要保存人工接管轨迹，和在线数据按 50/50 混合训练。

### 5.2 在线数据为什么一直是 20000

在线 buffer 配置为：

```json
"online_buffer_capacity": 20000
```

达到容量后，新数据覆盖最旧数据：

```python
position = (position + 1) % capacity
size = min(size + 1, capacity)
```

所以 20000 是“当前保留的最近经验数量”，不是累计样本数，也不是 epoch。

### 5.3 离线 buffer 容量

learner 在恢复数据时使用：

```python
capacity = min(configured_capacity, len(offline_dataset))
```

避免离线数据不足 10000 条时仍预分配完整的 10000 条空间，降低启动内存峰值。

## 6. 采集和人工接管策略

推箱子任务采用固定夹爪，重点是接近方块、对准和推动。

### 6.1 有效接管原则

推荐规则：

```text
策略能自主完成：零接管
纠正一次后能完成：短接管，然后交还
交还后马上又犯同样错误：本次接管持续到完整成功
```

可以在一段连续接管中进行多次小幅方向修正，但尽量不要反复按 Space 进入和退出接管。

反复切换会造成：

```text
策略向左 → 人工向右 → 策略又向左 → 人工又向右
```

相似视觉状态下出现冲突动作，模型更容易振荡。

### 6.2 成功标记

只有方块完整进入目标区域时才按：

```text
s
```

只推进一部分、但还没有到达目标时，不要按 `s`。

安全风险或已经无法恢复时按：

```text
Esc
```

### 6.3 左右泛化训练

固定位置连续训练通常会提高该位置成功率，但容易记住一条固定轨迹。

建议使用分阶段课程：

```text
中间稳定 → 小幅左侧 → 中间验证
→ 小幅右侧 → 中间验证 → 三个位置混合
```

每个新位置先固定训练 3～8 个 episode。每次训练后都要测试其他位置，检查是否遗忘。

不要在模型刚学会中间时立刻完全随机放置方块。那样每个位置的有效样本都很稀疏。

## 7. 启动训练

### 7.1 Learner

第一个终端：

```bash
cd /home/husky/Desktop/lerobot
./rl/run_learner.sh
```

需要看到：

```text
Valid checkpoint found: resume=True
Resuming from step <N>
[LEARNER] gRPC server started
```

`rl/run_learner.sh` 设置了：

```text
LD_LIBRARY_PATH
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### 7.2 Actor

第二个终端：

```bash
./rl/run_actor.sh
```

当前 `run_actor.sh` 会直接找到最新策略：

```text
outputs/hilserl/.../checkpoints/last/pretrained_model
```

并通过：

```text
--policy.pretrained_path
```

加载已有训练权重。这样 actor 第一回合不会再使用随机初始化策略。

启动后应看到：

```text
Loading weights from local directory
[ACTOR] 夹爪已锁定
```

### 7.3 停止顺序

建议：

```text
先停止 actor
再停止 learner
```

看到 checkpoint 正在保存时不要立即强制关闭，因为 replay dataset 可能仍在写 Parquet。

## 8. 独立推理

推理脚本：

```bash
./rl/infer_policy.py --episodes 1 --max-seconds 120
```

该脚本不启动 learner，不更新模型，不创建 replay buffer，也不写训练数据。

### 8.1 推理前检查

脚本会自动阻止同时运行的 actor 和 learner，因为两种程序会争用机械臂串口。

无硬件验证：

```bash
./rl/infer_policy.py --dry-run --device cpu
```

指定其他 checkpoint：

```bash
./rl/infer_policy.py \
  --checkpoint outputs/hilserl/so101_push_cube_hardcube15_20261112/checkpoints/400000/pretrained_model \
  --episodes 3
```

### 8.2 当前推理行为

当前脚本具有以下行为：

- 主臂不持续跟随从臂
- Space 仍可进入和退出接管
- Gaussian actor 使用随机采样动作
- gripper 保持
- 每个 episode 通过环境 reset 回到固定姿态
- 退出时把从臂送回固定关节位置

当前推理不是确定性推理，因此相同初始状态下，轨迹可能不同。策略没有时序记忆：

```text
n_obs_steps = 1
```

它只根据当前画面和关节状态选动作，不知道上一秒的动作历史。这会导致起步阶段前后摆动，尤其是在新复位状态附近。

### 8.3 扭矩注意事项

SOFollower 默认：

```python
disable_torque_on_disconnect = True
```

如果推理退出时没有显式保持扭矩，`env.close()` 会关闭电机扭矩，机械臂可能因重力下垂并碰桌。

需要手动释放扭矩时：

```bash
./rl/safe_z_control.py --release-torque
```

## 9. 这次代码中的关键修复

### 9.1 夹爪锁定

`GripperConfig` 增加：

```json
"lock_gripper": true
```

actor 后处理完成后强制：

```python
action[..., -1] = 1.0
```

主臂接管动作也强制 `gripper=1`。这保证推箱任务不会因为策略随机选择而张开夹爪，同时保持旧 checkpoint 的四维动作兼容性。

### 9.2 Space 按键去抖

`SO101LeaderFollower` 监听 Space 时加入按下和释放状态，避免按住 Space 或系统按键重复导致接管状态快速来回切换。

### 9.3 复位期间主臂释放

每个 episode reset 前关闭主臂扭矩，让操作者调整主臂；reset 完成后如果仍处于接管状态，则保持主臂可动，否则恢复跟随。

### 9.4 actor 不播放声音

训练 actor 调用：

```python
make_robot_env(..., play_sounds=False)
```

原因是 `log_say(play_sounds=True)` 会启动 `subprocess.Popen`。gRPC 接收线程运行时执行 `fork`，曾导致：

```text
Other threads are currently calling into gRPC, skipping fork() handlers
UNREACHABLE CODE
Aborted (core dumped)
```

关闭推理和训练 actor 的语音子进程可以避免这条故障路径。

### 9.5 replay 恢复流式化

原实现先把全部 transition 转成 Python list，再装入 replay buffer，图像数据会出现接近两份的内存峰值。

当前实现逐条生成并写入：

```python
first_transition = next(transitions)
...
for data in transitions:
    replay_buffer.add(...)
```

这降低了恢复 20000 帧在线 buffer 时的内存峰值。

## 10. 常见故障

### 10.1 串口不存在

```text
OSError: No such file or directory
```

处理：

```bash
ls -l /dev/serial/by-id/
```

确认主臂和从臂都枚举成功。优先直接插主板 USB，避免供电不足的 Hub。

### 10.2 串口被占用

```text
OSError: Device or resource busy
```

检查：

```bash
pgrep -af 'lerobot.rl.(actor|learner)'
sudo fuser -v /dev/serial/by-id/<device>
```

先正常退出旧 actor。卡死时可结束具体 PID，不要使用宽泛的递归删除或终止系统中的其他 Python 进程。

### 10.3 learner 被 systemd-oomd 杀死

日志：

```text
systemd-oomd killed ... due to memory pressure
```

检查：

```bash
journalctl --since '-10 minutes' --no-pager | rg -i 'oom|killed process|oomd'
free -h
```

减少同时运行的程序，避免在恢复 20000 帧图像 buffer 时保留大量浏览器页面。

### 10.4 Parquet 尾部损坏

典型错误：

```text
Parquet magic bytes not found in footer
```

这通常是进程在重写 replay dataset 时被强行终止。

先做只读检查：

```bash
./rl/repair_replay_dataset.py \
  outputs/hilserl/so101_push_cube_hardcube15_20261112/dataset
```

验证通过后写回：

```bash
./rl/repair_replay_dataset.py \
  outputs/hilserl/so101_push_cube_hardcube15_20261112/dataset \
  --apply
```

脚本会：

1. 保留最长的连续完好分片
2. 从存活数据重建 episode metadata
3. 在临时目录构造并加载验证
4. 将损坏文件移动到时间戳备份目录

损坏的最后一个分片无法恢复。模型 checkpoint 和较早的 replay 分片仍可保留。

## 11. 实验结果与经验

这不是最终 benchmark，只记录当前真机实验观察。

- 中间位置最早出现稳定成功。
- 放宽右侧负 Y 和调整 IK 姿态后，右侧开始出现成功。
- 左侧需要额外训练，位置改变后成功率明显下降。
- 相同画面下反复接管和交还会增加动作振荡。
- 固定位置更容易训练，但容易记住一条固定路线。
- 移动位置混合训练更接近泛化，但每个位置需要足够的成功数据。

因此训练目标应分阶段验证：

```text
固定中间成功
固定左侧成功
固定右侧成功
小范围位置变化成功
三个位置混合成功
```

## 12. 评估方法

不要只看训练过程中的成功事件，因为训练包含人工接管和持续策略更新。

独立推理时至少测试：

```text
中间 5 次
左侧 5 次
右侧 5 次
```

记录：

- 是否无接管完整成功
- 从开始到第一次接触方块的步数
- 是否在初始位置长时间徘徊
- 是否前后摆动
- 是否发生边界裁剪
- 完整任务耗时

如果训练 `20～30` 个定向 episode 后指标仍无改善，应优先补充 5～10 条干净演示，而不是无限增加在线训练。

## 13. 快速运行清单

### 训练

```bash
cd /home/husky/Desktop/lerobot
./rl/run_learner.sh
```

等待：

```text
Resuming from step <N>
[LEARNER] gRPC server started
```

然后：

```bash
./rl/run_actor.sh
```

### 推理

先停止 learner 和 actor：

```bash
./rl/infer_policy.py --episodes 1 --max-seconds 120
```

### 位姿检查

```bash
./rl/read_ee_z.py
```

### Z 安全标定

```bash
./rl/safe_z_control.py --start-current --max-joint-step-deg 1
```

### 释放扭矩

```bash
./rl/safe_z_control.py --release-torque
```

### 数据修复

```bash
./rl/repair_replay_dataset.py \
  outputs/hilserl/so101_push_cube_hardcube15_20261112/dataset
```

确认无误后加 `--apply`。

## 14. 最重要的一条经验

强化学习不是改的参数越多越好。

改变以下任何一项，都会改变同一策略动作的物理含义或训练分布：

- 末端 XYZ 边界
- IK 姿态权重
- 复位姿态
- 末端动作步长
- 相机位置或裁剪
- 夹爪动作空间
- 标定和电机增益
- 成功判定和 episode 长度

因此应先把机械臂、相机、安全边界和复位姿态确定下来，再持续训练和评估。每次改变动作映射后，都需要一段时间让策略重新适应。

固定配置、分阶段训练、按位置独立评估，是这套真机 HIL-SERL 流程最关键的实践原则。
