# P1 V0 可视化结果

> 日期：2026-10-15
> 状态：PASS
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，物理 GPU 3
> 动态判定：同一 Isaac 进程内局部 reset；固定脚本回放；渲染开启，不计吞吐性能

## 范围

按冻结的 P1 benchmark protocol，以 seed 42 在 Lunalab / Terrain_lidar01 中回放直行、
原地左转、撞击固定 `rock4` 和到达目标四条轨迹。环境使用完整 Task-v0 runtime、
Observation-v0、Reward-v0、termination 与具名 contact view；四个 case 复用同一 World，
case 间只执行局部 reset。

所有 action 恰好保持 6 个 physics step；物理步长为 `1/30 s`，policy 步长为 `0.2 s`。
相机只用于 V0 人工评估，没有进入 observation 或吞吐计时。

## 复现

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_v0_visual.py --gpu 3 --seed 42 \
  --output-dir data/benchmarks/p1/20261015_v0_gpu3_r2
```

通过产物：`data/benchmarks/p1/20261015_v0_gpu3_r2/`

- `summary.json`、`policy_steps.csv`、`transitions.csv`、`contact_samples.csv`；
- `resolved_config.yaml`、`command.txt`、`manifest.json`；
- `run.log`、`exit_code.txt`、`checksums.sha256`；
- `visual/trajectory.png`、`visual/controls_rewards.png`；
- `visual/straight.png`、`visual/turn.png`、`visual/collision.png`、
  `visual/success.png` 及四张原始场景截图；
- `git_status.txt`、`git_diff.patch`、`host_preflight.txt`、
  `environment_versions.txt`。

进程退出码为 0，manifest 为 `valid`，全部产物 SHA-256 校验通过。

## 动态结果

| case | policy step | 位移 XY [m] | yaw 变化 | 终止结果 | 末步 reward |
|---|---:|---:|---:|---|---:|
| straight | 12 | `[+1.285738, -0.034493]` | `+0.057552 rad` | 非终止 | `+0.588045` |
| turn | 20 | `[-0.177603, +0.058550]` | `+2.363015 rad`（`+135.391°`） | 非终止 | `+0.016866` |
| collision | 6 | `[+0.443341, +0.001038]` | `+0.030138 rad` | `terminated=True`，`collision` | `-9.998485` |
| success | 27 | `[+2.107892, -0.057555]` | `+0.006277 rad` | `terminated=True`，`success` | `+10.460565` |

四个 case 的 articulation invalid count 均为 0，且
`physics_steps_this_action == 6` 全部成立。撞岩 case 的 rock-filtered chassis 峰值为
`64.590796 N`，全来源 chassis 峰值为 `394.384974 N`；其余三个 case 的 chassis
峰值均为 0。成功 case 在目标距离 `0.421261 m` 时触发，与冻结的 `0.5 m` 成功半径
一致。

`trajectory.png` 同时标出四条 root 轨迹、起点、终点、目标、目标朝向、固定岩石和
termination reason。`controls_rewards.png` 使用统一拼接的仿真时间轴，绘制归一化
action、实测 `v/omega`、目标距离和各 Reward-v0 分量。四张 case 图把 Lunalab 场景
截图与对应俯视轨迹并排展示；人工检查可见 Husky、Terrain_lidar01 和撞击用
`rock4`。

## 首轮失败与修正

首轮产物保存在 `data/benchmarks/p1/20261015_v0_gpu3/`，退出码为 1。第一张截图由
单独创建的相机返回全黑帧，脚本按有效性检查中止。修正为复用已在独立撞岩探针验证
过的相机创建、focus 与渲染预热路径；没有修改动作、动力学、Task-v0、Reward-v0、
termination 或接触阈值。首轮失败产物继续保留。

## Gate 边界

冻结协议要求的 V0 直行、转向、碰撞、成功四条可视化轨迹及控制/奖励曲线已完成。
此前通过的 L0/C0、Task-v0 runtime、正式 T0 和独立撞岩保持有效。正式 B0/B1、R0、
D0、S0、B2/B3 及吞吐/资源测量仍未执行，因此完整 P1 Gate 仍为 `IN PROGRESS`，
尚不能作训练路线的最终 Go/No-Go。

本次证据没有否定冻结的轮速符号、轮径、轮距、Action-v0 时序、成功半径或接触
阈值，`task_v0_spec.md` 无需修订。
