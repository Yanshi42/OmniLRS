# P1 L0/C0 OmniLRS-native 动态证据

> 日期：2026-10-14
> 状态：L0/C0 功能切片通过；P1 Gate 尚未完成
> 环境：Pixi `default`、Isaac Sim 6.1.0.0、Gymnasium 1.4.0
> 开发运行卡：物理 GPU 6（GPU 3 被外部任务占用，因此本页不报告正式吞吐）

## 范围

本窗口只验证单车 native Gymnasium 生命周期、Husky adapter、局部 reset、
Action-v0 时序，以及 Lunalab/Terrain_lidar01 中的直行、左转和急停。运行内环直接
调用 Python/Isaac Sim API；没有启动 ROS 2、Yamcs、SDG manager 或 PPO/skrl。
场景构建复用了 Lunalab controller 和 `SimulatorMode.SDG` 枚举以关闭 ROS 路径，
但没有实例化或调用 SDG manager。

## 实现边界

- `OmniLRSEnv` 提供 Gymnasium `reset/step`，当前返回明确标记为
  `DiagnosticObservation-P1` 的 11 维诊断观测，不冒充完整 Observation-v0。
- 环境只调用车辆级 `set_velocity(v, omega)`；关节名、左右分组、关节方向、轮径、
  轮距、轮速限幅和允许接触 link 全部位于配置。
- adapter 遍历配置中的所有驱动关节；单元测试还覆盖了六轮配置。
- 局部 reset 顺序为：急停写零、所有驱动关节状态清零、teleport root、root 线/角
  速度清零、控制/接触/episode 历史清零、有限物理 settle；不调用 episode 级
  `World.reset()`。
- Action-v0 保持 `dt=1/30 s`、decimation 6、策略周期 0.2 s、仿真时间 slew limit，
  reset/非法 action/terminal 立即写零。

## 可复现命令

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python -m pytest -q test/rl

env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python scripts/rl/smoke_env.py --case l0 --gpu 6 --seed 42 \
  --output-dir data/benchmarks/p1/20261014_l0_dev_gpu6_recorded

env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python scripts/rl/smoke_env.py --case c0 --gpu 6 --seed 42 --capture-scene \
  --output-dir data/benchmarks/p1/20261014_c0_dev_gpu6_recorded
```

每个产物目录保存 `command.txt`、`resolved_config.yaml`、`manifest.json`、
`summary.json`、运行时 Git 状态和 diff；recorded 目录另保存完整 stdout、Kit 日志、
退出码和 SHA-256 清单。

## 实测结果

| 检查 | 结果 | 判定 |
|---|---:|---|
| Gymnasium import/version | 1.4.0 | PASS |
| L0 进程退出码 | 0 | PASS |
| L0 articulation ready | 第 1 个物理步 | PASS |
| L0 步进失效次数 | 0 | PASS |
| 每个 action 物理步 | 恰好 6 | PASS |
| L0 reset settle | 21 个物理步 | PASS |
| L0 reset 最大残余轮速 | 0.019733 rad/s | PASS（阈值 0.10） |
| C0 +X_B 直行位移 | +3.651876 m / 5 s | PASS（阈值 +1.0） |
| C0 直行横漂 | -0.059449 m | PASS（绝对值阈值 0.25） |
| C0 直行偏航 | -7.019352 deg | PASS（绝对值阈值 10） |
| C0 正 omega 转向 | +135.178836 deg / 4 s | PASS（左转阈值 +90） |
| C0 转向平移漂移 | 0.119053 m | PASS（阈值 0.50） |
| 急停 2 s 后 root 速度 | 7.259e-5 m/s | PASS（阈值 0.05） |
| 急停 2 s 后 yaw rate | 5.318e-6 rad/s | PASS（阈值 0.05） |
| 急停 2 s 后最大轮速 | 0.0 rad/s | PASS（阈值 0.10） |
| C0 articulation 失效次数 | 0 | PASS |

上述动态结果支持现有 Husky 关节符号、`wheel_radius=0.17775 m`、
`wheel_track=0.57080 m` 和时序候选值；本窗口没有修改冻结规格。不同起点的早期
探针显示粗糙地形会影响 5 秒直线横漂/偏航，因此 C0 使用可复现的
`(x, y)=(1.25, 2.25)`，这不是用策略层反号补偿。

## 产物

- L0：[summary.json](../../data/benchmarks/p1/20261014_l0_dev_gpu6_recorded/summary.json)
- C0：[summary.json](../../data/benchmarks/p1/20261014_c0_dev_gpu6_recorded/summary.json)
- 场景图：[husky_lunalab.png](../../data/benchmarks/p1/20261014_c0_dev_gpu6_recorded/visual/husky_lunalab.png)
- 轨迹图：[trajectory.png](../../data/benchmarks/p1/20261014_c0_dev_gpu6_recorded/visual/trajectory.png)
- 控制曲线：[controls.png](../../data/benchmarks/p1/20261014_c0_dev_gpu6_recorded/visual/controls.png)
- 每物理步记录：[transitions.csv](../../data/benchmarks/p1/20261014_c0_dev_gpu6_recorded/transitions.csv)

场景图已人工检查，能看到 Lunalab、Terrain_lidar01 和 Husky。轨迹图标出各段起点、
终点及车体 +X 朝向；控制图包含归一化 action、slew 后车辆命令、四轮期望/实测
轮速和实际车体 `v/omega`。

## P1 Gate 边界

本窗口通过的是 P1 的最小 L0/C0 功能切片，不是完整 P1 Gate。尚未执行：碰撞与
成功终止触发、完整 Observation/Reward/termination、T0 确定性、500 次 reset、
10k/100k 长跑、B0/B1/B2/B3 吞吐、D0/R0/S0/V0，以及自动选择独占 GPU 条件下的
正式资源与墙上时间测量。因此当前不能作原生训练的最终 Go/No-Go。

后续 Task-v0 runtime、正式 T0、独立撞岩、完整 V0 和 500/500 的 R0 reset 压力
已经通过；证据见 [`p1_v0_results.md`](p1_v0_results.md) 与
[`p1_r0_results.md`](p1_r0_results.md)。正式 B0/B1 吞吐、D0 确定性、S0 长跑及
B2/B3 诊断仍未执行，故完整 P1 Gate 状态不变。
