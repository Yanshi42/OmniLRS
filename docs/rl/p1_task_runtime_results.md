# P1 Task-v0 runtime 短程动态验证

> 日期：2026-10-15
> 状态：短程 runtime smoke 通过；完整 T0/R0/D0/benchmark 尚未执行
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，headless，`render=False`，物理 GPU 3

## 本次实现

- `OmniLRSEnv` 保留 L0/C0 的 11 维诊断模式，并可选接入 `TaskV0Runtime`；
- Task runtime 把 seeded/fixed task、Observation-v0、Reward-v0、physics-step contact/
  rollover 去抖、policy-step termination 和 episode 指标接入 `reset/step`；
- 新增 Lunalab DEM 双线性查询，越界/无效查询返回 `NaN`，Observation-v0 填 `-1`；
- 动态枚举 `nograph_husky.usd` 后，将 12 个具名刚体写入 `contact_links` 配置；
- 通过 Isaac 6.1 `RigidPrim(track_contact_forces=True)` 建立每 link 独立 contact view，
  四个允许轮地接触 link 仍由配置排除，不在任务逻辑中写死四轮；
- terminal、无效 action、仿真/terrain/contact provider 异常均立即写零目标。

## 单元测试

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python -m pytest -q test/rl
```

结果：`39 passed in 0.19s`。新增覆盖完整 observation/reward/termination 接线、每 action
精确 6 个 physics step、碰撞去抖、terminal observation、timeout truncation、terminal
急停、reset 清除 contact/path/reward 状态、DEM 插值/边界，以及 post-physics provider
异常急停。

## 动态命令与产物

通过运行：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/smoke_task_env.py --gpu 3 --seed 42 \
  --output-dir data/benchmarks/p1/20261015_task_v0_short_gpu3_r2
```

产物目录：`data/benchmarks/p1/20261015_task_v0_short_gpu3_r2/`

- `summary.json`：机器可读结论；
- `transitions.csv`：7 个 policy step 的状态、reward、接触和终止记录；
- `resolved_config.yaml`：环境、机器人、启动和固定任务配置；
- `command.txt`、`exit_code.txt`、`manifest.json`、`git_status.txt`、`git_diff.patch`；
- `run.log`：完整 Isaac 控制台日志；
- `checksums.sha256`：核心产物校验值。

实际结果：

| 指标 | 结果 |
|---|---:|
| 进程退出码 | 0 |
| contact view ready | true |
| articulation invalid count | 0 |
| Observation-v0 | `(10206,)`, `float32`, C-contiguous |
| action decimation | 每步恰好 6 |
| reset settle | 21 physics step；episode 外 |
| 初始/最终目标距离 | 3.000357 / 2.669654 m |
| 5 个前进 step 的距离进度 | +0.330703 m |
| 允许轮 link 接触力峰值 | 55.807585 N |
| 非轮 chassis link 接触力峰值 | 0 N |
| collision debounce | 全程 0 |
| 意外 termination/truncation | 无 |

这证明高于 20 N 的正常轮地接触没有被误分类为 collision，同时具名 chassis contact
view 和 articulation 可在同一直接 Isaac API 内环中共存。本次只有 2 个零动作 step 和
5 个前进 step，不作吞吐量、稳定长跑或确定性统计结论。

## 失败探针与修正

首次运行 `data/benchmarks/p1/20261015_task_v0_short_gpu3/` 保留了失败记录和完整日志。
Isaac 6.1 不接受“多个 sensor pattern + 空 filter pattern 列表”的单一 contact view，
返回空 contact tensor handle。修正为每个配置 link 一个具名 view 后，同一脚本通过。
这是 Isaac provider 实现修正，未修改 Task-v0 的 20 N 阈值、2-step 去抖或允许 link。

## P1 Gate 边界

本次通过：完整 Task-v0 API 接线的短程 headless 生命周期、真实 DEM observation、具名
contact view、正常轮地接触不误终止、前进 reward/距离进度、精确时序和局部 reset
缓存清理。

本报告执行时尚未完成底盘擦地/撞岩动态 collision、动态 rollover/out-of-bounds/
success/time-limit 全覆盖、同 seed 动态重复性、500 次 reset、10k/100k 长跑和正式
benchmark。后续正式 T0 与固定 `rock4` 独立撞岩已经通过，见
[`p1_t0_results.md`](p1_t0_results.md) 和
[`p1_rock_collision_results.md`](p1_rock_collision_results.md)；其余项目仍未执行，
完整 P1 Gate 尚未通过。

后续完整 V0 也已通过，见 [`p1_v0_results.md`](p1_v0_results.md)；正式 benchmark、
R0、D0 和 S0 仍未执行。
