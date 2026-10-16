# P1 T0 动态终止覆盖结果

> 日期：2026-10-15
> 状态：PASS
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，headless，`render=False`，物理 GPU 3

## 范围

在同一 Isaac 进程内复用 Lunalab、Terrain_lidar01、Husky articulation 和 12 个
具名 contact view，按冻结协议执行：

- success、collision、rollover、out-of-bounds、time-limit 各 10 次；
- 静止与正常轮地接触各 167 policy step，即各 1002 physics step；
- 每个 action 固定执行 6 个 physics step；
- 每个 terminal transition 检查 Gymnasium 语义、有限 observation、articulation
  生命周期、车辆命令和轮 drive target 急停。

collision 使用底盘向地形下移 `0.14 m` 的 chassis-scrape 注入。out-of-bounds
使用 `x=0.30 m`，并将 root 抬高 `0.75 m`，避免独立越界用例同时制造地形碰撞；
Task-v0 冻结的碰撞优先级与阈值均未修改。

## 复现

运行前确认物理 GPU 3 无其他 compute process，显存空载约 6 MiB：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/probe_terminations.py --gpu 3 --seed 42 \
  --repetitions 10 --stationary-policy-steps 167 \
  --wheel-contact-policy-steps 167 \
  --output-dir data/benchmarks/p1/20261015_t0_gpu3_r2
```

通过产物：`data/benchmarks/p1/20261015_t0_gpu3_r2/`

- `summary.json`、`termination_results.csv`：52 个聚合 case；
- `contact_samples.csv`：20244 个逐 physics-step 分类接触样本；
- `resolved_config.yaml`、`command.txt`、`manifest.json`；
- `run.log`、`exit_code.txt`、`checksums.sha256`；
- `git_status.txt`、`git_diff.patch`。

进程退出码为 0，产物 SHA-256 全部校验通过。

## 结果

| case | 结果 | 关键证据 |
|---|---:|---|
| success | 10/10 | `terminated=True`、`truncated=False`、reason `success` |
| out-of-bounds | 10/10 | `terminated=True`、`truncated=False`、reason `out_of_bounds` |
| rollover | 10/10 | `terminated=True`、`truncated=False`、75° roll 注入 |
| collision | 10/10 | chassis 峰值 6921.795916–39139.296924 N；连续超 20 N 为 2–6 sample |
| time-limit | 10/10 | 精确发生在 policy step 300；`terminated=False`、`truncated=True` |
| stationary | PASS | 1002 physics step；轮峰值 34.041858 N；chassis 峰值 0 N |
| wheel-contact | PASS | 1002 physics step；轮峰值 326.366927 N；chassis 峰值 0 N |

所有 52 个 case 的 articulation invalid count 合计为 0，所有 action 的
`physics_steps_this_action` 均为 6。50 个终止 transition 的 applied vehicle command
和全部轮 drive target 均立即为零；静止与轮地接触基线的 collision false positive
均为 0。

## 首轮失败与修正

首轮产物保存在 `data/benchmarks/p1/20261015_t0_gpu3/`，退出码为 2。该轮
success、rollover、collision、time-limit 均为 10/10，但 out-of-bounds 为 9/10：
trial 2 在粗糙 DEM 上同时产生 13144.841023 N chassis contact，按冻结优先级
`collision > rollover > out_of_bounds > success > time_limit` 正确报告 collision。

修正仅隔离测试夹具：越界 pose 额外抬高 0.75 m。没有修改运行时终止逻辑、20 N×2
去抖、轮/底盘分类或优先级。失败产物也已保存并通过 SHA-256 校验。

## Gate 边界

冻结协议的 T0、静止/轮地接触零误报及 chassis-scrape collision 10/10 已通过。
本次 T0 执行时尚未包含独立撞岩；后续固定 `rock4` 动态探针已通过，见
[`p1_rock_collision_results.md`](p1_rock_collision_results.md)。后续完整 V0 也已通过，
见 [`p1_v0_results.md`](p1_v0_results.md)。R0、D0、S0 和吞吐 benchmark 仍未执行，
因此完整 P1 Gate 尚未通过。

动态证据没有否定冻结的轮速符号、轮径、轮距、Action-v0 时序或接触阈值，
`task_v0_spec.md` 无需修订。
