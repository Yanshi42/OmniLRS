# P1 Task-v0 动态终止单次探针

> 日期：2026-10-15
> 状态：六个短程 case 全部通过；不是正式 T0
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，headless，`render=False`，物理 GPU 3

## 范围

在一个 Isaac 进程内复用 Lunalab、Terrain_lidar01、Husky articulation 和 12 个具名
contact view，通过局部 episode reset 依次执行：

- normal：5 个零动作 policy step；
- success：把 root 注入固定 goal 的合法 pose；
- out-of-bounds：把 root 注入有效 DEM 内但 Task-v0 边界外的 `x=0.30 m`；
- rollover：root 离地后注入 `75°` roll；
- collision：把已 settle 的 chassis 向地形下移 `0.14 m`，形成可重复的底盘擦地；
- time-limit：零动作执行完整 300 个 policy step。

每个 action 仍通过环境执行 6 个 physics step。脚本逐 physics step 保存允许轮 link 与
非轮 chassis link 的最大接触力；terminal transition 后检查 drive target 已立即为零。

## 复现

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/probe_terminations.py --gpu 3 --seed 42 \
  --output-dir data/benchmarks/p1/20261015_termination_probe_gpu3
```

结果目录：`data/benchmarks/p1/20261015_termination_probe_gpu3/`

- `summary.json`、`termination_results.csv`：各 case 的期望/实际 reason 和 Gym 语义；
- `contact_samples.csv`：1848 个逐 physics-step 分类接触样本；
- `resolved_config.yaml`、`command.txt`、`manifest.json`；
- `run.log`、`exit_code.txt`、`checksums.sha256`；
- `git_status.txt`、`git_diff.patch`。

进程退出码为 0，产物 SHA-256 全部校验通过。

## 结果

| case | 实际 reason | terminated | truncated | policy step | 关键证据 |
|---|---|---:|---:|---:|---|
| normal | `None` | false | false | 5 | 30 physics sample；轮峰值 34.041858 N，chassis 0 N |
| success | `success` | true | false | 1 | success flag true；terminal reward +10 |
| out-of-bounds | `out_of_bounds` | true | false | 1 | terminal reward -10 |
| rollover | `rollover` | true | false | 1 | max roll 1.308997 rad；3-step 去抖被锁存 |
| collision | `collision` | true | false | 1 | `base_link` 峰值 16039.270146 N；连续 5 sample >20 N |
| time-limit | `time_limit` | false | true | 300 | 1800 physics sample；terminal reward 0 |

全部 case：

- articulation invalid count 为 0；
- contact view 全程 ready；
- `physics_steps_this_action == 6`；
- terminal observation finite 且 shape `(10206,)`；
- terminal 后四轮 drive target 为 0；
- collision、rollover latch 在 reset 后被清除，后续 case 未串扰。

本次 chassis-scrape 是故意制造的深度穿透探针，16 kN 峰值只用于确认分类、去抖和
优先级链路，不能作为自然碰撞力分布或 20 N 阈值标定。现有动态证据没有否定冻结的
轮速符号、轮径、轮距、时序或接触阈值，因此未修改 `task_v0_spec.md`。

## Gate 边界

本次只完成每种 reason 1 次。正式 T0 要求五种 reason 各 10/10；静止和正常轮地接触
各要求 1000 physics step，chassis collision 要求 10/10，当时还需独立撞岩 case。
因此这份结果只是正式 T0 前的动态可行性证据。后续正式 T0 和固定 `rock4` 独立
撞岩均已于 2026-10-15 通过，见 [`p1_t0_results.md`](p1_t0_results.md) 与
[`p1_rock_collision_results.md`](p1_rock_collision_results.md)；完整 P1 Gate 仍未通过。
