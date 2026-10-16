# P1 Task-v0 独立撞岩结果

> 日期：2026-10-15
> 状态：PASS
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，物理 GPU 3
> 动态判定：headless、`render=False`；截图阶段单独启用渲染

## 范围

本探针在 Terrain_lidar01 中固定加载仓库真实岩石资产
`assets/USD_Assets/rocks/lunalab_rocks/rock_4/rock4.usd`，Husky 从固定起点沿
`+X_B` 驶向岩石。随机岩石关闭，固定靶标位于
`[2.60, 2.25, 0.082695] m`。

除了 Task-v0 使用的 12 个具名 link net-contact view，探针还为每个配置中的 contact
link 建立到岩石碰撞 Mesh
`/Lunalab/RLProbeAssets/rock4/rock4/rock4` 的过滤接触视图。因此可以区分岩石接触
与地形接触，不以车辆位置推断碰撞来源。

## 复现

运行前确认物理 GPU 3 无其他 compute process，显存空载约 6 MiB：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/probe_rock_collision.py --gpu 3 --seed 42 \
  --max-policy-steps 30 --capture-scene \
  --output-dir data/benchmarks/p1/20261015_rock_collision_gpu3_r3
```

通过产物：`data/benchmarks/p1/20261015_rock_collision_gpu3_r3/`

- `summary.json`、`contact_samples.csv`、`trace.csv`；
- `resolved_config.yaml`、`command.txt`、`manifest.json`；
- `run.log`、`exit_code.txt`、`checksums.sha256`；
- `visual/rock_collision.png`、`visual/trajectory.png`、
  `visual/contact_forces.png`；
- `git_status.txt`、`git_diff.patch`。

进程退出码为 0，全部产物 SHA-256 校验通过。

## 动态结果

| 指标 | 实测 |
|---|---:|
| 终止 policy step | 6 |
| physics contact sample | 36 |
| termination reason | `collision` |
| Gymnasium 语义 | `terminated=True`、`truncated=False` |
| 岩石过滤 chassis 峰值 | 52.902455 N，`base_link` |
| 岩石过滤连续超 20 N | 2 physics step |
| 全来源 chassis 峰值 | 716.837655 N，`base_link` |
| 全来源连续超 20 N | 2 physics step |
| 岩石过滤 wheel 峰值 | 0 N |
| articulation invalid | 0 |
| 每 action physics step | 6 |
| terminal reward | -10 |

岩石过滤 chassis 接触在 physics sample 29、30 分别为 26.456679 N 和
52.902455 N，恰好满足冻结的 `>20 N × 连续 2 physics step`。终止时车辆级 applied
command 为 `[0, 0]`，全部轮 drive target 为 0；contact view 与过滤 contact view
全程保持 ready。

场景截图可直接看到 Husky 的前部与 `rock4` 接触。轨迹从 root
`[1.249643, 2.248008] m` 到 `[1.691496, 2.230769] m`；岩石原点与 root 的间距
包含 Husky 前悬和岩石约 0.45 m 的负 X 几何范围，和过滤接触证据一致。

## 迭代记录

- `20261015_rock_collision_gpu3`：物理碰撞已发生，但 Replicator 首次返回全黑帧，
  脚本在写摘要前退出，exit code 1；
- `20261015_rock_collision_gpu3_r2`：动态摘要 `pass=true` 且截图有效，但
  `resolved_config.yaml` 接收到未转换的 NumPy array，exit code 1；
- `20261015_rock_collision_gpu3_r3`：数组改为普通 list 后完整 exit 0。

三个目录均保留，已有文件的 SHA-256 均校验通过。修正只涉及截图预热与产物
序列化，不修改碰撞阈值、去抖、link 分类或 Task-v0 运行时。

## Gate 边界

至此，Task-v0 要求的正常轮接触、底盘擦地和独立撞岩三类动态接触证据均已完成，
正式 T0 保持通过。后续完整 V0 也已通过，见
[`p1_v0_results.md`](p1_v0_results.md)。R0、D0、S0 和吞吐 benchmark 仍未执行，
因此完整 P1 Gate 仍为 `IN PROGRESS`。

本次证据没有否定冻结的轮速符号、轮径、轮距、Action-v0 时序或 20 N×2 接触
阈值，`task_v0_spec.md` 无需修订。
