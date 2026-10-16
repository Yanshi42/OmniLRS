# P1 D0 确定性结果

> 日期：2026-10-15
> 状态：D0 Gate PASS（三个共享 GPU 进程；只使用正确性指标）
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，GPU 6，共享诊断

## 固定 workload

- seed 42；
- 固定 start `(1.25, 2.25, 0.0)`、goal `(4.25, 2.25, 0.0)`；
- 每进程 1000 个冻结的 PCG64 uniform float32 Action-v0；
- terminal 后局部 reset，动作序列不重启；
- 三个全新 Isaac 进程逐步比较 episode boundary、termination reason、root position、
  wrapped yaw 和 reward；
- Gate 阈值保持 `1e-3 m`、`1e-3 rad`、`1e-4 reward`。

动作序列 SHA-256：
`f339db003da38251ac7e826860b1aab34488f66f13bd37b79b5b3aa6ea3b4c29`。

## 动态结果

三个进程均完成 1000/1000 policy step，articulation invalid、decimation error、
non-finite 和完整 `World.reset()` 次数均为 0。每个进程发生 7 次局部 reset，episode
boundary 都是：

```text
[108, 136, 263, 563, 619, 649, 949]
```

termination reason 也逐项相同：

```text
[out_of_bounds, out_of_bounds, collision, time_limit,
 out_of_bounds, out_of_bounds, time_limit]
```

三对进程比较结果：

| 指标 | 实测最大值 | Gate |
|---|---:|---:|
| 离散字段 mismatch | 0 | 必须为 0 |
| root position error | 0.0 m | `<=1e-3 m` |
| wrapped yaw error | 0.0 rad | `<=1e-3 rad` |
| reward error | 0.0 | `<=1e-4` |

聚合产物：
`data/benchmarks/p1/20261015_d0_1000_gpu6_shared_aggregate_r2/`。

## 结论边界

数值确定性比较为 PASS，D0 Gate 也为 PASS。GPU6 上存在外部 PID，但 D0 不使用
wall-time、整卡 utilization 或 card-level memory；它只比较三个新进程的离散事件和
数值轨迹，因此共享运行不污染判定。项目显存仍按 PID 独立归因，三个进程均无 OOM、
Xid、invalid articulation 或非 finite。

共享条件下的吞吐、延迟和整卡资源数据仍不具备性能资格，不能用于 B0/B1 或最终
Green/Yellow/Red 分档。

## 可复现命令

自动选择剩余显存最多、至少保留 8 GiB 的共享 GPU：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python scripts/rl/run_d0_determinism.py --gpu auto --allow-shared-gpu \
  --seed 42 --policy-steps 1000 --output-dir <new-run-dir>
```

共享诊断比较：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python scripts/rl/summarize_d0_determinism.py --allow-shared-runs \
  --run-dir data/benchmarks/p1/20261015_d0_1000_gpu6_shared_run1 \
  --run-dir data/benchmarks/p1/20261015_d0_1000_gpu6_shared_run2 \
  --run-dir data/benchmarks/p1/20261015_d0_1000_gpu6_shared_run3 \
  --output-dir data/benchmarks/p1/20261015_d0_1000_gpu6_shared_aggregate
```
