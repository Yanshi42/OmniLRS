# P1 B1 完整 Task-v0 benchmark 结果

> 日期：2026-10-16
> 状态：`shared_conservative` PASS，性能分档 Yellow
> 场景：Lunalab / `Terrain_lidar01` / Husky / `render=False`

## 协议修订证据

原协议要求 B0/B1 都输出无终止 `step_core`。B0 seed 42 的真实周期动作轨迹在第 291
policy step 到达 `(x=5.798237, y=0.392030) m`，已越过 Task-v0 的 `y>=0.45 m`
边界，早于 500-step warmup。因此 B1 的无终止 step-core 不可满足。协议已修订为：
B0 保留两种口径；B1 只测完整 Task-v0 / Observation-v0 的
`rollout_with_reset`，它也是原有 B1 Gate 和性能分档唯一使用的口径。

## Harness 与动态预检

`scripts/rl/run_p1_benchmark.py --case b1` 直接接入 Lunalab DEM、101×101 几何高度图、
12 个具名 contact view、Reward-v0、termination 和局部 reset。每条 transition 校验
Observation-v0 的 10206 维 shape/version、finite action/observation/reward、6-step
decimation 和 articulation readiness。

首次 310-step 预检完成了动态 rollout，但保存 resolved YAML 时 NumPy `float64` 不被
OmegaConf 接受，进程 exit 1；该失败产物位于
`data/benchmarks/p1/20261016_b1_preflight_auto_shared_r2/`，不进入 Gate。配置已在写盘前
转换为 JSON primitive，并修正 execution failure 优先覆盖 validity/pass 的元数据逻辑。

修复后预检位于 `data/benchmarks/p1/20261016_b1_preflight_auto_shared_r3/`：310/310
measured step、1 次 out-of-bounds→局部 reset、Observation 契约错误 0，exit 0；短测
吞吐为 41.851534 env-step/s。因长度不是 500/10000，`gate_eligible=false`，只作动态
接线证据。

## 三 seed 结果

三个全新进程均自动选择物理 GPU 2（UUID
`GPU-9394be27-72a5-9643-4d7b-72e11e23e999`），完成 500 warmup + 10000 measured，
exit code 均为 0：

- `data/benchmarks/p1/20261016_b1_seed42_auto_shared_r1/`
- `data/benchmarks/p1/20261016_b1_seed43_auto_shared_r1/`
- `data/benchmarks/p1/20261016_b1_seed44_auto_shared_r1/`
- 聚合：`data/benchmarks/p1/20261016_b1_auto_shared_aggregate/`

| seed | env-step/s | RTF | p99 latency | local resets |
|---:|---:|---:|---:|---:|
| 42 | 41.184755 | 8.236951 | 0.039680 s | 45 |
| 43 | 41.203994 | 8.240799 | 0.036476 s | 43 |
| 44 | 42.486459 | 8.497292 | 0.034886 s | 41 |

聚合 median/min/max 为 41.203994/41.184755/42.486459 env-step/s，CV 1.463%；median
RTF 为 8.240799。三次 p99 的最坏值为 0.039680 s。按 median 估算 1M policy step
为 6.741526 小时，100M 为 28.089690 天。

三次 measured interval 共完成 129 次局部 reset：collision 26、out-of-bounds 30、
time-limit 73；另有 3 个 measurement-end 尾段。共保存 30000 条 measured transition。
articulation invalid、action decimation、Observation-v0 contract、action/observation/reward
non-finite 和完整 `World.reset()` 计数全部为 0。

项目 PID 显存峰值为 3390033920 bytes（3232.988 MiB），进程 RSS 峰值为
8353832960 bytes；分别满足 `<90% GPU total` 和 `<=24 GiB RSS`。共享卡的整卡
utilization、总显存、功耗和温度不归因给项目，aggregate 标记
`card_resource_metrics_eligible=false`。

## Gate 判定

- rollout median `41.203994 >= 5 env-step/s`：PASS；
- 三次最坏 p99 `0.039680 <= 1.0 s`：PASS；
- 三次吞吐 CV `1.463% <= 15%`：PASS；
- PID GPU 显存、RSS、finite、Observation、decimation、articulation、局部 reset：PASS。

B1 判定为 `decision=pass`、`performance_validity=shared_conservative`。按冻结分档，
`41.203994` 位于 `[5,50)`，因此是 **Yellow**：P1 功能路线可继续，但进入 P3 训练前
必须优化、使用多环境/多进程，或评估 off-policy；它不是 Green 单进程吞吐。

## 可复现命令

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case b1 --gpu auto --allow-shared-gpu \
  --seed 42 --warmup-steps 500 --measured-steps 10000 \
  --output-dir data/benchmarks/p1/20261016_b1_seed42_auto_shared_r1
```

seed 43、44 使用对应 seed 和新目录。聚合命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/summarize_p1_benchmark.py --case b1 --allow-shared-runs \
  --run-dir data/benchmarks/p1/20261016_b1_seed42_auto_shared_r1 \
  --run-dir data/benchmarks/p1/20261016_b1_seed43_auto_shared_r1 \
  --run-dir data/benchmarks/p1/20261016_b1_seed44_auto_shared_r1 \
  --output-dir data/benchmarks/p1/20261016_b1_auto_shared_aggregate
```

三次完整运行的 Isaac Kit 日志：

- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_001848.log`
- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_002337.log`
- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_002823.log`

收尾验证为 `61 passed`，`git diff --check` 无输出并以 exit code 0 结束。

独占 GPU 基线尚未执行，不影响共享保守 PASS，但卡级资源结论仍无资格。S0 与 B2/B3
已于后续正式通过；P1 最终判定见 [`p1_final_report.md`](p1_final_report.md)。
