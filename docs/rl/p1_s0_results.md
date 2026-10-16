# P1 S0 稳定性长跑结果

> 日期：2026-10-16
> 状态：`valid_shared_stability` PASS

## 实现

`scripts/rl/run_p1_benchmark.py --case s0` 复用完整 B1 Task-v0 / Observation-v0
rollout。冻结默认值为 seed 42、500 warmup、100000 measured policy transition。S0
允许共享 GPU，并沿用自动选择“占用最低且至少剩余 8 GiB”的 RTX 4090 和 PID 归因
资源监控。

为避免 benchmark 自身污染长跑内存斜率，S0 的 `transitions.csv` 在 rollout 中流式
写盘，不在 Python 列表中累计 100000 个 transition。独立 1 Hz monitor 仍分别记录项目
进程树 RSS、项目 PID 显存、其他 PID 显存和整卡指标。

资源 Gate 只使用 measured interval 内样本：分别计算前 10% 与后 10% median 的差，
以及对全部样本按估算 policy-step 做 least-squares slope。冻结阈值保持不变：RSS/GPU
增长各不超过 512 MiB，斜率各不超过 1 MiB/1000 env-step；同时检查 RSS 24 GiB、
项目 PID 显存 90%、100000/100000 完成度和全部正确性计数。

## 短程共享预检

命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case s0 --gpu auto --allow-shared-gpu \
  --seed 42 --warmup-steps 100 --measured-steps 1000 \
  --output-dir data/benchmarks/p1/20261016_s0_preflight_auto_shared_r1
```

运行自动选择物理 GPU 2（UUID
`GPU-9394be27-72a5-9643-4d7b-72e11e23e999`），exit code 0。实际结果：

- 1000/1000 transition，1001 行流式 CSV；
- 3 次局部 reset：collision 1、time-limit 2；
- articulation、decimation、Observation-v0、finite、完整 `World.reset()`错误均为 0；
- 41.343086 env-step/s，p99 0.039288 s；
- 项目 PID 显存峰值 3390033920 bytes，RSS 峰值 8377393152 bytes；
- measured interval 有 24 个资源样本；显存增长/斜率均为 0；
- RSS 首尾增长 1519616 bytes，短区间拟合斜率 1929986 bytes/1000 step。

短测不是冻结长度，manifest 正确标记 `validity=diagnostic_shared`、
`gate_eligible=false`、`performance_validity=not_applicable`。短区间 RSS 斜率高于正式阈值，
只说明 24 秒区间仍受缓存/瞬态影响；不得外推为 S0 FAIL。正式 S0 必须完整完成 100000
transition，再按完整 measured interval 判定。

产物目录：`data/benchmarks/p1/20261016_s0_preflight_auto_shared_r1/`。Isaac Kit 日志：
`.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_003902.log`。

## 正式 S0

命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case s0 --gpu auto --allow-shared-gpu \
  --seed 42 --warmup-steps 500 --measured-steps 100000 \
  --output-dir data/benchmarks/p1/20261016_s0_seed42_auto_shared_r1
```

运行自动选择同一物理 GPU 2；启动前已有外部 PID `301825`，因此整卡指标不归因给
项目。运行完成 500 warmup + 100000/100000 measured transition，exit code 0，manifest
为 `valid_shared_stability`、`gate_eligible=true`、`pass=true`。

- measured wall time 2331.283504 s，吞吐 42.894826 env-step/s，RTF 8.578965；
- 452 次局部 reset：collision 94、out-of-bounds 114、success 1、time-limit 243；
- articulation、decimation、Observation-v0、finite 和完整 `World.reset()`错误均为 0；
- step p99 0.035887 s，reset p99 0.154915 s；
- 100001 行 transition CSV、453 行 episode JSONL；
- measured interval 2331 个独立 1 Hz 资源样本；
- RSS 峰值 8334913536 bytes；前/后 10% median 增长 61808640 bytes
  （58.945 MiB），斜率 720726.504 bytes/1000 step（0.687 MiB/1000 step）；
- 项目 PID 显存峰值 2836385792 bytes；前/后 10% median 增长和斜率均为 0；
- RSS/GPU 峰值、增长、斜率、完成度和正确性 Gate 全部 PASS。

Kit 在 measurement 完成后、shutdown 前记录一次
`Streaming client gpu.foundation.TextureStreaming timed out`。该事件没有 OOM、Xid、CUDA
error、Python traceback、hang 或非零退出；100000 条 transition、summary 和资源样本均已
完整写盘，因此不否定 S0 Gate。完整 Kit 日志为
`.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_004247.log`。

正式产物目录：`data/benchmarks/p1/20261016_s0_seed42_auto_shared_r1/`。S0 判定 PASS。
