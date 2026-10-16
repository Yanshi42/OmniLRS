# P1 B0 benchmark harness 进展

> 日期：2026-10-15
> 状态：B0 `shared_conservative` 三 seed PASS；独占基线尚未执行
> 目标设备：运行时自动选择满足显存余量的 RTX 4090，不固定物理 index

## 已完成

- 新增 `scripts/rl/run_p1_benchmark.py`，单个全新 Isaac 进程依次执行冻结的
  `step_core` 和 `rollout_with_reset`；默认参数为 500 warmup、10000 measured。
- `step_core` 使用冻结的 6 段周期动作，每段保持 25 policy step；意外 terminal
  立即失败。
- `rollout_with_reset` 使用 seeded uniform random action，terminal 后只做局部 reset，
  测量墙上时间包含局部 reset 和 settle。
- 新增 `scripts/rl/resource_monitor.py`，以独立进程和 host monotonic timestamp 每秒
  记录递归进程 RSS/VMS/CPU、系统内存以及 GPU 利用率、显存、温度、功耗、时钟和
  全部 GPU PID。GPU 指标通过 NVML 直接读取，不在每秒采样中启动 `nvidia-smi`。
- 新增 `scripts/rl/summarize_p1_benchmark.py`，只接受 seed 42/43/44、500/10000、
  exit-0 的三份同类 RTX 4090 结果。默认只接受独占结果；显式
  `--allow-shared-runs` 时只接受明确标记为 `valid_shared_conservative` 且具备 Gate
  资格的完整运行。旧 `diagnostic_shared` 探针不能进入聚合。三个 seed 可以使用不同
  index，但每次必须记录 UUID/型号，并计算 median/min/max/CV 和冻结的
  `step_core >=20 env-step/s` Gate。
- benchmark 显式关闭环境的逐 physics-step 内存 trace，但仍输出每 policy step 的
  `transitions.csv`。环境默认 trace 行为未变，L0/C0/V0 证据不受影响。

runner 默认 `--gpu auto`：扫描全部物理卡，选择无 compute/graphics PID、显存低于
1024 MiB、利用率低于 5% 且 `memory.used` 最低的候选，再对所选卡执行连续 10 秒
1 Hz 独占检查。没有候选时 exit 3，不退化为共享正式测量。

自动发现路径已在 `data/benchmarks/p1/20261015_b0_auto_gpu_preflight/` 动态验证：当时
8 张 RTX 4090 均有 compute PID，runner 保存全部 index/UUID/型号/显存/利用率/PID
快照，返回 `no_idle_exclusive_gpu` 和 exit code 3，没有启动 Isaac。该结果证明无空闲卡
时会安全拒绝，不是 B0 性能数据。

纯函数、环境生命周期和统计测试累计 59 项通过；`git diff --check` 通过。

## 动态预检与失败证据

`data/benchmarks/p1/20261015_b0_preflight_gpu3/` 完成了 20 warmup + 100 measured 的
两个短 workload。根据保存的逐步 latency，诊断值约为：

| workload | 短预检 env-step/s | 说明 |
|---|---:|---|
| step_core | 25.4850 | 100 step，不是正式结果 |
| rollout_with_reset | 25.9537 | 100 step，区间内未发生 reset |

该目录为无效预检：首版 runner 在 `SimulationApp.close()` 后才写最终文件，而 Isaac
6.1 fast shutdown 会在该调用中结束解释器，因此缺少 summary/manifest。原始
transition、资源 CSV 和 stdout 保留，不用于 Gate。

随后 GPU 3 被外部进程 PID `3114826` 占用约 40 GiB。runner 按当时冻结的独占条件拒绝
运行并返回 exit code 3；完整拒绝 manifest 位于
`data/benchmarks/p1/20261015_b0_preflight_gpu3_r3/`。没有终止或干扰该外部进程，
也没有把共享运行冒充正式数据。

runner 已修正为在 shutdown 请求前完成全部产物，并调用
`SimulationApp.close(exit_code=exit_code)` 保留真实进程状态码。

## GPU 4 共享动态预检

经用户指定，在已有外部 PID `2114079` 的 GPU 4 上显式使用
`--allow-shared-gpu` 执行开发预检。runner 将结果标为 `diagnostic_shared` 和
`formal_eligible=false`。

通过产物：`data/benchmarks/p1/20261015_b0_preflight_gpu4_shared_r2/`

| 指标 | step_core | rollout_with_reset |
|---|---:|---:|
| warmup / measured | 20 / 310 | 20 / 310 |
| env-step/s | 24.981366 | 25.150559 |
| RTF | 4.996273 | 5.030112 |
| step latency p99 | 0.055832 s | 0.056704 s |
| measured local reset | 不适用 | 1 |
| reset latency p95 | 不适用 | 0.124457 s |

rollout 在第 300 policy step 正确产生 `time_limit`，局部 reset 后继续到第 310 step。
articulation invalid、action decimation error、action/observation/reward non-finite 和完整
`World.reset()` 次数均为 0。进程 exit 0，620 个 measured transition、2 条 episode
记录、64 组 1 Hz 资源样本和全部 SHA-256 校验均通过。卡级 GPU 峰值含外部进程，
不用于正式资源 Gate。

### GPU 4 显存余量复测

为确认外部进程占用约 29.5 GiB 时是否仍能运行 B0，于同一卡再次执行 20 warmup +
310 measured 的两个 workload：

`data/benchmarks/p1/20261015_b0_gpu4_vram_probe_r1/`

运行前 GPU 4 总显存为 49140 MiB，已用 29545 MiB、空闲 18966 MiB。55 组 1 Hz
NVML 样本记录到整卡峰值 32807.5625 MiB；本次 Isaac 进程峰值 3232.988 MiB，整卡
相对运行前增加约 3262.5625 MiB，峰值时仍余 16332.4375 MiB。进程 RSS 峰值
8321417216 bytes，未出现 OOM/Xid，exit code 0。

`step_core` 为 23.953532 env-step/s，`rollout_with_reset` 为 24.867471 env-step/s；
后者在第 300 step 正确 time-limit 并完成一次局部 reset。articulation invalid、decimation
error、非 finite 和完整 `World.reset()` 均为 0。因此当前余量足以运行单车 B0，但该卡
仍有外部 PID `2114079`，结果保持 `diagnostic_shared`、`formal_eligible=false`，不用于
正式吞吐或资源 Gate。

### 完整长度共享尝试：运行时争用中止

随后尝试在 GPU 4 上执行 seed 42 的完整 500 warmup + 10000 measured 开发诊断，产物
目录为 `data/benchmarks/p1/20261015_b0_seed42_gpu4_shared_full_r1/`。运行期间新外部
PID `3248801` 于 `2026-10-15T09:16:49.306927+00:00` 加入，峰值占用 5046 MiB，整卡
利用率升至 98%。为避免继续干扰外部 workload，项目进程被主动停止。

中止前保存了 260 组 1 Hz 资源样本；项目进程显存峰值 2602.988 MiB，整卡峰值
37228.5625 MiB，最小剩余 11911.4375 MiB。由于 workload 未完成，该目录没有
`summary.json`、`manifest.json` 或 runner `exit_code.txt`，并由 `interruption.json`
明确标记 `runtime_resource_contention`、`formal_eligible=false`。它只证明运行期争用
检测与人工止损有效，不是吞吐或稳定性结果。

基于该证据，独立资源监控器已加入运行期争用哨兵：启动时已有的共享 PID 作为允许
集合，新出现的 compute/graphics PID 会写入 `resource_contention.json`。正式独占
runner 每个 policy step 检查该文件，并以 exit code 3、`invalid_resource_contention`
主动结束；共享开发诊断继续运行，只保留争用记录且不具备正式资格。相关纯函数、
PID 显存归因和 runner 哨兵测试均已纳入 51 项测试。
独立 NVML 集成探针也已验证：允许 PID `2114079` 时，监控器在首个采样周期识别出
未允许的 PID `3248801` 并写入包含时间戳和 PID 集合的争用文件。

GPU CSV 现按每个采样点保存项目 PID 列表、逐 PID 显存、项目 PID 去重总和、其他
PID 去重总和和未归属显存。`gpu_memory_peak_bytes` 与
`project_gpu_memory_peak_bytes` 都表示项目进程树峰值；整卡峰值另存为
`card_gpu_memory_peak_bytes`，不再用整卡前后差值估算项目显存。

### PID 显存完整长度验证

新口径先以 2 warmup + 5 measured 的真实 Isaac smoke 验证，随后在 GPU 4 完成 seed 42
的完整 500 warmup + 10000 measured 两段 workload：

`data/benchmarks/p1/20261015_b0_seed42_gpu4_shared_pid_full_r1/`

| 指标 | step_core | rollout_with_reset |
|---|---:|---:|
| measured policy step | 10000 | 10000 |
| measured wall time | 305.900940 s | 310.982878 s |
| env-step/s | 32.690321 | 32.156111 |
| physics-step/s | 196.141928 | 192.936667 |
| RTF | 6.538064 | 6.431222 |
| step latency p99 | 0.051224 s | 0.050056 s |
| measured local reset | 不适用 | 33 |
| reset latency p95 | 不适用 | 0.120430 s |

686 组 1 Hz 样本记录到项目 PID `3273024` 显存峰值 3232.988281 MiB（3.157215
GiB），进程 RSS 峰值 7.722481 GiB。外部 PID 显存峰值为 34558 MiB，整卡峰值为
37836.875 MiB；这些值没有计入项目显存。运行中新的 PID `3291184` 加入并占用约
5024 MiB，监控器正确写入争用哨兵，但共享诊断继续到 exit 0。

共保存 20000 个 measured transition 和 34 条 episode 记录；rollout 完成 33 次
time-limit→局部 reset。articulation invalid、action decimation error、非 finite 和完整
`World.reset()` 均为 0，无 OOM/Xid。结果为 `pass=true`、`diagnostic_shared`、
`formal_eligible=false`：它验证完整长度生命周期与项目 PID 显存，但吞吐值不进入正式
三 seed Gate。

## 当前三 seed 共享保守结果

2026-10-15 使用当前代码和 `--gpu auto --allow-shared-gpu` 分别运行 seed 42、43、44。
自动选择结果为 seed 42/43 使用物理 GPU 6，seed 44 使用当时显存占用更低的物理
GPU 4；没有固定 GPU index。三个进程均完成 500 warmup、10000 step-core measured 和
10000 rollout measured，exit code 均为 0。

原始产物：

- `data/benchmarks/p1/20261015_b0_seed42_auto_shared_current_r1/`
- `data/benchmarks/p1/20261015_b0_seed43_auto_shared_current_r1/`
- `data/benchmarks/p1/20261015_b0_seed44_auto_shared_current_r1/`
- `data/benchmarks/p1/20261015_b0_auto_shared_current_aggregate/`

每个原始目录均包含执行命令、解析配置、环境版本、preflight、PID/整卡资源 CSV、
逐 transition CSV、episode JSONL、summary、manifest、exit code、git status 和 diff 快照。
对应 Isaac Kit 完整日志为：

- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261015_124415.log`
- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261015_125646.log`
- `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261015_131017.log`

| workload | median env-step/s | min / max | CV | median RTF | median p99 latency |
|---|---:|---:|---:|---:|---:|
| step_core | 30.031241 | 28.283701 / 32.959477 | 6.340% | 6.006248 | 0.055040 s |
| rollout_with_reset | 26.756761 | 26.514410 / 29.211256 | 4.431% | 5.351352 | 0.054540 s |

step-core median 比冻结的 20 env-step/s 门槛高 50.16%，因此按共享性能单向保守规则
判定 `decision=pass`、`performance_validity=shared_conservative`。按 step-core median
估算 1M/100M policy step 分别为 9.249627 小时/38.540113 天；rollout 对应估算为
10.381592 小时/43.256633 天。这些是当时共同负载下的观测值，不是独占 GPU 能力上限。

三次 rollout 共完成 99 次 measured time-limit→局部 reset，保存 60000 条 measured
transition。articulation invalid、action decimation、action/observation/reward non-finite
和完整 `World.reset()` 计数全部为 0。项目 PID 显存峰值在三次中均为
3390033920 bytes（3232.988 MiB），进程 RSS 最大值为 8359587840 bytes。共享卡整卡
显存、utilization、功耗和温度不归因给项目，aggregate 明确标记
`card_resource_metrics_eligible=false`。

实际执行命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case b0 --gpu auto --allow-shared-gpu --seed 42 \
  --warmup-steps 500 --measured-steps 10000 \
  --output-dir data/benchmarks/p1/20261015_b0_seed42_auto_shared_current_r1
```

seed 43、44 使用相同命令和各自目录。三次完成后运行：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/summarize_p1_benchmark.py --case b0 --allow-shared-runs \
  --run-dir data/benchmarks/p1/20261015_b0_seed42_auto_shared_current_r1 \
  --run-dir data/benchmarks/p1/20261015_b0_seed43_auto_shared_current_r1 \
  --run-dir data/benchmarks/p1/20261015_b0_seed44_auto_shared_current_r1 \
  --output-dir data/benchmarks/p1/20261015_b0_auto_shared_current_aggregate
```

后续若获得独占卡，可去掉两个 `--allow-shared-*` 参数补充干净性能基线；这不是当前
共享保守 B0 PASS 的前置条件。

## Gate 边界

B0 三 seed、500 warmup + 两段各 10000 measured 已在共享 GPU 保守轨通过。独占 B0
基线未执行，不影响当前保守 PASS，但卡级资源结论仍无资格。后续 B1、S0 与 B2/B3
均已完成；P1 最终判定见 [`p1_final_report.md`](p1_final_report.md)。
