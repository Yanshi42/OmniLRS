# OmniLRS-native RL P1 benchmark protocol

> 状态：Frozen for P1 execution
> 协议日期：2026-10-14
> 适用规格：[`task_v0_spec.md`](task_v0_spec.md)
> 主机基线：8 × RTX 4090，Isaac Sim 6.1，Pixi default environment

本文冻结 P1 的功能、稳定性和性能测量方法。所有 P1 Go/No-Go 结论必须由本文
定义的产物支持；单次视频、Kit FPS、physics FPS 或去掉 reset 的最好成绩都不能
替代主指标。

## 1. 基准不变量

- 唯一入口：`env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default`。
- 单个 Lunalab、单个 Husky、非 ROS、非 Yamcs、非 SDG 生命周期。
- physics dt 精确 `1/30 s`，action decimation 6，policy period `0.2 s`。
- `environment.enforce_realtime=False`；benchmark 内禁止 `sleep` 或实时 rate limiter。
- Task-v0 主性能测试使用 headless、`World.step(render=False)`。
- 不包含 PPO、神经网络推理、日志上传或 ROS 2 消息开销。
- 吞吐量计时使用 `time.perf_counter()`；MDP 状态仍只使用仿真时钟。
- 每个正式重复必须启动全新的 Isaac Sim 进程，不能在同一进程重复后取最好值。

代码实现后，冻结命令模板为：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi \
  pixi run -e default python scripts/rl/run_p1_benchmark.py \
  --case b0 --gpu auto --seed 42 \
  --output-dir data/benchmarks/p1/<run_id>
```

当前入口已支持 B0/B1/B2/S0；后续 case 继续复用相同的自动 GPU 选择、资源监控和非零失败码
契约。

## 2. 资源隔离与有效运行条件

### 2.1 按 case 自动选择 GPU

主机的 8 张卡均为 RTX 4090。P1 不再固定物理 index；每个全新进程启动前扫描所有
物理 GPU，并按 case 类型选择卡。显存相同时选择较小 index。

**性能/相对成本 case**：B0、B1 以及需要和 B1 比较相对开销的 B2/B3 优先独占，
用于取得可复现的单卡隔离基线。命令默认使用 `--gpu auto`；显式 index 只用于复现
历史运行。

preflight 同时满足才允许开始：

```text
所选 GPU memory.used < 1024 MiB
所选 GPU utilization.gpu < 5%
所选 GPU 上没有其他 compute/graphics 进程
连续 10 s、每秒采样一次均满足以上条件
```

没有独占候选卡时，默认命令必须 exit non-zero 并记录 `no_idle_exclusive_gpu`，避免
静默改变结果语义。显式加入 `--allow-shared-gpu` 后，runner 自动选择占用最低且至少
剩余 8 GiB 的 RTX 4090，结果标记为 `shared_conservative`。seed 42/43/44 可以落在
不同物理 index，但必须是同一冻结型号；每次运行和最终聚合都保存实际 index、UUID、
型号、选择前快照及全部外部 PID。

独占测量期间如果出现外来 GPU 进程、ECC/Xid、显存 OOM、温度降频或 GPU 利用率被
其他任务污染，本次标记 `invalid_resource_contention`。共享测量继续运行并记录外部
PID；OOM、Xid 或本项目错误仍直接判无效。

共享性能采用单向保守判定：三次完整重复的 observed throughput 若仍达到冻结阈值，
可以判 PASS；若未达到，只能判 `inconclusive_shared`，不能据此作性能 FAIL 或 No-Go。
共享 step/s、latency 和训练时间估算表示当时共同负载下的实际下界，不声称等于单卡
隔离能力。B2/B3 必须记录与 B1 的资源争用快照；共享相对倍率只作近似，不能覆盖之后
取得的隔离配对结果。

**正确性 case**：L0/C0、T0、R0、D0、V0 可以使用共享 GPU；S0 的稳定性、项目 PID
显存/RSS 趋势也可以在共享 GPU 上验证。自动选择 `memory.used` 最低且至少保留
8 GiB 显存的 RTX 4090。项目显存以 benchmark 主进程及其递归子进程的 NVML PID
显存去重求和，其他 PID 单独记录。只要没有 OOM、Xid、invalid articulation 或本项目
进程错误，这些 case 的离散/数值正确性仍可进入对应 Gate；但共享运行的 wall-time、
整卡 utilization、功耗、温度和 card-level memory 不得作为项目独占性能结论。

渲染器、PhysX 和 CUDA 必须全部绑定本次选择的物理 GPU。manifest 必须保存解析后的
`activeCudaGpus`、`physics/cudaDevice` 和 `CUDA_VISIBLE_DEVICES`；禁止逻辑编号与
物理编号含混。

### 2.2 主机状态

运行前保存但不强制整机空闲：

- `nvidia-smi` 完整输出与 GPU 查询 CSV；
- CPU 型号、逻辑核数、load average；
- 总内存、available memory 和 swap 使用量；
- 同机 Isaac/Kit 进程列表；
- Git commit、`git status --short`、`pixi.lock` SHA-256。

正式结果应来自 clean tracked worktree。若为诊断未提交代码，必须保存
`git diff --binary`，并将结果标为 `development_only`。

## 3. 冻结场景矩阵

所有 case 使用 `Terrain_lidar01`；`Terrain_lidar02` 留给 P2/P4 的未见地形验证，
不能用来挑选更好看的 P1 性能结果。

| ID | 内容 | 高度图 | 相机 | 地形变形 | render | P1 Gate |
|---|---|---|---|---|---|---|
| `L0` | 启动、加载 Husky、articulation ready、固定步进、退出 | 关 | 关 | 关 | false | 必须 |
| `C0` | 直行、转向、急停和轮速符号标定 | 关 | 关 | 关 | false | 必须 |
| `B0` | robot state/action 核心 step，不构建完整 observation | 关 | 关 | 关 | false | 必须 |
| `B1` | 完整 Task-v0 / Observation-v0 rollout | 101×101 几何图 | 关 | 关 | false | 主 Gate |
| `B2` | B1 + D455 RGB/depth 成本诊断 | 开 | 320×240 @ 5 Hz | 关 | 每 6 physics step | 只报告 |
| `B3` | B1 + terrain deformation 成本诊断 | 开 | 关 | 开 | false | 只报告 |
| `V0` | 固定脚本的可视化回放 | 开 | 可选 | 关 | true | 必须产物，不计性能 |

`B2/B3` 可以因尚未进入对应 actor 阶段而显著变慢，但必须独立运行并报告相对
`B1` 的吞吐量、显存和内存增量。不得同时打开相机和变形后把下降归因给其中一个。

B3 冻结为每个 policy step 的 6 个 physics step 完成后调用一次现有
`TerrainManager.deformTerrain()`；输入 link 来自 robot adapter 的
`allowed_ground_contact_links`，姿态使用 Isaac `wxyz` 顺序。更新后的 DEM 必须同步给
Observation-v0 查询。当前成本诊断沿用现有 `update_collider=False`，因此它覆盖 DEM 与
USD mesh 更新成本，不冒充动态 collider 或完整轮壤耦合验证。

## 4. 固定 seed、动作和运行长度

### 4.1 独立性能重复

`B0` 和 `B1` 各运行三个全新进程：

```text
seeds = [42, 43, 44]
warmup = 500 policy steps = 3000 physics steps = 100 sim s
measured = 10,000 policy steps = 60,000 physics steps = 2000 sim s
```

`B0` 输出两种口径：

1. `step_core`：一次 reset 后执行不会触发终止的冻结周期动作，计时不含启动和
   初始 reset，用于定位纯 step 成本。
2. `rollout_with_reset`：seeded uniform random action，终止后立即局部 reset；计时
   包含所有局部 reset 和 settle 的墙上时间。

`B1` 只输出完整 Task-v0 / Observation-v0 的 `rollout_with_reset`，这是训练容量主指标，
也是 B1 Gate 和 Green/Yellow/Red 分档的唯一吞吐口径。不得用 B0 `step_core` 替代。

B0 `step_core` 的周期动作固定为下列 normalized action，每项保持 25 policy step 并循环：

```text
[0.0,0.0], [0.5,0.0], [0.0,0.5], [0.5,0.0], [0.0,-0.5], [-0.25,0.0]
```

B0 `step_core` 使用固定安全起点；意外 terminal 立即判该次失败，不能自动 reset 后
继续伪装成 core 成绩。B1 不复用这一口径：2026-10-15 的 B0 动态轨迹显示冻结周期
动作在第 291 policy step 已越过 Task-v0 边界，而 warmup 本身为 500 step，因此为
B1 强行保留“无 terminal step_core”会形成不可满足的 workload。该修订不改变 B0
既有 Gate，也不改变 B1 的 rollout 主 Gate。

### 4.2 长跑、reset 与确定性

| Case | 固定长度 |
|---|---|
| `S0` 稳定性长跑 | B1、seed 42、100,000 个 measured policy transition，包含自动局部 reset |
| `R0` reset 压力 | 500 次连续局部 reset；每次 reset 前执行 20 个随机 policy step |
| `D0` 确定性 | seed 42、冻结 start/goal 和动作序列、1000 policy step、3 个全新进程 |
| `T0` 终止覆盖 | success/collision/rollover/out-of-bounds/time-limit 各至少 10 次脚本触发 |
| `B2/B3` 诊断 | warmup 100、measured 1000 policy step，各 seed 42 单进程 |
| `V0` 可视化 | seed 42 的直行、转向、碰撞、成功各 1 条轨迹 |

S0 必须完成全部 100,000 transition；墙上时间过长本身就是吞吐结论，不能用固定
时长提前截断后外推稳定性。

S0 的 transition 必须流式写盘，不得在 benchmark 进程内累计完整 100,000-step 行列表，
避免日志缓存本身污染内存增长 Gate。RSS/GPU 趋势仅使用 measured interval 时间戳内的
1 Hz PID 归因样本，启动、warmup 和 shutdown 样本不进入首尾 10% 或斜率计算。

D0 固定使用 start pose `(1.25, 2.25, 0.0)`、goal pose `(4.25, 2.25, 0.0)`；动作由
NumPy `default_rng(42)` 的 PCG64 生成 1000 个 `uniform([-1,1], shape=(2,))`，随后
转为 `float32`。动作序列按全局 policy index 连续消费，episode terminal 后只做局部
reset，不重新启动动作序列。每次运行保存动作原始值及其字节级 SHA-256。

## 5. 指标定义

### 5.1 计时与吞吐量

```text
env_steps_per_s     = measured policy transitions / measured wall seconds
physics_steps_per_s = 6 * measured policy transitions / measured wall seconds
real_time_factor    = measured simulated seconds / measured wall seconds
                    = 0.2 * env_steps_per_s
T_1M_hours          = 1e6 / env_steps_per_s / 3600
T_100M_days         = 1e8 / env_steps_per_s / 86400
```

报告每步 wall latency 的 p50/p95/p99、reset latency 的 p50/p95/p99，以及三次独立
重复的 median、min、max 和 coefficient of variation。startup 单独分解为：

```text
process_start -> app_ready -> scene_ready -> articulation_ready -> first_observation
```

shutdown 从请求关闭到进程退出单独计时，不并入 env throughput。

### 5.2 正确性与 episode 指标

- termination reason 计数与 episode length 分布；
- observation/action 非 finite、shape/range 违规次数；
- articulation/contact view 初始化失败次数；
- full `World.reset()` 次数；
- success、collision、rollover、out-of-bounds、time-limit 次数；
- path length、最大 roll/pitch、reward 分量总和；
- reset 前后 root/wheel velocity、drive target、history/counter 残留。

### 5.3 资源指标

独立监控器以 1 Hz 采样，使用 host monotonic timestamp：

- benchmark 主进程及递归子进程 RSS、VMS、CPU%；
- 系统 available memory 与 swap；
- benchmark 主进程及递归子进程对应的 NVML PID、逐 PID 显存及其去重总和；
- 所选 GPU utilization、卡级 memory.used、temperature、power.draw、SM/memory clocks；
- GPU 上所有其他 PID、逐 PID 显存，以及未归属到 NVML process list 的卡级显存。

显存主指标为项目进程树的 `project_gpu_memory_peak_bytes`，按 PID 去重，不能用整卡
`memory.used` 增量替代。卡级利用率、功耗、温度和总显存占用只有在第 2 节的独占
条件成立时可用于正式资源 Gate；共享诊断中的项目 PID 显存仍可作为隔离后的诊断值。
进程 RSS 使用 `psutil.Process(pid).memory_info().rss` 加所有存活 descendants，避免
遗漏 Kit 子进程。

长跑内存趋势在 warmup 后计算：取 measured 区间前 10% 和后 10% 样本的 median
差，并对全部样本做 least-squares slope，分别报告 MiB/1000 env-step。峰值不能
代替趋势，趋势也不能代替峰值。

## 6. 数值 Gate P1

### 6.1 生命周期与控制

| 条件 | PASS 阈值 |
|---|---|
| L0 first observation | 进程启动后 `<=180 s` |
| L0 shutdown | 请求后 `<=60 s` 且 exit code 0 |
| articulation | 初始化成功且所有正式 case 中失效次数为 0 |
| 直行 | `[1,0]` 共 5 sim s 后 `+X_B >=1.0 m`；横向漂移 `<=0.25 m`；yaw 漂移 `<=10°` |
| 左转 | `[0,1]` 共 4 sim s 后 yaw 增量 `>=90°`；root XY 漂移 `<=0.50 m` |
| 急停 | 2 sim s 后 root speed `<0.05 m/s`、yaw rate `<0.05 rad/s`、每轮 `<0.10 rad/s` |
| action timing | 每个 action 恰好保持 6 physics step，错一帧即失败 |

直行/转向测试包含 Task-v0 的 slew limit，不允许为通过阈值绕过 action adapter。

### 6.2 终止、reset 与确定性

| 条件 | PASS 阈值 |
|---|---|
| T0 | 五种 reason 各 10/10 正确，`terminated/truncated` 无混淆 |
| 静止/轮地接触 | 各 1000 physics step，collision false positive 为 0 |
| chassis 碰撞 | 10/10 在 `>20 N` 连续 2 physics step 后触发，轮接触不触发 |
| R0 完成度 | 500/500 成功；期间完整 `World.reset()` 次数为 0 |
| reset 残留 | applied action/drive target/counter/contact debounce 全为 0 |
| reset 速度 | root linear/angular `<0.05`，每轮 `<0.10 rad/s` |
| reset pose | settle 后目标 XY/yaw 误差各 `<=0.02 m/rad` |
| reset latency | p95 `<=3.0 s` wall time，settle 永不超过 60 physics step |
| D0 离散结果 | start/goal、termination reason 和 episode boundary 完全一致 |
| D0 数值轨迹 | 三次运行 root position/yaw max error `<=1e-3 m/rad`，reward max error `<=1e-4` |
| 非 finite | action、observation、reward、pose 中出现次数为 0 |

### 6.3 吞吐、稳定性与内存

| 条件 | PASS 阈值 |
|---|---|
| B0 `step_core` | 三次重复 median `>=20 env-step/s` |
| B1 `rollout_with_reset` | 三次重复 median `>=5 env-step/s`，即 RTF `>=1.0` |
| B1 step latency | p99 `<=1.0 s` |
| 重复噪声 | B1 三次吞吐 coefficient of variation `<=15%` |
| S0 | 100,000/100,000 transition，无 crash、hang、traceback、invalid articulation |
| CPU RSS peak | `<=24 GiB` |
| GPU memory peak | `<90%` total，至少保留 10% headroom |
| RSS/GPU growth | 后 10% median − 前 10% median 各 `<=512 MiB` |
| RSS/GPU slope | 各 `<=1 MiB / 1000 env-step` |

任一功能、reset、非 finite 或稳定性条件失败，P1 Gate 直接失败。B2/B3 没有绝对
吞吐硬门槛，但必须报告相对 B1 的倍率和资源增量。

## 7. 吞吐量决策区间

以 B1 `rollout_with_reset` 三次 median 为唯一分档值：

| 档位 | 条件 | P1 决策 |
|---|---|---|
| Green | `>=50 env-step/s`，RTF `>=10`，单进程 `T_100M<=23.2 d` | 原生路线 Go；P2 后可进入并行/算法评估 |
| Yellow | `[5,50) env-step/s` | P1 功能 Gate 可通过，P2 可继续；进入 P3 前必须优化、多环境/多进程或评估 off-policy |
| Red | `<5 env-step/s`，或任一硬 Gate 失败 | No-Go；先修稳定性/性能，仍失败则启用 RLRoverLab 预训练回退路线 |

禁止把 B0、去掉 reset 的 step_core、B2/B3 之外的另一个场景，或三次中的最好值
用于分档。报告还必须同时给出 1M/100M 样本墙上时间，便于以后调整规模判断。

## 8. 冻结产物结构

每次执行写入新目录，不覆盖历史结果：

```text
data/benchmarks/p1/<UTC timestamp>_<commit>/
├── manifest.json
├── resolved_config.yaml
├── git_status.txt
├── git_diff.patch              # clean 时可为空
├── environment_versions.txt
├── host_preflight.txt
├── gpu_samples.csv
├── process_samples.csv
├── transitions.csv             # 可分块/压缩
├── episodes.jsonl
├── summary.json
├── summary.md
├── stdout.log
├── checksums.sha256
└── visual/
    ├── trajectory.png
    ├── controls_rewards.png
    ├── straight.png
    ├── turn.png
    ├── collision.png
    └── success.png
```

`manifest.json` 至少包含协议版本、commit/diff hash、seed、case、GPU、所有 dt、
warmup/measured 数、terrain/robot USD、sensor flags、start/end UTC、退出码和 validity
状态。`summary.json` 保存原始数值，不从 Markdown 反向解析。

V0 的可视化必须让评估者同时看到地形、车辆、起点、目标、目标朝向和轨迹；
`trajectory.png` 标出 termination reason，`controls_rewards.png` 使用同一仿真时间轴
绘制 action、实际 v/omega、距离及 reward components。视觉产物用于发现坐标反号、
碰撞漏检和 reward hacking，不参与吞吐量计时。

## 9. P1 最终报告要求

P1 结束时必须给出：

1. 每条硬 Gate 的 PASS/FAIL 与原始产物路径；
2. B0/B1 三次独立结果、B2/B3 相对开销和 S0 内存趋势；
3. 1M/100M 样本时间估算；
4. Green/Yellow/Red 书面结论；
5. 若非 Green，明确选择优化、并行、off-policy 评估或回退路线；
6. 失败类型和下一步，不得只给平均 reward 或一段视频。
