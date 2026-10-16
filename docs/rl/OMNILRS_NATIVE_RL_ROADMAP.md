# OmniLRS-native RL 技术路线与项目宪章

> 状态：Active
> 当前阶段：P1 已完成（PASS / Yellow）；下一阶段为 P2 MDP 基线复现
> 最后更新：2026-10-16
> 适用 worktree：`/home/spatial/OmniLRS-isaac6`
> 开发分支：`feature/isaac6-native-rl`

本文档是后续所有实现窗口的共同依据。代码能运行并不等于项目成功；任何实现都必须能追溯到本文定义的最终目标、阶段 Gate 和接口边界。

---

## 1. 不可偏离的最终目标

在 OmniLRS 中建立原生、高效、可评测的强化学习环境，训练月球车的**局部导航与运动决策策略**；随后将该策略通过 ROS 2 接入由感知、SLAM、地图和全局规划组成的完整自主系统，最终完成月面环境中的建图、导航和安全行进。

目标系统的数据流是：

```text
深度相机 / 点云 / IMU / 轮速
              │
              ├── SLAM ──> 位姿 + 全局/局部地图
              │                       │
任务目标 ──> 全局规划 ──> 局部目标 ──┤
                                      v
                      局部高程/可通行性表示
                                      │
                                      v
                            RL 局部导航策略
                                      │
                         线速度 + 角速度/曲率
                                      │
                                      v
                         确定性底盘控制器 ──> 月球车
```

### 1.1 RL 负责什么

- 根据局部地形、车辆状态和局部目标，输出短时域运动命令。
- 学习绕障、越障、坡面行驶、避免高风险地形和减少无效振荡。
- 在传感器噪声、控制延迟、物理参数变化和有限定位误差下保持鲁棒。

### 1.2 RL 不负责什么

- 不把端到端 RL 当作 SLAM；SLAM 独立生成位姿和地图，并独立评测。
- 不让局部 RL 代替任务规划或全局路径规划。
- 第一阶段不直接学习各个车轮的底层力矩；策略先输出车辆级速度/曲率命令。
- 不在训练内环通过 ROS 2 往返采样。
- 当前阶段不处理真实月球车 sim-to-real；但接口和观测不得依赖真实系统无法获得的永久特权信息。

### 1.3 项目成功的定义

项目不是以“PPO 能启动”为完成标准，而是同时满足：

1. OmniLRS 中存在稳定、可复现、可自动 reset 的 RL 环境。
2. 策略能在未见过的月面地形上优于明确基线。
3. actor 最终只使用可由深度/点云、IMU、轮速、SLAM 位姿和规划器提供的观测。
4. 策略能在 OmniLRS 的传感器、光照和选定月壤/地形物理配置下闭环运行。
5. ROS 2 模式下，SLAM、规划、策略和底盘控制能够完成端到端任务。

---

## 2. 已冻结的架构原则

### 2.1 OmniLRS-native，而不是复制 RLRoverLab

复用 RLRoverLab 的思想与可验证配置：

- MDP 分解方法；
- 奖励项和 curriculum 思路；
- PPO/skrl 配方及网络结构作为基线；
- 日志、视频和评测指标。

在 OmniLRS 中重新实现：

- 仿真生命周期和 `reset/step`；
- 机器人控制与状态读取；
- 高程图/点云处理；
- 碰撞、终止和 episode 管理；
- 并行环境与域随机化。

RLRoverLab 是参考基线，不是运行时依赖。不要因追求源码一一对应而破坏 OmniLRS 原生结构。

### 2.2 训练内环不使用 ROS 2

训练路径：

```text
RL algorithm -> Python Env -> OmniLRS/Isaac Sim API -> physics step
```

部署与系统集成路径：

```text
ROS 2 sensors/SLAM/planner -> policy node -> vehicle controller
```

原因是 ROS 2 消息序列化、异步调度和实时发布会限制高速采样，也会使 reset 与多环境同步变复杂。ROS 2 只从 P6 开始进入闭环验证。

### 2.3 使用“特权到真实可用”的观测课程

```text
真值位姿 + 几何查询高度图
        ↓
真值位姿 + 深度/点云生成局部地图
        ↓
带噪位姿 + 传感器局部地图
        ↓
SLAM 位姿 + 在线局部地图
        ↓
完整 ROS 2 闭环
```

真值用于早期算法排错、训练标签和最终误差评估，不是最终 actor 的永久输入。可采用 privileged critic、teacher-student 或蒸馏，但必须记录 actor 与 critic 各自能看到的变量。

### 2.4 单一版本基线

OmniLRS-native RL 的唯一开发目标已确定为：

- worktree：`/home/spatial/OmniLRS-isaac6`；
- 开发分支：`feature/isaac6-native-rl`；
- 仿真基线：Isaac Sim 6.1；
- `/home/spatial/OmniLRS` 仅作为官方 `main` 参考 worktree，不承载 RL 开发。

P0 已用可复现命令记录 Isaac Sim、Python、CUDA、驱动及关键扩展的精确版本。
RLRoverLab 使用的 Isaac Sim/Isaac Lab 组合只能作为参考，不应与 OmniLRS Python
进程混装。

### 2.5 必须保留止损路线

若 OmniLRS 无法达到可接受的稳定性或采样吞吐量，不进行无上限重构。保留：

```text
RLRoverLab 大规模预训练
          ↓
统一 actor 观测/动作接口
          ↓
OmniLRS 高保真评测与少量微调
```

采用这条回退路线不改变最终 OmniLRS + ROS 2 系统目标，只改变样本主要从哪里获得。

---

## 3. 稳定接口契约

接口版本发生变化时必须更新本文、配置版本和 checkpoint 元数据；禁止静默改变张量顺序、单位或坐标系。

### 3.1 环境 API

采用 Gymnasium 语义：

```python
observation, info = env.reset(seed=seed, options=options)
observation, reward, terminated, truncated, info = env.step(action)
```

- `terminated`：成功、碰撞/倾覆、驶出有效区域等任务终止。
- `truncated`：超时等外部截断。
- 每次 reset 必须清零控制命令、刚体和关节速度、历史观测、奖励缓存、接触状态及 episode 统计。
- 优先局部 reset，不在每个 episode 调用完整 `World.reset()`。

### 3.2 时序基线

初始候选值：

- physics `dt = 1/30 s`；
- action decimation `= 6`；
- policy period `= 0.2 s`。

P0 已冻结这些数值；P1 的 L0/C0 动态 smoke 必须验证实际控制和物理稳定性。
如果动态证据否定候选值，必须更新规格和决策日志。任何修改都要同时说明物理
时间、策略频率和传感器频率。

### 3.3 坐标、单位与角度

P0 必须显式记录：

- 世界坐标系、车体坐标系、相机坐标系和 ROS 坐标系；
- 前/左/上方向和旋转正方向；
- 长度为米、角度内部统一为弧度、速度为 SI 单位；
- 高程图原点、分辨率、覆盖范围、无效值语义；
- 时间戳来自仿真时钟还是墙上时钟。

### 3.4 Observation 版本

`Observation-v0`：只用于 MDP 与学习基线。

- 上一步动作：2；
- 目标距离：1；
- 车体坐标系下目标方位：1；
- 航向误差：1；
- 以车辆为中心、朝向对齐的 `101 × 101` 理想局部高度图；
- 所有项必须归一化，并有独立的 shape/range 测试。

`Observation-v1`：加入动力学与可控性信息。

- 车体线/角速度；
- roll、pitch 或重力投影；
- IMU、轮速、转向角；
- 命令跟踪误差；
- 可选滑移估计与短历史。

`Observation-v2`：最终 actor 候选。

- 由深度/点云在线生成的局部 elevation/traversability map；
- 有效性 mask 或不确定度；
- SLAM/里程计提供的相对局部目标与运动状态；
- 不包含不可在部署侧获得的真值量。

### 3.5 Action 版本

`Action-v0`：归一化二维动作，经限幅和确定性映射后得到：

- 期望线速度 `v [m/s]`；
- 期望角速度 `ω [rad/s]` 或曲率。

车辆控制器负责转换为轮速/转角。必须记录饱和、加速度限制、命令延迟和急停行为。只有车辆级控制被证明不足时，才另立实验评估关节级动作。

### 3.6 Reward-v0

奖励由可单独记录的分量组成：

- 向目标的势函数进度；
- 到达目标奖励；
- 目标朝向或路径对齐；
- 碰撞、倾覆、越界惩罚；
- 过大 roll/pitch、危险坡度或高滑移惩罚；
- 动作变化、反复转向和停滞惩罚；
- 可选能耗/时间代价。

原则：先实现最少奖励项并做单元测试；每增加一项都要说明它解决的失败模式。禁止只看总 reward，日志必须输出各分量。

### 3.7 `info` 最小字段

```text
episode_id, seed, terrain_id, success, termination_reason,
distance_to_goal, path_length, elapsed_sim_time,
collision_count, max_roll, max_pitch, mean_slip,
reward_components, observation_version, action_version
```

P0 已冻结的完整数值契约、坐标系、reset 顺序和测试矩阵见
[`task_v0_spec.md`](task_v0_spec.md)。若本文概述与该规格的 Task-v0 数值细节冲突，
P1 实现以该规格为准，并通过决策日志修订，禁止静默选择第三种行为。

---

## 4. 建议代码边界

在确认仓库现有模块约定后，优先形成以下职责，而不是先写一个巨型训练脚本：

```text
src/environments_wrappers/rl/
├── env.py                 # Gymnasium 生命周期；不包含算法
├── env_cfg.py             # 环境、时序、观测与终止配置
├── robot_adapter.py       # OmniLRS 机器人状态/命令适配
├── observations.py        # 真值、几何和传感器观测
├── rewards.py             # 纯函数式奖励分量
├── terminations.py        # 成功、碰撞、倾覆、越界、超时
├── reset.py               # 起终点采样与局部 reset
├── terrain_sampler.py     # train/val/test 地形划分
└── metrics.py             # episode 与 benchmark 指标

scripts/rl/
├── smoke_env.py
├── benchmark_env.py
├── train.py
└── evaluate.py

test/rl/
├── test_observations.py
├── test_rewards.py
├── test_terminations.py
├── test_reset.py
└── test_determinism.py
```

实际路径可适配仓库结构，但职责边界不变：环境不依赖 PPO，策略不直接操作 USD 场景，传感器处理不混入奖励函数。

---

## 5. 分阶段实施路线与 Gate

规则：前一阶段 Gate 未通过，不进入后一阶段主开发。可以做只读调研或小型风险探针，但不得让其取代当前阶段交付物。

## P0：冻结版本、任务和测量方法

### 目标

消除版本和接口歧义，让后续结果可复现。

### 工作项

- 盘点 OmniLRS、Isaac Sim、Python、CUDA、驱动、GPU、关键扩展版本。
- 选择唯一训练基线与唯一启动命令；保存最小 headless smoke 日志。
- 画清机器人控制接口、状态接口、碰撞接口、传感器接口和 reset 路径。
- 冻结 Task-v0：起点/终点采样、成功半径、超时、碰撞/倾覆/越界定义。
- 冻结 Observation-v0、Action-v0、Reward-v0、坐标系和单位。
- 冻结 benchmark 协议：场景、seed、预热步数、测量步数、是否渲染、传感器开关、GPU 指标和内存采样方法。
- 建立决策日志，不改 PPO 代码。

### 交付物

- `docs/rl/baseline_environment.md` 或等价文档；
- 版本锁定结果与可复现启动命令；
- `Task-v0` 接口规格；
- benchmark 规格和 P1 的明确验收阈值。

### Gate P0

- 同一命令可重复启动、加载场景、步进并退出。
- 不再存在“到底用哪个 OmniLRS/Isaac Sim 环境”的分歧。
- 所有张量 shape、单位、坐标系和终止语义可以由测试直接验证。

**P0 结论（2026-10-14）：PASS。** Isaac 6.1/Pixi 基线和 headless 场景 smoke
已有复现证据；Task/Observation/Action/Reward/reset 契约见
[`task_v0_spec.md`](task_v0_spec.md)；测量方法与 P1 数值 Gate 见
[`p1_benchmark_protocol.md`](p1_benchmark_protocol.md)。机器人运动、碰撞和局部
reset 的动态证据属于 P1 的首批交付物，不能由 P0 文档结论替代。

## P1：单车 Gym 环境与可行性基准

### 目标

证明 OmniLRS 能作为稳定的 RL 采样器；本阶段不训练 PPO。

### 工作项

- 实现最小 `OmniLRSEnv.reset/step`。
- 用随机动作和脚本动作测试前进、转弯、停止、碰撞和成功。
- 实现局部 reset：teleport、清速度、清控制、重采样目标、清缓存。
- 分别测试 GUI、headless、`render=False`；训练默认 headless。
- 记录 aggregate env-steps/s、real-time factor、GPU/CPU 使用、显存和常驻内存趋势。
- 依次测试：理想状态；几何高度图；相机；地形变形。不要一次全部开启。
- 对固定 seed 验证 reset 与前若干步的可重复性。

### Gate P1

- 随机策略可连续完成分阶段长跑，不崩溃、不发生未受控内存增长、不出现失效 articulation。
- 至少数百次局部 reset 后控制和状态无残留。
- 所有终止原因可被脚本触发并正确区分。
- 产出完整吞吐量表，据此估算一百万和一亿样本的墙上时间。
- 根据实测结果作出书面 Go/No-Go：原生训练、先优化/并行，或启用回退路线。

**P1 最终结论（2026-10-16）：PASS / Yellow。** 最小 native Gymnasium 生命周期、
Husky adapter、局部 reset 及 headless L0/C0 已在开发 GPU 6 上通过；直行、正
omega 左转、急停、每 action 恰好 6 个 physics step 和 articulation 生命周期均有
动态证据。Task-v0 runtime 已在物理 GPU 3 上接入真实 Lunalab DEM、Reward-v0、
termination 和 12 个具名 Husky contact view；7-step `render=False` smoke exit 0，
Observation-v0 shape/dtype、6-step decimation、前进距离进度和正常轮接触不误终止通过。
正式 T0 已在物理 GPU 3 上通过：五种 reason 各 10/10，静止与轮地接触各 1002
physics step 无 collision false positive，chassis-scrape 10/10，timeout 精确发生在第
300 policy step。固定 `rock4` 独立撞岩也已由过滤接触 view 确认：`base_link`
52.902455 N 连续 2 physics step 后正确 collision 并急停。V0 也已在物理 GPU 3 上
完成同进程局部 reset 的直行、左转、撞岩和成功四条固定脚本轨迹；四个 case 均保持
articulation ready 与精确 6-step decimation，场景、俯视轨迹和统一时间轴控制/奖励图
均已保存。B0 的冻结 workload、自动 GPU 选择、独立 1 Hz 资源监控和三 seed
聚合器、PID 显存归因及运行期资源争用哨兵已经实现；GPU 4 共享动态预检覆盖 310 measured
step 和一次 time-limit→局部 reset，标记为 `diagnostic_shared`。GPU 4 在外部进程
占用约 29.5 GiB 时的显存复测也 exit 0：B0 相对启动前增加约 3.19 GiB，峰值仍余
约 15.95 GiB。按 PID 归因后的完整长度共享 B0 也已完成：两个 10000 measured
workload 均 exit 0，项目显存峰值 3.157 GiB，33 次局部 reset，全部正确性错误计数为
0；运行中新外部 PID 的 5024 MiB 被单独归类。随后按修订协议完成 seed 42/43/44
三个全新进程的完整共享保守 B0：自动选择 GPU 6/6/4，step-core median/min/max 为
30.031241/28.283701/32.959477 env-step/s，CV 6.340%，超过冻结的 20 env-step/s 门槛；
rollout median 为 26.756761 env-step/s，三次共完成 99 次局部 reset，全部正确性错误
计数为 0，项目 PID 显存峰值 3232.988 MiB。因此 B0 判定
`shared_conservative` PASS；独占性能基线和卡级资源结论仍未执行。
完整 B1 也已按修订协议完成 seed 42/43/44 三进程共享保守聚合：完整
Task-v0/Observation-v0 rollout median/min/max 为
41.203994/41.184755/42.486459 env-step/s，CV 1.463%，三次最坏 p99 为 0.039680 s；
共完成 129 次局部 reset，所有正确性错误为 0，项目 PID 显存峰值 3232.988 MiB。
B1 Gate 判定 `shared_conservative` PASS；按冻结分档属于 Yellow，进入 P3 前仍需优化、
并行化或评估 off-policy。
R0 修复前的完整运行发现 15/500 次 settle 后
pose 超过冻结上限；增加“保留 settled Z/roll/pitch、回投 XY/yaw、再次清零且不追加
physics step”的末端处理后，短探针 50/50 通过，完整 R0 也以 500/500、10000
transitions、0 residual failure 通过；最大 XY/yaw 误差分别为 `1.192093e-7 m` 和
`2.556042e-10 rad`，0 次完整 `World.reset()`。D0 harness 也已完成三进程、每进程
1000-step 的共享诊断：episode boundary/reason 完全一致，三对 root position/yaw/reward
最大误差均为 0。D0 只使用离散/数值正确性指标，项目显存按 PID 归因，因此共享
GPU 不影响该 Gate，D0 判定 PASS；共享 wall-time 和卡级指标不进入性能结论。
正式 S0 也已在共享 GPU 上完成 100000/100000 measured transition：452 次局部 reset，
所有正确性错误为 0；RSS 首尾增长 58.945 MiB、斜率 0.687 MiB/1000 step，项目 PID
显存增长和斜率均为 0，全部稳定性 Gate 通过。B2 也已完成冻结的 100/1000 共享成本
诊断：1000/1000 RGB/depth 帧有效，吞吐 14.643520 env-step/s，是同卡共享 B1 seed 42
的 0.355557×；项目 PID 显存/RSS 峰值近似增加 72.250/151.840 MiB。B3 也已完成冻结
共享诊断：1000/1000 形变成功，吞吐 23.515312 env-step/s，是同卡共享 B1 seed 42 的
0.570971×；项目 PID 显存/RSS 峰值近似增加 19.875/17.434 MiB。完整 P1 判定为
`PASS / Yellow`。L0/C0 证据见
[`p1_l0_c0_results.md`](p1_l0_c0_results.md)。Task/Observation/Reward/termination
纯函数层证据见 [`p1_task_contract_results.md`](p1_task_contract_results.md)；runtime
接线、39 项累计单元测试与 GPU 3 短程动态证据见
[`p1_task_runtime_results.md`](p1_task_runtime_results.md)；单次动态终止证据见
[`p1_termination_probe_results.md`](p1_termination_probe_results.md)，正式 T0 证据见
[`p1_t0_results.md`](p1_t0_results.md)，独立撞岩证据见
[`p1_rock_collision_results.md`](p1_rock_collision_results.md)，V0 证据见
[`p1_v0_results.md`](p1_v0_results.md)，B0 harness 状态见
[`p1_b0_harness_results.md`](p1_b0_harness_results.md)，B1 证据见
[`p1_b1_results.md`](p1_b1_results.md)，R0 证据见
[`p1_r0_results.md`](p1_r0_results.md)，D0 诊断证据见
[`p1_d0_results.md`](p1_d0_results.md)，S0 长跑证据见
[`p1_s0_results.md`](p1_s0_results.md)，B2 诊断证据见
[`p1_b2_results.md`](p1_b2_results.md)，B3 诊断证据见
[`p1_b3_results.md`](p1_b3_results.md)，最终 Gate 汇总见
[`p1_final_report.md`](p1_final_report.md)。

当前 RL 单元测试为 68/68 通过，Gymnasium 1.4.0 可从 Pixi default 导入，
`git diff --check` 通过。B1 三次完整运行对应的 Isaac
Kit 日志为 `.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_001848.log`、
`kit_20261016_002337.log` 和 `kit_20261016_002823.log`。

## P2：复现 RLRoverLab 的 MDP 基线

### 目标

在不引入真实传感器和复杂土壤的情况下，验证任务、观测、奖励与 reset 正确。

### 工作项

- 在平地和简单程序化坡地上实现 Observation-v0 / Action-v0 / Reward-v0。
- 为每个奖励项、终止条件和归一化写单元测试。
- 建立手工构造的 golden cases：向目标靠近、远离、原地振荡、碰撞、到达、翻覆。
- 加入零动作、恒定前进、目标指向控制器、随机动作等非学习基线。
- 生成 episode 轨迹、reward 分量曲线和少量验证视频。

### Gate P2

- golden cases 的奖励符号和终止结果全部符合规格。
- 简单目标指向控制器明显优于随机策略，证明环境信号不是反的。
- 固定 seed 的回归测试可复现。
- 环境层不导入训练算法，且可单独运行测试。

## P3：学习基线与并行方案决策

### 目标

证明学习确实发生，并基于数据选择训练架构。

### 工作项

- 先用与 RLRoverLab 接近的 PPO/skrl 配置建立可比基线。
- 记录 learning curve、成功率、碰撞率、episode 长度和策略熵。
- 保存配置、代码提交、seed、版本、归一化统计和 checkpoint 元数据。
- 评估三种扩展方式：单场景多 rover、多个 headless 进程、少环境的 off-policy 算法。
- 不以单个 seed 或训练地形上的最高 reward 作为结论。

### 决策原则

- 能稳定运行数十个并行环境：优先 PPO。
- 只能稳定运行少量环境且单样本昂贵：比较 SAC/TD3 等 off-policy 方法。
- 单场景多 rover 出现传感器、碰撞组或 reset 污染：转向多进程采样。
- 若预计训练时间不可接受：启用 RLRoverLab 预训练 + OmniLRS 微调/评测路线。

### Gate P3

- 至少多个 seed 在简单未见地形上稳定优于所有非学习基线。
- 训练、恢复训练和独立评测都可复现。
- 形成有测量数据支持的算法与并行架构决策。

## P4：OmniLRS 月面几何与轮壤课程

### 目标

把已验证的局部导航策略扩展到 OmniLRS 的月面几何和选定物理模型。

### Curriculum

1. 平地、小坡度、稀疏小障碍；
2. 坡度、坑缘、岩石密度逐步增大；
3. 真实或程序化 DEM，训练/验证/测试区域严格分离；
4. 摩擦、质量、质心、控制增益、延迟和扰动随机化；
5. 在确认数值稳定且有测量意义后启用地形变形/轮迹与轮壤参数变化。

### 物理边界

“OmniLRS 提供月壤参数或可变形地形”不自动等于物理真实。必须记录所用模型、参数来源、适用尺度和已知缺陷。现阶段目标是在 OmniLRS 模型内获得鲁棒策略，而不是宣称已解决真实月壤 sim-to-real。

### Gate P4

- 在冻结的 unseen terrain test set 上达到预先规定的成功率与安全指标。
- 对主要物理参数扰动给出敏感性曲线，而非只报告平均 reward。
- 能识别失败类型：困住、打滑、翻覆、碰撞、规划死锁、超时。

## P5：从理想高度图迁移到深度/点云

### 目标

让 actor 使用 OmniLRS 可生成、ROS 2 可重建的感知表示。

### 工作项

- 从 OmniLRS 深度相机/激光数据构建车体坐标系局部点云。
- 建立点云过滤、地面/障碍处理、elevation map、有效性 mask 和时间同步。
- 用仿真真值计算地图误差，但不把真值输入最终 actor。
- 依次加入量化、噪声、空洞、遮挡、尘埃/低照导致的缺测、延迟和外参误差。
- 比较三条路线：直接训练 sensor actor；teacher-student 蒸馏；privileged critic。
- 保持 Observation-v2 的离线/在线预处理一致，并记录推理延迟。

### Gate P5

- 使用 sensor-derived map 的策略在未见场景中达到预设性能保留率。
- 断开真值观测后 actor 仍可运行。
- 地图生成和策略推理满足目标控制周期，并通过固定数据包回归测试。

## P6：SLAM、地图与规划汇合

### 目标

将 RL 局部控制器放回完整自主导航栈；SLAM 主要是应用、集成和评测对象，不默认用 PPO 端到端训练。

### 工作项

- 使用 OmniLRS 相机、深度/点云、IMU 和轮速数据运行候选 SLAM。
- 用仿真真值作为 ground truth 评估 ATE、RPE、漂移和失跟率。
- 构建/更新全局地图，运行全局规划器，并按固定接口生成局部目标。
- 训练时逐步向相对目标和局部地图注入与实测 SLAM 相符的误差。
- 评估重定位、短时丢失、地图更新和规划重算对 RL 策略的影响。
- 清晰区分：SLAM 失败、全局规划失败、局部策略失败和底盘跟踪失败。

### Gate P6

- 不使用仿真真值位姿作为控制输入，也能在代表性任务中闭环到达。
- SLAM 误差、规划性能和局部策略性能分别有独立指标。
- 回放同一传感器数据可复现 SLAM/地图输入与策略输出。

## P7：ROS 2 完整闭环与系统验证

### 目标

建立与未来实验室系统一致的消息边界，同时保留训练内环的直接 Python API。

### ROS 2 边界

- 输入：传感器、TF、里程计/SLAM 位姿、地图、全局路径或局部目标；
- 策略节点：执行与训练一致的预处理、归一化和推理；
- 输出：带时间戳和限幅的车辆级速度/曲率命令；
- 安全层：超时、无效地图、失定位、碰撞风险和急停处理；
- rosbag：记录可离线重放的最小消息集合。

### Gate P7

- 完整 ROS 2 闭环完成冻结的建图—规划—导航任务集。
- 策略节点输出与直接 Python 推理在相同输入上数值一致。
- 验证实时性、时间同步、掉帧、节点重启和失定位恢复。
- 所有系统级失败都有明确归属和日志证据。

## P8：实验室 sim-to-real（暂缓）

### 目标

在月面模拟实验室中验证传感器、定位、执行器、轮壤作用和安全性差异。

当前只保留接口兼容性和数据记录要求，不提前声称已解决：

- OmniLRS 到真实传感器的 domain gap；
- 仿真车辆到底盘执行器、延迟和回差；
- 仿真轮壤模型到实验室材料；
- 实车安全认证和操作流程。

进入 P8 前另立实车测试计划、停止条件和安全审查。

---

## 6. 统一评测协议

### 6.1 数据划分

- Train：允许 curriculum 与随机化。
- Validation：用于选 checkpoint 和调超参数，不进入训练。
- Test：冻结后只做正式评测；地形/DEM、岩石布局和 seed 不与训练重复。
- Stress：超出训练分布，用于刻画失效边界，不混入主成功率。

### 6.2 核心指标

- 成功率；
- 碰撞率、倾覆率、越界率和超时率；
- SPL 或路径效率；
- 到达时间与路径长度；
- 最大/均方 roll、pitch；
- 滑移、停滞时间、控制振荡和能耗代理；
- 分地形难度与分参数区间结果；
- 仿真吞吐量、训练墙上时间、推理与预处理延迟；
- P6 后增加 ATE、RPE、失跟率、规划重算次数。

### 6.3 必备对照组

- 随机策略；
- 零动作；
- 简单目标指向/反应式控制器；
- 传统局部规划器（进入系统集成后）；
- RLRoverLab 配方或 checkpoint（仅在接口可比时）。

所有“提升”必须说明相对哪个对照、在哪个固定测试集、多少个 seed、置信区间或离散程度如何。

---

## 7. 风险登记与触发条件

| 风险 | 早期证据 | 应对与触发动作 |
|---|---|---|
| Isaac Sim/OmniLRS 版本冲突 | import/extension 错误、行为随环境变化 | P0 只保留一个训练基线，禁止混装 |
| reset 污染 | 速度、接触、命令或传感器历史跨 episode | 为 reset 写状态断言；必要时分层重建对象 |
| 采样吞吐量不足 | 训练时间估算不可接受 | 关渲染、降低传感器频率、多 rover/多进程、off-policy 或回退路线 |
| 多机器人互相污染 | 碰撞组、地形、传感器或 reset 干扰 | 空间/碰撞隔离；失败则多进程实例 |
| reward hacking | reward 上升但成功率/安全性不升 | 分量日志、脚本基线、视频、冻结测试指标 |
| 策略依赖真值 | 替换为传感器/SLAM 后性能崩溃 | P5 前完成观测审计，采用蒸馏/privileged critic |
| 月壤模型过度解读 | 参数无来源却宣称真实 | 记录模型边界；把参数变化当鲁棒性范围而非现实证明 |
| SLAM 与 RL 责任混淆 | 用 RL reward 掩盖定位失败 | P6 分开指标、日志与故障分类 |
| ROS 2 拖慢训练 | 每步等待消息或 wall clock | ROS 2 只用于 P6/P7；训练继续直接 API |
| 只在训练地形有效 | test 成功率显著下降 | terrain split、多个 seed、stress test |

---

## 8. 防偏离规则

后续每个窗口必须遵守：

1. 开始前先阅读本文，声明当前 Phase、唯一交付物和 Gate。
2. 一次窗口只承担一个可验收的主交付物；额外想法写入 backlog。
3. 前一 Gate 未通过，不把后续阶段功能并入主实现。
4. 不让 ROS 2 进入训练采样内环。
5. 不把 RL 局部导航称为 SLAM，也不让它代替全局规划。
6. 最终 actor 不得静默依赖真值位姿、真值地形或其他部署不可用信息。
7. 不为追求吞吐量而改变物理时间却不记录；所有频率必须以仿真时间说明。
8. 不以视频“看起来会动”或单次 reward 曲线作为完成证据。
9. 不覆盖已有用户修改；开始和结束都检查 Git 状态。
10. 新依赖、新版本、接口变更和重要假设必须写入决策日志。
11. 每次实现必须有对应的测试、benchmark 或可复现日志。
12. 如果实际证据否定当前路线，更新本文并记录理由，而不是悄悄改变目标。

### 需求追踪表

每个实现 PR/提交/窗口至少关联一个编号：

| ID | 需求 | 首次验证阶段 |
|---|---|---|
| G-01 | OmniLRS 原生 reset/step | P1 |
| G-02 | 稳定且可测量的采样吞吐量 | P1 |
| G-03 | 正确的导航 MDP | P2 |
| G-04 | 学习策略优于非学习基线 | P3 |
| G-05 | 未见月面地形与物理扰动鲁棒性 | P4 |
| G-06 | actor 使用深度/点云派生观测 | P5 |
| G-07 | SLAM 位姿替代真值位姿 | P6 |
| G-08 | SLAM、规划、RL 与控制闭环 | P6/P7 |
| G-09 | ROS 2 部署接口与回放一致性 | P7 |
| G-10 | 实验室 sim-to-real | P8，当前暂缓 |

---

## 9. 决策日志

只记录会影响架构、复现或阶段结论的决定。

| 日期 | 决策 | 状态 | 依据/证据 |
|---|---|---|---|
| 2026-10-13 | 采用 OmniLRS-native RL，RLRoverLab 仅作 MDP/算法参考 | 已接受 | 避免把跨仿真器迁移作为主路线 |
| 2026-10-13 | 训练内环使用 Python API，ROS 2 留给系统集成 | 已接受 | 吞吐量、同步与 reset 可控性 |
| 2026-10-13 | 先用理想观测验证，再迁移到深度/点云和 SLAM | 已接受 | 分离控制、感知和定位错误 |
| 2026-10-13 | 真实月球车 sim-to-real 暂缓到实验室阶段 | 已接受 | 当前集中验证仿真内完整自主系统 |
| 2026-10-13 | 使用 `/home/spatial/OmniLRS-isaac6`、Isaac Sim 6.1 和 `feature/isaac6-native-rl` 作为唯一 RL 开发基线 | 已接受 | 用户确认实际 OmniLRS 已在 Isaac 6 上实现，RL 也将在该环境部署 |
| 2026-10-14 | 使用仓库 `pixi.toml`、`pixi.lock` 与 `pixi run -e default` 作为唯一 Python/Isaac 启动入口；Kit 缓存放入 `.cache/isaacsim` | 已接受 | Python 3.12.15 / Isaac Sim 6.1.0.0 可安装；GPU 4 headless Lunalab RGB+语义 smoke 退出码 0，证据见 `docs/rl/headless_smoke_isaac61.md` |
| 2026-10-14 | 冻结单车 Lunalab/Husky 的 Task-v0、Observation-v0、Action-v0、Reward-v0、坐标/单位及局部 reset 契约 | 已接受 | 源码和 USD 审计确认现有 articulation API 可作底层，但碰撞、车辆级动作和 reset 必须由原生 RL adapter 补齐；完整规格见 `docs/rl/task_v0_spec.md` |
| 2026-10-14 | 冻结 P1 benchmark protocol，并以物理 GPU 3 作为正式测量卡（运行前必须重新验证独占） | 已修订 | 2026-10-15 起不再固定 index；其余 3 seeds、10k-step 重复、100k-step 长跑、500 reset、资源采样与 Green/Yellow/Red 阈值保持不变 |
| 2026-10-14 | P1 首个实现切片采用最小 Gymnasium 环境 + 可配置差速/滑移转向 adapter；诊断观测明确版本化，不提前实现 PPO | 已接受 | Pixi default 单元测试 9 项通过；GPU 6 上 L0/C0 exit 0，+X_B 直行、正 omega 左转和急停均通过；正式 benchmark 未执行，见 `docs/rl/p1_l0_c0_results.md` |
| 2026-10-14 | Task-v0 采样、Observation-v0、Reward-v0 和 termination 先实现为无仿真依赖的纯函数层 | 已接受 | 32 项累计单元测试通过；数学契约可在接入 terrain/contact runtime 前独立验证，见 `docs/rl/p1_task_contract_results.md` |
| 2026-10-15 | Task-v0 以可注入 terrain/contact provider 接入 `OmniLRSEnv`；Husky 使用配置驱动的逐-link `RigidPrim` contact view | 已接受 | 39 项累计单测通过；GPU 3 上 7-step headless smoke exit 0，12 个 contact view ready，轮接触峰值 55.807585 N 未误终止，chassis 0 N，见 `docs/rl/p1_task_runtime_results.md` |
| 2026-10-15 | 正式 T0 前先以单次 pose/contact 注入探针验证动态 termination 链路，不用探针代替 10/10 Gate | 已接受 | GPU 3 单进程中 normal 及五种 termination reason 全部按冻结 Gym 语义通过，time-limit 在 step 300；见 `docs/rl/p1_termination_probe_results.md` |
| 2026-10-15 | 正式 T0 的独立 out-of-bounds 注入必须离地，避免粗糙 DEM 上同时碰撞按冻结优先级抢先终止 | 已接受 | 首轮 out-of-bounds 9/10，失败样本同时出现 13144.841023 N chassis contact；夹具隔离后五种 reason 各 10/10，静止/轮接触各 1002 physics step 零误报；见 `docs/rl/p1_t0_results.md` |
| 2026-10-15 | 独立撞岩使用固定真实 rock USD 和 link→rock Mesh 过滤接触 view，不用 root 位置代替碰撞来源证据 | 已接受 | GPU 3 上 `rock4` 探针 exit 0；`base_link` 岩石过滤接触 52.902455 N 连续 2 physics step，reason `collision` 且急停清零；见 `docs/rl/p1_rock_collision_results.md` |
| 2026-10-15 | V0 使用同一 Isaac 进程和局部 reset 回放直行、左转、撞岩、成功，并将场景、轨迹、控制和 Reward-v0 分量绑定到原始 CSV | 已接受 | GPU 3、seed 42 上四个 case 全部通过，articulation invalid 为 0，精确 6-step decimation；collision/success 分别正确终止，见 `docs/rl/p1_v0_results.md` |
| 2026-10-15 | B0 关闭可选的逐 physics-step 内存 trace，以独立 1 Hz NVML/psutil 进程测资源 | 已接受 | GPU 4 共享预检覆盖真实局部 reset 并明确排除于正式聚合；见 `docs/rl/p1_b0_harness_results.md` |
| 2026-10-15 | P1 正式运行不固定 GPU index；每个全新进程自动选择并连续 10 秒验证独占的 RTX 4090，三 seed 可使用不同空闲卡 | 已接受 | 用户确认“哪个空闲用哪个”；自动发现实测在 8 卡均繁忙时保存全量快照并 exit 3；runner 记录实际 index/UUID/型号，聚合器拒绝共享、非 RTX 4090 或缺少 formal-eligible 标记的结果 |
| 2026-10-15 | GPU 4 共享状态只用于显存可行性探针，不改变正式运行的自动独占要求 | 已接受 | 外部 PID `2114079` 占用约 29.5 GiB 时，B0 短测新增约 3.19 GiB、峰值仍余约 15.95 GiB，exit 0、无 OOM；结果标记 `diagnostic_shared`，见 `docs/rl/p1_b0_harness_results.md` |
| 2026-10-15 | 共享开发诊断运行中出现新外部 workload 时主动中止，不以受污染数据补齐正式结果 | 已修订 | 首次完整长度尝试因 PID `3248801` 加入而人工中止并保留 260 组样本；后续改为共享诊断继续、正式独占运行才自动中止 |
| 2026-10-15 | 项目显存改为按 benchmark 进程树的 NVML PID 去重统计；共享诊断遇到新 PID 继续运行 | 已接受 | 项目、其他进程和整卡显存分别记录；共享卡的项目显存仍可归因，吞吐保持非正式。正式独占运行遇到新 PID 仍以 exit 3 拒绝，见 `docs/rl/p1_benchmark_protocol.md` |
| 2026-10-15 | 用完整长度共享 B0 验证 PID 显存归因，不以共享吞吐代替正式三 seed 结果 | 已接受 | seed 42 两个 10000-step workload exit 0；项目 PID 峰值 3232.988 MiB，运行中新 PID `3291184` 的约 5024 MiB 被隔离；33 次局部 reset、正确性错误均为 0，见 `docs/rl/p1_b0_harness_results.md` |
| 2026-10-15 | R0 保持逐次 pose `<=0.02 m/rad` Gate，不以 485/500 或 p95 达标替代 | 已接受 | 500/500 已执行；yaw 超限 14 次、XY 超限 2 次、重叠 1 次，共 15 次失败。其他 reset 条件通过，R0 Gate FAIL，见 `docs/rl/p1_r0_results.md` |
| 2026-10-15 | 局部 reset 在速度 settle 后保留 Z/roll/pitch 并回投请求的 XY/yaw，再次清零动态状态且不追加 physics step | 已接受 | 先以 50/50 短探针验证，再以完整 500/500、10000 transitions、0 residual failure 通过 R0；冻结 `0.02 m/rad` Gate 未改，最大 XY/yaw 仅 `1.192093e-7 m`/`2.556042e-10 rad`，见 `docs/rl/p1_r0_results.md` |
| 2026-10-15 | D0 固定 start/goal 与 PCG64 float32 动作序列，terminal 后局部 reset 但不重启动作 index | 已修订 | GPU6 上三个全新进程各 1000-step，boundary/reason 完全一致，root position/yaw/reward 三对最大误差均为 0；D0 不使用卡级性能指标，共享运行可进入正确性 Gate，见 `docs/rl/p1_d0_results.md` |
| 2026-10-15 | GPU 独占优先用于 B0/B1 吞吐、B2/B3 相对成本及卡级资源结论；正确性 case 自动选择至少剩余 8 GiB 的最低占用卡 | 已修订 | 性能 case 后续增加共享保守双轨；PID 显存归因隔离项目与外部进程。共享 wall-time 可作保守吞吐证据，但整卡 utilization/功耗/温度不得归因给项目 |
| 2026-10-15 | 性能 case 增加 `shared_conservative` 双轨，不再因没有空卡而停工 | 已接受 | 共享三次完整重复若达到冻结阈值可保守 PASS；未达到只能 `inconclusive_shared`，不得作 No-Go。项目显存继续按 PID 归因，整卡指标保留但不归因给项目 |
| 2026-10-15 | B0 使用自动 GPU 选择完成三 seed 共享保守聚合 | 已通过 | seed 42/43/44 使用 GPU 6/6/4；step-core median 30.031241 env-step/s、CV 6.340%，三次共 99 次局部 reset、0 正确性错误；`shared_conservative` PASS，独占基线与卡级资源结论仍未执行 |
| 2026-10-15 | B1 只测完整 Task-v0 / Observation-v0 的 `rollout_with_reset`，不保留不可满足的无终止 step-core | 已修订 | B0 seed 42 真实轨迹在冻结周期动作第 291 step 越过 Task-v0 下边界 `(y=0.392030 m)`，早于 500-step warmup；B1 Gate 与性能分档原本就只使用 rollout，B0 Gate 不变 |
| 2026-10-16 | B1 使用完整 Task-v0 / Observation-v0 完成三 seed 共享保守聚合 | 已通过 | rollout median 41.203994 env-step/s、CV 1.463%、三次最坏 p99 0.039680 s；共 129 次局部 reset、0 正确性错误，判 Yellow；见 `docs/rl/p1_b1_results.md` |
| 2026-10-16 | S0 允许共享 GPU，并以 measured interval 的 PID RSS/显存趋势判稳定性；transition 使用流式写盘 | 已接受 | 1000-step 短程共享预检 exit 0，3 次局部 reset、0 正确性错误、PID 显存增长/斜率为 0；非冻结长度明确标记 `diagnostic_shared`，不进入 Gate，见 `docs/rl/p1_s0_results.md` |
| 2026-10-16 | 正式 S0 在共享 GPU 上完成冻结的 100000-step 稳定性长跑 | 已通过 | 100000/100000、452 次局部 reset、0 正确性错误；RSS 增长 58.945 MiB、斜率 0.687 MiB/1000 step，项目 PID 显存增长/斜率均为 0；`valid_shared_stability` PASS，见 `docs/rl/p1_s0_results.md` |
| 2026-10-16 | B2 直接读取 Husky D455 的 320×240 RGBA/depth，每 6 physics step 渲染一次，相机不进入 Observation-v0 | 已通过 | 冻结 100/1000 共享诊断 1000/1000 帧有效，14.643520 env-step/s；相对同卡共享 B1 为 0.355557×，项目 PID 显存/RSS 峰值近似增加 72.250/151.840 MiB，见 `docs/rl/p1_b2_results.md` |
| 2026-10-16 | B3 每个 policy step 后按 adapter 的允许轮地 link 更新一次 DEM/USD mesh，使用 Isaac `wxyz`，相机关闭、render=false | 已通过 | 冻结 100/1000 共享诊断 1000/1000 形变成功、0 错误，23.515312 env-step/s；相对同卡共享 B1 为 0.570971×，项目 PID 显存/RSS 峰值近似增加 19.875/17.434 MiB；现有路径 `update_collider=False`，见 `docs/rl/p1_b3_results.md` |
| 2026-10-16 | P1 最终 Gate 汇总 | 已通过 / Yellow | 所有硬 Gate 通过，B2/B3 报告项完成；B1 median 41.203994 env-step/s，允许进入 P2，P3 前需优化/并行或评估 off-policy，见 `docs/rl/p1_final_report.md` |
| 待定 | PPO、多进程 PPO 或 off-policy | P3 待决策 | 由 P1 吞吐量与并行能力决定 |

---

## 10. 当前唯一工作包

P0、P1 已完成。P1 的 L0/C0、Task-v0 runtime、正式 T0、独立撞岩、V0、完整 R0、
D0、共享保守 B0/B1、正式 S0 和共享 B2/B3 诊断均已完成，最终判定为
**PASS / Yellow**。独占 B0/B1 与隔离配对 B2/B3 可作为补充，但不阻塞按修订协议进入
**P2：复现 RLRoverLab 的 MDP 基线**。

P1 不应创建 PPO 训练、不接 ROS 2、不启用相机训练、不做复杂月壤随机化。
其首个实现窗口只应建立 native RL 生命周期、robot adapter 和最小 headless
`reset/step`，随后按冻结协议逐层加入观测、碰撞与 benchmark。

建议完成顺序：

1. ~~只读盘点两个 OmniLRS 候选环境和启动入口。~~
2. ~~对候选基线执行最小 headless smoke test。~~
3. ~~选择并记录唯一基线。~~
4. ~~写 Task-v0/Observation-v0/Action-v0/Reward-v0 规格。~~
5. ~~写 benchmark protocol 与 P1 数值 Gate。~~
6. ~~更新本文“状态”和“决策日志”，再进入 P1。~~

P1 建议完成顺序：

1. ~~加入 Gymnasium，并建立不依赖 ROS/Yamcs 的 native RL manager。~~
2. ~~实现 robot adapter、Action-v0 和可清残留的局部 reset。~~
3. ~~完成 L0/C0 headless 动态 smoke，确认直行、转向、急停和 articulation 生命周期。~~
4. ~~实现 Task/Observation/Reward/termination 纯函数与测试。~~
5. ~~Task-v0 runtime、正常轮地接触、正式 T0、独立撞岩、V0、R0、D0、共享保守 B0/B1、正式 S0 和 B2/B3 已完成。~~
6. ~~根据 `p1_benchmark_protocol.md` 作出 Green/Yellow/Red 决策：Yellow。~~

---

## 11. 新窗口启动提示（直接复制）

```text
请继续 OmniLRS-native RL 项目。

首先完整阅读：
/home/spatial/OmniLRS-isaac6/docs/rl/OMNILRS_NATIVE_RL_ROADMAP.md

严格遵守其中的“最终目标”“防偏离规则”和阶段 Gate。先检查工作区现状与决策日志，说明当前 Phase、这次窗口的唯一交付物和验收标准，再开始实施。不要提前跨阶段，不要让 ROS2 进入训练内环，不要把局部导航 RL 当成 SLAM，也不要覆盖已有用户修改。

P0、P1 已通过。当前应从 P2 开始：复用已经冻结并动态验证的 Gymnasium 生命周期、
Task-v0/Observation-v0/Reward-v0/termination 与局部 reset，先建立 golden cases 和
zero/forward/random/goal-directed 非学习基线，再用固定 seed 证明目标指向控制器明显
优于随机策略。不要重复 P1 长跑，也不要提前加入 PPO。
```

如果新窗口只做某个子任务，在上述提示后追加：

```text
本窗口的唯一交付物：<填写一个交付物>
验收方式：<填写命令、测试或文档证据>
```

---

## 12. 每个窗口结束时的交接模板

```text
当前 Phase：
本窗口唯一交付物：
关联需求 ID：

完成内容：
修改文件：
验证命令及结果：
性能/正确性证据：

未解决问题：
新发现风险：
做出的架构决定（已写入决策日志）：

当前 Gate 是否通过：是 / 否
未通过项：
下一窗口唯一建议任务：
```

此模板应保证新窗口从证据继续，而不是重新讨论或凭印象推进。
