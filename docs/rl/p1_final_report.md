# P1 单车 Gym 环境与可行性基准最终报告

> 日期：2026-10-16
> 结论：P1 功能与稳定性 Gate PASS；吞吐分档 Yellow

## 结论

OmniLRS-native 单车路线已具备可运行的 Gymnasium 生命周期、可替换车辆 adapter、
Task-v0/Observation-v0/Reward-v0/termination、局部 reset、动态正确性、可视化、确定性、
长跑稳定性和分层成本数据。训练内环直接使用 Python/Isaac Sim API，没有接入 ROS 2、
Yamcs、SDG manager runtime 或 PPO/skrl。

B1 三进程共享保守 median 为 41.203994 env-step/s、RTF 8.240799，达到冻结的
`>=5 env-step/s` Gate，但低于 Green 的 50 env-step/s，因此 P1 判 **Yellow**：允许进入
P2；进入 P3 训练前必须优化、多环境/多进程并行，或评估 off-policy。

## Gate 状态

| 项目 | 状态 | 核心证据 |
|---|---|---|
| L0/C0 生命周期与运动方向 | PASS | exit 0；articulation 失效 0；直行 +3.651876 m；左转 +135.178836°；急停轮速 0 |
| Action-v0 / 6-step decimation | PASS | 动态 case 的 decimation error 均为 0 |
| Task/Observation/Reward/termination | PASS | Observation-v0 10206 维；运行时 contract/non-finite error 为 0 |
| T0 终止覆盖 | PASS | success/collision/rollover/out-of-bounds/time-limit 各 10/10 |
| V0 人工评估产物 | PASS | 场景、四轨迹、控制/奖励曲线齐全 |
| R0 局部 reset | PASS | 500/500；10000 transition；0 次完整 World reset；残留错误 0 |
| D0 确定性 | PASS | 三进程边界/reason 一致；position/yaw/reward 最大误差均为 0 |
| B0 核心吞吐 | PASS（共享保守） | step-core median 30.031241 env-step/s，阈值 20 |
| B1 主吞吐 | PASS（共享保守） | median 41.203994 env-step/s；p99 最坏 0.039680 s；CV 1.463% |
| S0 稳定性 | PASS | 100000/100000；452 reset；RSS/GPU 增长与斜率全部达标 |
| B2 相机成本 | 已执行，报告项 | 14.643520 env-step/s；B2/B1 0.355557×；1000/1000 帧有效 |
| B3 地形形变成本 | 已执行，报告项 | 23.515312 env-step/s；B3/B1 0.570971×；1000/1000 更新成功 |

证据文档：[`p1_l0_c0_results.md`](p1_l0_c0_results.md)、
[`p1_t0_results.md`](p1_t0_results.md)、[`p1_v0_results.md`](p1_v0_results.md)、
[`p1_r0_results.md`](p1_r0_results.md)、[`p1_d0_results.md`](p1_d0_results.md)、
[`p1_b0_harness_results.md`](p1_b0_harness_results.md)、
[`p1_b1_results.md`](p1_b1_results.md)、[`p1_s0_results.md`](p1_s0_results.md)、
[`p1_b2_results.md`](p1_b2_results.md)、[`p1_b3_results.md`](p1_b3_results.md)。

## 性能与资源

以 B1 median 估算，1M policy step 为 6.741526 小时，100M 为 28.089690 天。单进程
虽满足 P1 可行性门槛，但不足以作为高效 on-policy 训练配置。

正式 S0 的项目 PID 显存峰值为 2836385792 bytes，measured interval 增长和斜率均为
0；RSS 峰值 8334913536 bytes，首尾增长 58.945 MiB、斜率 0.687 MiB/1000 step。
这些值满足冻结稳定性 Gate。

B2/B3 相对 B1 的 wall-time 倍率是共享近似值。项目显存均按 PID 归因；共享卡的整卡
utilization、功耗、温度和总显存没有归因给本项目。

收尾验证：Pixi default 中 Gymnasium 1.4.0 导入成功，RL 单元测试 68/68 通过，
`git diff --check` 以 exit code 0 结束。

## 仍未执行但不阻塞 P1

- B0/B1 的独占 GPU 补充基线与隔离配对的 B2/B3 相对成本；当前协议允许共享保守
  Gate 和共享近似诊断，这些不是 P1 PASS 的前置条件。
- PPO/skrl、100M 训练、并行环境和算法选择；属于 P3，不在 P1 范围。
- B3 的动态 collider 更新与完整轮壤耦合；当前只验证现有 DEM/USD mesh 形变路径，
  `update_collider=False`。
- B1/B2/B3 的其他 seed 成本诊断；冻结协议只要求 B2/B3 seed 42 单进程。

## P1 决策

P1 结束状态为 **PASS / Yellow**。下一工作阶段是 P2：复现 RLRoverLab 的 MDP 基线；
P3 前保留明确的性能工作项：优先评估向量化/多进程吞吐，再据实选择 PPO 或
off-policy，不在本阶段提前安装或实现训练器。
