# P1 R0 局部 reset 压力测试结果

> 日期：2026-10-15
> 状态：R0 Gate PASS（500/500，pose/速度/控制/历史残留全部达标）
> 仿真：Isaac Sim 6.1.0.0，Pixi `default`，GPU 4 共享诊断，PID 显存归因

## 冻结 workload

- seed 42；
- 单个 Lunalab `Terrain_lidar01` 与四轮 Husky；
- 每次 reset 前执行 20 个 seeded uniform random Action-v0；
- 500 次局部 reset，总计 10000 policy transition；
- 不调用完整 `World.reset()`；
- 每次检查 action、drive target、episode/history、root/轮速度、目标 pose、settle step、
  articulation 和 observation finite。

通过产物：`data/benchmarks/p1/20261015_r0_500_gpu4_shared_r3/`

## 结果

500 次均执行完成，500 行逐次记录全部通过；没有 crash、OOM、Xid、invalid
articulation、decimation error 或非 finite，完整 `World.reset()` 次数为 0。reset
latency 为：

| 指标 | 数值 |
|---|---:|
| p50 | 0.110333 s |
| p95 | 0.159982 s |
| p99 | 0.191708 s |
| max | 0.209391 s |

动作、所有 drive target、episode step/time 和历史缓冲在 500 次 reset 后均为零；
root linear/angular speed 和所有轮速最大值也均为零。目标 pose 的逐次上限没有放宽：
最大 XY 误差 `1.192093e-7 m`，最大 yaw 误差 `2.556042e-10 rad`，均远小于冻结的
`0.02 m/rad`。settle 为 19--37 physics step，未超过 60。

修复是在速度 settle 后保留地形接触形成的 Z/roll/pitch，只把 XY/yaw 回投到请求的
episode pose，再立即清零 drive target、关节速度和 root 速度；回投后不额外推进物理，
并再次清除控制、接触与 episode 历史。50-reset 短探针先得到 50/50、0 residual
failure，随后才执行本次完整 R0。

资源方面，项目 PID 显存峰值 3232.988 MiB、进程 RSS 峰值约 7.349 GiB；卡上其他
进程显存单独记录。共享状态不影响 PID 显存和 reset 正确性证据，但本页不使用卡级
吞吐或利用率作正式性能结论。

## 无效首轮

`20261015_r0_500_gpu4_shared_r1` 曾把 settle 后正常重新采样的 `joints_state` 和
`contacts_state` 当成历史残留，发现后立即中止。修正版只检查
`joints_history/contacts_history`，并由单元测试覆盖；r1 不计入结果。

## 修复前对照

`20261015_r0_500_gpu4_shared_r2` 完成 500/500，但 settle 后有 15 次 pose 超过冻结
上限：yaw 14 次、XY 2 次、重叠 1 次，最大分别为 `0.045250 rad` 和
`0.020692 m`。当时按逐次 Gate 正确判为 FAIL，没有用 p95 或 485/500 放宽判定。

## 结论与下一步

R0 Gate 为 PASS。该结论只覆盖局部 reset 正确性和延迟，不替代尚未执行的正式独占
B0/B1 吞吐、D0 确定性与 S0 100000-step 长跑。
