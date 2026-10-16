# P1 B3 地形形变成本诊断

> 日期：2026-10-16
> 状态：冻结长度 `valid_shared_diagnostic` PASS；相对成本为共享近似值

## 实现边界

B3 复用完整 B1 Task-v0 / Observation-v0 rollout，并加载
`environment=lunalab_deformable`。形变输入直接来自 Task-v0 已建立的 Isaac 具名刚体
view：按 `RobotAdapterCfg.allowed_ground_contact_links` 读取轮 link 世界位置、Isaac
`wxyz` 姿态和净接触力，不写死四个车轮；当前 Husky 配置解析出四个 link。

每个 policy step 的 6 个 physics step 完成后执行一次形变。Lunalab 现有
`TerrainManager.deformTerrain()` 更新 DEM 与 USD mesh、保持 `update_collider=False`；
更新后的 DEM 同步给 Observation-v0 高度查询。相机关闭，环境 `render=False`，形变不
进入 action、reward 或 termination 接口，也不经过 ROS、Yamcs 或 SDG manager。

这项 B3 是现有视觉网格/高度观测形变路径的成本诊断，不证明动态 collider 或完整轮壤
耦合；后者留给后续阶段。

## 动态预检

5 warmup + 10 measured 预检位于
`data/benchmarks/p1/20261016_b3_preflight_auto_shared_r1/`，exit code 0。10/10 次更新成功、
错误为 0，1538 个 DEM 样本变化，最大绝对变化 0.047555819 m；articulation、
Observation-v0、finite 与 decimation 检查全部通过。

## 冻结长度结果

命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case b3 --gpu auto --allow-shared-gpu \
  --seed 42 --warmup-steps 100 --measured-steps 1000 \
  --output-dir data/benchmarks/p1/20261016_b3_seed42_auto_shared_r1
```

运行自动选择物理 GPU 2（UUID
`GPU-9394be27-72a5-9643-4d7b-72e11e23e999`）。启动前已有外部 PID `301825`，因此
结果标记为 `performance_validity=shared_approximate`，整卡指标不归因给项目。

- exit code 0，1000/1000 measured policy transition；
- 1000/1000 地形形变更新成功，异常为 0；
- measured interval 内 61168 个 DEM 样本变化，最大绝对变化 0.084211856 m，RMS
  变化 0.003125405 m；
- 最大轮 link 法向接触力 179.381760 N；
- 3 次局部 reset：collision 1、time-limit 2；
- articulation、decimation、Observation-v0、action/observation/reward finite 和完整
  `World.reset()` 错误均为 0；
- 吞吐 23.515312 env-step/s，RTF 4.703062，step p99 0.055211 s；
- 项目 PID 显存峰值 3410874368 bytes，RSS 峰值 8336261120 bytes；
- measured interval 项目 PID 显存增长/斜率为 0；RSS 首尾增长 77824 bytes，斜率
  93648.349 bytes/1000 step。

## 相对 B1 成本

比较对象为同一物理 GPU 2、seed 42、共享轨的
`data/benchmarks/p1/20261016_b1_seed42_auto_shared_r1/`。B1 是 500/10000，B3 是协议
冻结的 100/1000，且外部负载快照不同，因此倍率只作近似，不能替代隔离配对实验。

- B1：41.184755 env-step/s；
- B3：23.515312 env-step/s；
- B3/B1 吞吐倍率 0.570971，即下降 42.903%；等价 slowdown 1.751402×；
- 项目 PID 显存峰值增加 20840448 bytes（19.875 MiB）；
- RSS 峰值增加 18280448 bytes（17.434 MiB）。

B3 没有绝对吞吐硬 Gate。冻结诊断已完整执行并 PASS；共享相对倍率保持
`shared_approximate` 语义。产物目录为
`data/benchmarks/p1/20261016_b3_seed42_auto_shared_r1/`，Kit 日志为
`.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_013751.log`。
