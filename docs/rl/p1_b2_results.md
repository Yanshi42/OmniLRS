# P1 B2 D455 RGB/depth 成本诊断

> 日期：2026-10-16
> 状态：冻结长度 `valid_shared_diagnostic` PASS；相对成本为共享近似值

## 实现边界

B2 复用完整 B1 Task-v0 / Observation-v0 rollout，并直接绑定 Husky USD 中已有的
`/Robots/husky/d455/RSD455/Camera_OmniVision_OV9782_Color`。Isaac Camera 输出
320×240 RGBA 和 `distance_to_image_plane` depth；环境按全局 physics step 每 6 step
执行一次 render，即 5 Hz。每个 policy step 读回一对 RGB/depth，但相机数据不进入
Observation-v0，也不经过 ROS、Yamcs 或 SDG manager。

首次 10 warmup + 50 measured 预检完成相机初始化和 rollout 后，在保存 startup 元数据时
因 B2 分支未赋值 `first_observation` 而 exit 1。失败产物
`data/benchmarks/p1/20261016_b2_preflight_auto_shared_r1/` 不进入诊断结论。修复后的 r2
exit 0，50/50 帧有效，短测吞吐 16.703184 env-step/s。

## 冻结长度结果

命令：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python \
  scripts/rl/run_p1_benchmark.py --case b2 --gpu auto --allow-shared-gpu \
  --seed 42 --warmup-steps 100 --measured-steps 1000 \
  --output-dir data/benchmarks/p1/20261016_b2_seed42_auto_shared_r1
```

运行自动选择物理 GPU 2（UUID
`GPU-9394be27-72a5-9643-4d7b-72e11e23e999`），该卡已有外部 PID `301825`，因此结果
标记为 `performance_validity=shared_approximate`，整卡指标不归因给项目。

- exit code 0，1000/1000 measured policy transition；
- 1000/1000 RGB/depth capture 有效，异常、空帧、shape、finite 错误均为 0；
- RGBA shape `(240,320,4)`，depth shape `(240,320)`；
- 最低 RGB 非零比例 0.726519，最低 depth finite 比例 0.727344；
- 3 次局部 reset：collision 1、time-limit 2；
- articulation、decimation、Observation-v0、action/observation/reward finite 和完整
  `World.reset()`错误均为 0；
- 吞吐 14.643520 env-step/s，RTF 2.928704，step p99 0.088511 s；
- 项目 PID 显存峰值 3465793536 bytes，RSS 峰值 8477196288 bytes。

## 相对 B1 成本

比较对象为同一物理 GPU 2、seed 42、共享轨的
`data/benchmarks/p1/20261016_b1_seed42_auto_shared_r1/`。B1 是 500/10000，B2 是协议冻结的
100/1000，且运行时外部负载快照不同，因此倍率只作近似，不能替代隔离配对实验。

- B1：41.184755 env-step/s；
- B2：14.643520 env-step/s；
- B2/B1 吞吐倍率 0.355557，即下降 64.444%；等价 slowdown 2.812490×；
- 项目 PID 显存峰值增加 75759616 bytes（72.250 MiB）；
- RSS 峰值增加 159215616 bytes（151.840 MiB）。

B2 没有绝对吞吐硬 Gate。冻结诊断已完整执行并 PASS；共享相对倍率保持
`shared_approximate` 语义。产物目录为
`data/benchmarks/p1/20261016_b2_seed42_auto_shared_r1/`，Kit 日志为
`.cache/isaacsim/logs/Kit/Isaac-Sim Python/6.1/kit_20261016_013001.log`。
