# P1 Task-v0 纯函数契约验证

> 日期：2026-10-14
> 状态：纯函数与单元测试通过；尚未接入完整 episode runtime
> 适用规格：[`task_v0_spec.md`](task_v0_spec.md)

## 本窗口范围

在已通过的 L0/C0 生命周期上增加 Task-v0 纯函数层，不修改 Action-v0、Husky
adapter 或 Isaac smoke 路径，也不引入 ROS 2、Yamcs、SDG runtime、PPO/skrl。

实现内容：

- seeded start/goal 采样、距离/边界/有效地形/footprint callback 约束；
- 车体系目标距离、bearing 和 wrapped heading error；
- `101×101`、0.05 m 分辨率、车体朝向对齐的 Observation-v0 高度图与固定
  `(10206,)` C-contiguous `float32` 张量；
- Reward-v0 六个独立分量及总和；
- 碰撞、倾覆、越界、成功、超时的冻结优先级；
- 20 N×2 physics step 碰撞去抖、60°×3 physics step 倾覆去抖，并把已触发状态
  锁存到 episode reset；
- 允许轮地接触 link 由 adapter 配置传入，终止逻辑不写死四轮。

## 验证

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default \
  python -m pytest -q test/rl
```

结果：`32 passed`。其中原 L0/C0 adapter/reset 测试 9 项继续通过，新增 Task、
Observation、Reward 和 termination 契约测试 23 项通过。`compileall` 同时通过。

覆盖的关键 golden cases：

- identity/yaw 90° 下目标 bearing 与高度图前/左方向；
- observation shape、dtype、flatten 顺序、无效高度填充值和 finite/range；
- Reward-v0 进度 clip、近目标 heading、动作变化、倾斜、成功、三类失败和 timeout；
- 正常轮地接触不碰撞、chassis 严格大于 20 N 后触发、接触消失后 latch 不丢失；
- collision > rollover > out-of-bounds > success > time-limit；
- `terminated`/`truncated` 与 success flag 的冻结语义。

## 结论边界

本节记录 2026-10-14 当时的边界：证据只说明数学契约与状态转换符合 Task-v0，
尚未证明动态 runtime。2026-10-15 已完成可选 Task-v0 runtime、真实 DEM 和具名
contact view 接线；后续证据见 [`p1_task_runtime_results.md`](p1_task_runtime_results.md)。
完整 T0 动态终止覆盖、R0、D0、B0/B1 和 S0 仍未执行，因此完整 P1 Gate 状态不变。
