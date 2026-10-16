# OmniLRS-native RL Task-v0 接口规格

> 状态：Frozen for P1 implementation
> 规格日期：2026-10-14
> 适用环境：Isaac Sim 6.1 / Lunalab / `nograph_husky.usd`
> 接口版本：`Task-v0`、`Observation-v0`、`Action-v0`、`Reward-v0`

本文把 P0 的机器人接口审计转成可直接实现和测试的契约。P1 可以标定底盘
驱动与接触阈值，但不得静默改变张量顺序、单位、坐标、终止语义或奖励公式；
如证据要求修改，必须更新本文和决策日志。

## 1. 范围与固定运行条件

Task-v0 是单车、局部目标位姿导航任务。训练内环直接调用 OmniLRS/Isaac
Python API，不经过 ROS 2 或 Yamcs。P1 不启用相机、地形变形或训练算法。

| 项目 | Task-v0 固定值 |
|---|---|
| 场景 | Lunalab |
| 机器人 | `robot=husky`，`assets/USD_Assets/robots/nograph_husky.usd` |
| 环境数 | 1 |
| physics dt | 精确 `1 / 30 s`，不得使用截断值 `0.0333` |
| action decimation | 6 个 physics step |
| policy period | `0.2 s`，5 Hz |
| episode time limit | `60.0 s`，即 300 个 policy step |
| rendering | 训练/benchmark 默认关闭；可视化 smoke 可单独开启 |
| terrain deformation | 关闭 |
| MDP 时间源 | 只使用仿真 step/time，不使用 wall clock |

环境必须实现 Gymnasium 五元组语义。`terminated` 和 `truncated` 不得同时为
真；终止 transition 的 observation 是终止状态，不得自动替换为下一 episode
的初始 observation。

## 2. 坐标、单位与四元数

### 2.1 世界系与车体系

- 世界系 `W`：右手、Z-up，单位米；Lunalab 的 `+X/+Y` 沿地形网格递增。
- 车体系 `B`：`+X` 前、`+Y` 左、`+Z` 上；正 yaw 绕 `+Z` 逆时针，表示左转。
- 线速度使用 `m/s`，角速度使用 `rad/s`，力使用 N，时间使用仿真秒。
- 内部角度统一包裹到 `[-pi, pi]`。

静态 USD 检查得到四轮中心前后位置约 `x=±0.256 m`、左右位置约
`y=±0.2854 m`。车轮包围盒直径约 `0.3555 m`，因此 Action-v0 使用：

```text
wheel_radius r = 0.17775 m
wheel_track  b = 0.57080 m
```

这两个值是动作适配器的初值；P1 必须用直行和原地转向 smoke 验证符号与
实际位移，验证失败时先修正配置并记录证据，不允许在策略层补偿反号。

### 2.2 四元数边界

| 边界 | 顺序 |
|---|---|
| YAML pose、USD `Gf.Quatd`、Isaac `get_world_pose()` | `(w, x, y, z)` |
| Gym/Task-v0 对外状态 | `(x, y, z, w)` |
| `Robot.teleport()` 参数 | `(x, y, z, w)` |

`RobotRigidGroup.get_pose_of_base_link()` 当前 docstring 写成 xyzw，但底层
`SingleXFormPrim.get_world_pose()` 实际返回 wxyz；P1 不得依赖该错误注释，
应由单一 adapter 完成显式转换并写 identity/yaw-90°测试。

### 2.3 相机、IMU 与 ROS 约定

Task-v0 不使用相机或 IMU 观测，但 P0 静态审计结果作为后续接口边界保留：

- Husky 的 USD camera 以局部 `-Z` 向前、`+Y` 向上；换成常见 optical frame
  后，`z_optical=+X_B`、`x_optical=-Y_B`、`y_optical=-Z_B`。
- USD 内 IMU 相对车体存在 180° yaw：传感器 `+X/-Y` 分别对应
  `-X_B/-Y_B`，`+Z` 对应 `+Z_B`。现有 getter 还对 roll/pitch 取反，必须在
  Observation-v1 前用已知姿态校验，不能直接视为车体系真值。
- ROS 部署遵守 REP-103 的 `base_link`/optical frame；转换只在部署 adapter
  发生，不改变本规格张量。

## 3. Task-v0

### 3.1 地图边界与采样

当前 Lunalab 网格有 650×1000 个点、分辨率 0.01 m，因此实际顶点范围是：

```text
x in [0.00, 6.49] m
y in [0.00, 9.99] m
```

采样使用环境自己的 seeded `numpy.random.Generator`，禁止全局随机状态。

| 参数 | 值 |
|---|---|
| start/goal XY 采样区 | `x∈[0.75,5.74]`、`y∈[0.75,9.24]` |
| start-goal 平面距离 | `[2.0, 8.0] m` |
| start yaw / goal yaw | 各自独立均匀采样 `[-pi, pi]` |
| 初始 root z | 起点地形高度 `+0.50 m`，随后零动作 settle |
| 最大拒绝采样次数 | 1000；超过后抛出错误，不返回违规样本 |

起点和终点必须落在有效 DEM 上，且 footprint 内不得与已知静态障碍重叠。
P1 的固定平地用例可以通过 `reset(options={"start_pose": ..., "goal_pose": ...})`
绕过随机采样，但仍必须执行合法性检查。

### 3.2 成功、终止与截断

每个 policy step 完成 6 次 physics step 后，以 post-step 状态判定：

| reason | 精确定义 | Gymnasium 结果 |
|---|---|---|
| `collision` | 除四个轮 link 外的 Husky 刚体与地形/岩石/障碍接触，合力范数 `>20 N`，连续 2 个 physics step | `terminated=True` |
| `rollover` | `abs(roll)>60°` 或 `abs(pitch)>60°`，连续 3 个 physics step | `terminated=True` |
| `out_of_bounds` | root XY 离开 `x∈[0.45,6.04]` 或 `y∈[0.45,9.54]` | `terminated=True` |
| `success` | 平面目标距离 `<=0.50 m` 且 `abs(goal_yaw-yaw)<=20°` | `terminated=True` |
| `time_limit` | 已执行 300 个 policy step，且本步没有上述 termination | `truncated=True` |

同一步满足多个条件时只报告一个原因，优先级固定为：

```text
collision > rollover > out_of_bounds > success > time_limit
```

碰撞检测必须基于可识别的刚体/碰撞对或专用 base/chassis contact view；不能用
`ArticulationTelemetry` 中匿名的 `link_0`、`link_1` wrench 代替。20 N 和去抖
长度是冻结的 Task-v0 初值，P1 必须分别用静止、正常轮地接触、底盘擦地和撞岩
脚本给出混淆矩阵；若正常状态误触发，则在进入 P2 前升版或记录规格修订。

## 4. Observation-v0

对外 observation 是 C-contiguous `np.float32`，shape 为 `(10206,)`，顺序固定：

| slice | shape | 语义 | 归一化/范围 |
|---|---:|---|---|
| `[0:2]` | 2 | 上一步实际施加的 `[v, omega]` 对应归一化动作 | `[-1,1]`；reset 后为 0 |
| `[2]` | 1 | 当前 root 到 goal 的 XY 距离 | `clip(d/8, 0, 1)` |
| `[3]` | 1 | 目标位置在 B 系的 bearing | `wrap(atan2(y_B,x_B))/pi` |
| `[4]` | 1 | 目标 yaw 与当前 yaw 的误差 | `wrap(goal_yaw-yaw)/pi` |
| `[5:10206]` | 10201 | `101×101` 理想几何高度图，C-order flatten | `[-1,1]` |

### 4.1 理想高度图

- 分辨率 `0.05 m`，覆盖车体中心周围 `5.0×5.0 m`，含两端点。
- `H[50,50]` 对应车体 XY 原点。
- 图像顶部是前方、左侧是车体左方：

```text
x_B(i,j) = (50 - i) * 0.05
y_B(i,j) = (50 - j) * 0.05
```

- 值为相对中心地形高度：

```text
H = clip((z_terrain(x_B,y_B) - z_terrain(0,0)) / 1.0 m, -1, 1)
```

- 超出 DEM 或无效查询的 cell 固定填 `-1.0`。Observation-v0 不另带 mask；
  Observation-v2 必须加入有效性/不确定度。
- 该高度图是 P2 的 privileged geometric observation，不能宣称为最终可部署
  actor 输入。

所有 observation 元素必须 finite。测试至少覆盖中心点、前/左方向、yaw 旋转、
边界填充值、flatten 顺序、dtype、shape 和范围。

## 5. Action-v0

输入为 `np.float32` shape `(2,)`、范围 `[-1,1]`：

```text
action[0] = normalized linear command
action[1] = normalized yaw-rate command
v_des     = 0.80 * action[0]       m/s
omega_des = 1.20 * action[1]       rad/s
```

环境应拒绝错误 shape 或 NaN/Inf；有限的越界值可以 clip，同时在 `info` 记录
`action_clipped=True`。命令使用以下仿真时间 slew limit：

```text
abs(v_k - v_{k-1})         <= 1.0 m/s² * 0.2 s = 0.20 m/s
abs(omega_k - omega_{k-1}) <= 2.0 rad/s² * 0.2 s = 0.40 rad/s
```

实际施加的车辆级命令转换为四轮目标角速度：

```text
left_rad_s  = (v - omega * b/2) / r
right_rad_s = (v + omega * b/2) / r
r = 0.17775 m, b = 0.57080 m
```

左右前后轮同侧同值，轮速最终限幅为 `[-7,7] rad/s`，命令保持 6 个 physics
step。Task-v0 没有人为 command delay。reset、异常和 terminal 后的急停绕过 slew
limit，立即向全部四轮写 0。

现有 `Robot.drive_straight(linear_velocity)` 实际把参数直接作为轮关节 `rad/s`
发送，并不是 `m/s`。P1 adapter 必须调用命名 joint velocity 接口并实现上述转换，
不得把该 legacy 方法当作 Action-v0 映射。

## 6. Reward-v0

奖励在每个 policy step 计算一次，所有分量写入 `info["reward_components"]`。
公式使用 post-step 状态和上一个 policy step 的缓存，不再乘 `dt`：

```text
r_progress = 5.0 * clip(d_prev - d_now, -0.25, 0.25)
r_heading  = 0.5 * (abs(e_prev) - abs(e_now)) / pi
             only if d_prev < 1.0 and d_now < 1.0, else 0
r_action   = -0.02 * sum((a_applied_now - a_applied_prev)^2)
r_tilt     = -0.05 * clip((max(abs(roll),abs(pitch))/radians(45))^2, 0, 1)
r_time     = -0.01
r_terminal = +10 for success; -10 for collision/rollover/out_of_bounds; 0 otherwise
reward     = r_progress + r_heading + r_action + r_tilt + r_time + r_terminal
```

这里的 `e` 是未归一化 goal heading error（rad），`a_applied` 是经过 slew limit
后再按最大 `v/omega` 归一化的动作。timeout 没有额外 terminal penalty。距离进度
clip 防止 teleport/reset 或数值跳变产生巨额奖励。

单元测试必须包含：靠近/远离目标、近目标朝向改善/恶化、动作突变、倾斜、成功、
三类失败和 timeout；除规格所列分量外不得存在隐藏 reward。

## 7. reset 契约

每次 `reset()` 必须按以下顺序执行，且 settle step 不计入 episode：

1. 立即把策略动作、车辆命令和四轮 drive target 全部置零。
2. 显式把所有 wheel joint velocity 置零；连续轮角可以保留，但缓存必须清空。
3. teleport root 到采样起点，并把 root linear/angular velocity 全部置零。
4. 清空 contact 去抖、telemetry/history、previous action/observation、距离、奖励、
   path length、计数器和 termination 状态。
5. 在零轮速下最少 settle 15 个 physics step；若 root 竖直速度仍
   `>=0.05 m/s`，最多延长到 60 步，超限则 reset 失败并报告。
6. settle 后建立 `d_prev/e_prev`，episode step 归零，再返回初始 observation。

局部 episode reset 不得调用完整 `World.reset()`。只有场景构建、物理 view 失效或
不可恢复错误才允许完整 reset，并必须重建 articulation/contact views。

现有 `Robot.teleport()` 已清 root 线/角速度，但没有清 joint velocity、drive
target、接触历史或 RL 缓存，所以单独调用它不满足本契约。

## 8. `info` 与 episode 指标

每步至少返回：

```text
episode_id, seed, terrain_id, success, termination_reason,
distance_to_goal, heading_error, path_length, elapsed_sim_time,
collision_count, max_roll, max_pitch, mean_slip,
reward_components, observation_version, action_version,
task_version, action_clipped
```

- 活跃 episode 的 `termination_reason` 为 `None`。
- `mean_slip` 在 v0 尚未实现时为 `float("nan")`，不得伪造为 0。
- version 字符串精确为 `Observation-v0`、`Action-v0`、`Task-v0`。
- path length 累加相邻 policy post-step root XY 距离，不含 reset/settle。
- elapsed time 等于 `episode_steps * 0.2`，不能取 wall-clock。

## 9. 现有实现审计与 P1 缺口

| 路径 | 已有能力 | 不能直接满足 Task-v0 的部分 | P1 动作 |
|---|---|---|---|
| `Robot` / `ArticulationControl` | 命名关节速度/位置/力命令；root pose；关节 telemetry | legacy 直行参数单位误导；无车辆级 v/omega 限幅器 | 新建 RL robot adapter，保留现有底层 API |
| `ArticulationTelemetry` | 关节位置、速度、effort、measured wrench | contact 名称只是 `link_i`；timestamp/rate limit 是 wall-clock | 不用于任务碰撞；step 内禁用 wall-clock gate |
| `RobotRigidGroup` | 四轮 pose、速度、net contact force | 只配置四轮，主要服务地形变形；base 只有 Xform pose | 新建 chassis contact view；轮接触不触发 collision |
| camera / IMU | RGB、depth 和 IMU getter 已存在 | frame/sign 未动态验证；相机 getter会初始化渲染产品 | Task-v0 关闭；留到 Observation-v1/v2 |
| `Robot.teleport/reset` | root teleport 与 root velocity 清零 | 不清 drive target、joint velocity、contact/RL history | 按第 7 节组合局部 reset |
| ROS2/Yamcs managers | 会加载 robot 并在主循环更新 articulation | 生命周期绑定消息/实时循环；每次 reset 路径含 `World.reset()` | P1 新建非 ROS 的 native RL manager |
| SDG manager | 已用于 Isaac 6.1 headless 场景 smoke | 不加载/控制机器人 | 不作为 RL runtime |

现有 `test_ros2_sanity.py` 只验证启动 marker 和 ROS topic 存在，不能证明轮速符号、
位移、碰撞分类或 reset 无残留。

## 10. P1 实现验收矩阵

P1 在实现 Gym 环境后至少提供以下自动化/脚本测试：

| 测试 | 通过条件 |
|---|---|
| contract | action/observation shape、dtype、range、flatten 顺序全符合本文 |
| frame | identity 与 yaw±90°下 goal bearing、heading error、heightmap 方向正确 |
| straight | 正 action 使 `+X_B` 位移；左右轮符号相同；横移/yaw 漂移有记录 |
| turn | 正 omega 左转；左右轮符号符合公式；原地位移漂移有记录 |
| stop | 急停后 drive targets 为 0，速度在给定 settle 窗内衰减 |
| collision | 正常轮地接触不终止；底盘/岩石接触按 20 N×2 step 终止 |
| termination | success/collision/rollover/out-of-bounds/time-limit 可分别脚本触发且 reason 正确 |
| reset | 至少 500 次局部 reset 后无 command、速度、接触、history 或 episode 计数残留 |
| deterministic | 同 seed 的采样、初始 observation 与脚本动作轨迹在声明容差内一致 |
| timing | 每个 action 精确保持 6 physics step；300 步仅触发 timeout truncation |

这份矩阵定义“接口正确”，不替代下一份 benchmark protocol 的吞吐量、显存和长跑
门槛。
