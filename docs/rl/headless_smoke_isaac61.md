# Isaac Sim 6.1 最小 headless smoke 记录

> 状态：仓库内 Pixi 可重建 smoke 通过（PASS WITH WARNINGS）
> 最新执行日期：2026-10-14
> worktree：`/home/spatial/OmniLRS-isaac6`
> commit：`eb1d3362e6c5b9bae5bedfa957bf920ad6021455`

## 1. 范围与判定

本次只验证 Isaac Sim 6.1 下的以下最小链路：

```text
pixi.lock -> Python 启动 -> headless Kit -> Lunalab 加载 -> World 步进
-> 1280 x 720 RGB/语义帧写出 -> 正常关闭
```

使用现有 SDG/Replicator 路径，不启动 ROS 2。它证明当前 Pixi 环境可以重建、
加载、渲染、步进和退出，但**不证明**机器人控制、局部 reset、碰撞检测或 RL
吞吐量已经可用。

通过条件：

- `pixi lock --check` 通过；
- Kit 日志出现 `app ready`；
- Lunalab 日志出现 `After lab is loaded`；
- 生成非空 RGB、语义图和相机内参；
- 进程退出码为 0；
- 无 Python traceback 或 `Error executing job`；
- 无外部安装目录的缓存锁满或写权限错误。

上述条件全部满足，因此最新功能 smoke 判定为 **PASS WITH WARNINGS**。

## 2. 冻结命令

首次安装：

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi install -e default
env PIXI_HOME=/home/spatial/.local/share/pixi pixi lock --check
```

从仓库根目录执行 smoke：

```bash
set -o pipefail
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python run.py mode=SDG environment=lunalab4SDG rendering=ray_tracing rendering.renderer.headless=True rendering.renderer.active_gpu=null 'rendering.renderer.active_cuda_gpus=[4]' rendering.renderer.physics_gpu=4 mode.generation_settings.num_images=1 mode.generation_settings.data_dir=data/smoke/isaac61_pixi_20261014 'mode.generation_settings.camera_resolutions=[[1280,720]]' 'mode.generation_settings.annotators_list=[[rgb,semantic_segmentation]]' 2>&1 | tee data/smoke/isaac61_pixi_20261014/run_gpu4.log
```

不要同时给出非空 `active_gpu` 和 `active_cuda_gpus`。本次命令让渲染器和
PhysX 都直接选择物理 GPU 4，日志确认：

```text
--portable-root /home/spatial/OmniLRS-isaac6/.cache/isaacsim
--/renderer/multiGpu/activeCudaGpus=4,
--/physics/cudaDevice=4
--no-window
```

## 3. 最新结果

| 项目 | Pixi warm run |
|---|---|
| CPython | 3.12.15 |
| Isaac Sim package | 6.1.0.0 |
| Kit experience | `isaacsim.exp.base.python-6.1.0` |
| 渲染/物理 GPU | physical GPU 4 |
| headless | 是，`--no-window` |
| Kit portable root | `.cache/isaacsim` |
| `app ready` | 93.093 s |
| 首次地形 mesh 更新 | 803.27 ms |
| 生成帧数 | 1 RGB + 1 semantic segmentation |
| 图像规格 | 1280 × 720, RGBA PNG |
| 开始 shutdown | 132.504 s |
| 退出码 | 0 |
| 日志 | 475 行，36727 bytes |

运行时日志显示 Kit 初始化 Warp 1.16.0、CUDA Toolkit 12.9、driver
compatibility 13.2。Python 分发包元数据是 `warp-lang 1.18.0`；后续
benchmark 仍应同时记录 Kit 扩展版本和 Python 分发包版本。

### 3.1 重建时发现并修复的依赖问题

第一次 Pixi smoke 在 Kit 已经 `app ready` 后失败于
`numpy-quaternion 2023.0.3` 的 NumPy 1.x ABI。升级该包后，又暴露了
Numba 与 Coverage 的 API 组合问题。最终固定为已在 6.1 环境验证的组合：

```text
numpy                 2.3.1
numpy-quaternion      2024.0.13
numba                 0.63.1
llvmlite              0.46.0
coverage              7.16.2
```

失败证据保存在
`data/smoke/isaac61_pixi_20261014/run_gpu4_attempt1_numpy_abi.log`。

### 3.2 与旧外部环境运行的对照

2026-10-13 使用 `.venv-isaac61`（底层指向外部
`/mnt/netdisk2/.../isaacsim61`）也完成过相同 smoke：warm run 的
`app ready` 为 67.080 s、151.227 s 开始 shutdown。该记录证明功能可用，
但不再作为唯一复现入口。新基线只使用仓库 `pixi.toml`、`pixi.lock` 和
`pixi run -e default`。

## 4. 可视化产物

- RGB：[preview_rgb.png](/home/spatial/OmniLRS-isaac6/data/smoke/isaac61_pixi_20261014/preview_rgb.png)
- 语义分割：[preview_semantic.png](/home/spatial/OmniLRS-isaac6/data/smoke/isaac61_pixi_20261014/preview_semantic.png)
- 标签映射：[semantic_labels.json](/home/spatial/OmniLRS-isaac6/data/smoke/isaac61_pixi_20261014/semantic_labels.json)
- 相机内参：[camera_intrinsics.csv](/home/spatial/OmniLRS-isaac6/data/smoke/isaac61_pixi_20261014/camera_intrinsics.csv)
- 完整标准输出：[run_gpu4.log](/home/spatial/OmniLRS-isaac6/data/smoke/isaac61_pixi_20261014/run_gpu4.log)

RGB 人工检查结果：

- 能看到 Lunalab 的墙体、帘幕、照明、起伏地形和岩石；
- 几何轮廓、材质和阴影连续，没有全黑、花屏或明显的相机姿态错误；
- 相机高度较低，前景主要是地形，符合 SDG 相机当前离地 0.2 m 的实现；
- 室内后部偏暗，但仍可辨认，不属于渲染失败。

语义图人工和像素检查结果：

- 包含 `BACKGROUND`、`UNLABELLED`、`rock` 和 `ground` 四类；
- 背景/未标注实验室结构、地面和岩石的边界与 RGB 图一致；
- 四类像素数分别为 43320、452630、11432、414218，总计 921600；
- RGB 与语义图均为 `(720, 1280, 4)`、`uint8`。

相机矩阵为：

```text
[[686.222229,   0.0, 640.0],
 [  0.0,      686.222229, 360.0],
 [  0.0,        0.0,       1.0]]
```

主点 `(640, 360)` 与本次 1280 × 720 输出一致。

记录版校验和：

```text
RGB        07fbf2d5543f85cc2ef727457f266e308eabc4ea9159243a5a46441d67bb9e9b
semantic   8c7f7bf1f38bbe98f8ec6ba3dd88716fa455250f964caafb9d96e87b21a12d51
intrinsics f68eb1e9ee55c602fad2efdcc8e5689e971304abb0d1af7a1a7c868fdb61afa2
labels     a5a7287eb2c757c58743892ede56d843bb50ee50a4b1e1dd99eafa3631ac7fb7
log        f6c7274b10d92be031b13838690c26df666d352e83bfef109a095dbedd1ddb9b
```

### 4.1 仍存在的内参实现问题

参考目录 `data/smoke/Fskm7CbVwGzpPISH` 中的旧 RGB 实际尺寸是 640 × 480，
但旧目录保存的相机内参主点仍是 `(640, 360)`。原因是
`AutonomousLabeling.get_intrinsics_matrix()` 当前把 1280 × 720 写死，没有
使用每个 render product 的真实分辨率。

本次显式生成 1280 × 720 图像，所以图像与内参一致；但这只是规避问题，尚未
修复实现。后续应读取真实分辨率并增加“图像尺寸—主点—焦距”一致性测试。

## 5. 告警与结论边界

项目内 `--portable-root` 已消除此前外部 Conda 安装目录中的以下问题：

- `DerivedDataCache` 256 个锁耗尽；
- `new_stage` chmod 权限错误；
- material cache 无法保存。

以下告警仍存在，未阻止 smoke：

1. full base experience 在纯 headless smoke 中仍加载 GUI、示例、机械臂和传感器扩展；
2. asset converter 报告同进程存在不同 OpenUSD build，存在 ABI/singleton 风险；
3. RTX transform history 需要 102，但默认只保留 8，长曝光/运动相机可能错误；
4. synthetic data 从 texture 直接复制到 host buffer，有性能告警；
5. 无显示器的 headless 环境会报告 GLFW 初始化失败，但 `--no-window` 路径正常；

因此当前结论是：**Isaac Sim 6.1 + OmniLRS Lunalab 已具有锁定、可安装、缓存
隔离的 headless 加载、渲染、步进、可视化输出和正常退出链路。P0 的环境基线
子任务完成；最小扩展集合、机器人控制/reset 和采样吞吐量仍未验收。**
