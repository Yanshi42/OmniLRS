# OmniLRS-native RL 基线环境

> 状态：P0 已完成——可重建环境基线、smoke、Task-v0 与 benchmark 协议均已冻结
> 最新验证日期：2026-10-14
> 适用 worktree：`/home/spatial/OmniLRS-isaac6`
> 开发分支：`feature/isaac6-native-rl`

本文记录 P0 阶段对仓库、宿主机、GPU/CUDA 和 Python/Isaac Sim
环境的实测结果。除非标为“声明值”，表中的版本均由下文命令在本机直接采集。

仓库内 Pixi 环境的最小 headless 功能 smoke 已通过，证据见
[`headless_smoke_isaac61.md`](headless_smoke_isaac61.md)。该 smoke 只覆盖
Lunalab 加载、步进、可视化写出和退出，尚不覆盖机器人控制、局部 reset 或
RL 采样吞吐量。

## 1. 版本快照

### 1.1 仓库

| 项目 | 实测值 |
|---|---|
| worktree | `/home/spatial/OmniLRS-isaac6` |
| 分支 | `feature/isaac6-native-rl` |
| OmniLRS commit | `eb1d3362e6c5b9bae5bedfa957bf920ad6021455` |
| WorldBuilders submodule | `23bd494b1bbcdd54cf314250a4312352d285c712` |
| 采集前工作树 | clean |
| OmniLRS Python package | `2.0.0`，editable install 指向本 worktree |

### 1.2 宿主机与 GPU

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble) |
| Kernel | Linux 6.8.0-137-generic x86_64 |
| glibc | 2.39 |
| GPU | 8 × NVIDIA GeForce RTX 4090 |
| 每张 GPU 由驱动报告的显存 | 49140 MiB |
| NVIDIA driver | 595.84 |
| `nvidia-smi` 报告的 CUDA compatibility | 13.2 |
| 系统 `nvcc` | CUDA Toolkit 13.0, V13.0.48 |

`nvidia-smi` 的 “CUDA Version” 是驱动支持的最高 CUDA 兼容版本，不等于
Isaac Sim 或 PyTorch 实际装载的 CUDA runtime。GPU 利用率和已用显存是动态
状态，不属于冻结基线；后续 benchmark 必须记录 GPU 编号和测量时的占用。

### 1.3 冻结的 Python/Isaac Sim 环境

唯一推荐入口是仓库内 Pixi 环境：

```text
/home/spatial/OmniLRS-isaac6/.pixi/envs/default/bin/python
```

环境由仓库的 `pixi.toml` 与 `pixi.lock` 创建，不依赖此前
`.venv-isaac61` 指向的外部 Conda site-packages。后者只保留为历史对照。

| Python 包/组件 | 实测值 |
|---|---|
| CPython | 3.12.15 |
| Isaac Sim metapackage | 6.1.0.0 |
| `isaacsim-app` | 6.1.0.0 |
| `isaacsim-core` | 6.1.0.0 |
| `isaacsim-kernel` | 6.1.0.0 |
| `isaacsim-rl` | 6.1.0.0 |
| `isaacsim-robot` | 6.1.0.0 |
| `isaacsim-ros2` | 6.1.0.0 |
| `isaacsim-sensor` | 6.1.0.0 |
| `isaacsim-replicator` | 6.1.0.0 |
| NumPy | 2.3.1 |
| `packaging` | 26.0 |
| PyTorch | 2.11.0+cu130 |
| Warp Python distribution | 1.18.0 |
| Warp Kit runtime | 1.16.0, CUDA Toolkit 12.9 |
| Numba | 0.63.1 |
| llvmlite | 0.46.0 |
| numpy-quaternion | 2024.0.13 |
| Coverage | 7.16.2 |
| Hydra | 1.3.2 |
| OmegaConf | 2.3.0 |
| OpenCV import | 可用，4.14.0 |
| GDAL Python import | 可用，3.13.3 |
| Gymnasium | **未安装** |
| skrl | **未安装** |

Pixi activation 自动设置 `OMNI_KIT_ACCEPT_EULA=YES`、
`OMNILRS_ROOT` 和 `OMNILRS_KIT_ROOT`。运行时缓存、日志和 shader cache
落在仓库 `.cache/isaacsim`，不再写入 Python 包安装目录。

## 2. 声明、锁文件与运行环境的一致性

`pixi.toml` 与 `pixi.lock` 现已统一为 Python 3.12 和 Isaac Sim 6.1。

为使 Isaac 6.1 在 Pixi 0.81 下可求解、可导入，清单显式记录：

- `packaging == 26.0`，满足 `isaacsim-core==6.1.0.0`；
- `tinyobjloader == 2.0.0rc13`，显式允许 Isaac 固定的预发布版本；
- `numpy-quaternion == 2024.0.13`，避免旧 wheel 的 NumPy 1.x ABI；
- `numba == 0.63.1`、`llvmlite == 0.46.0`、`coverage == 7.16.2`，
  使用已经过 smoke 的兼容组合；
- `OMNILRS_KIT_ROOT=.cache/isaacsim`，隔离 Kit 可写缓存。

一致性证据：

```text
pixi install -e default  -> exit 0
pixi lock --check        -> Lock-file was already up-to-date
Isaac Sim metadata       -> 6.1.0.0
GDAL/OpenCV import       -> 通过
headless SDG smoke       -> exit 0
```

结论：**仓库已经有与 Isaac Sim 6.1/Python 3.12 匹配、可安装、可重复启动的
锁定环境；P0 的单一环境基线子任务已验收。**

## 3. 可复现盘点命令

所有命令从 `/home/spatial/OmniLRS-isaac6` 执行。

### 3.1 Git 与系统

```bash
git status --short --branch
git rev-parse HEAD
git submodule status
uname -a
sed -n '1,20p' /etc/os-release
ldd --version | head -1
```

### 3.2 GPU、驱动与 CUDA Toolkit

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
nvidia-smi
nvcc --version
```

### 3.3 Pixi 环境与关键导入

```bash
env PIXI_HOME=/home/spatial/.local/share/pixi pixi install -e default
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python --version
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python -c 'import importlib.metadata as m; print(m.version("isaacsim"))'
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python -c 'import cv2, quaternion; from osgeo import gdal; print(cv2.__version__, gdal.__version__)'
```

### 3.4 Pixi 一致性检查

```bash
pixi --version
env PIXI_HOME=/home/spatial/.local/share/pixi pixi lock --check
env PIXI_HOME=/home/spatial/.local/share/pixi pixi run -e default python --version
```

验证时 Pixi 版本为 0.81.0；锁文件检查、安装和运行均成功。glibc 2.35 最低
要求使用 Pixi 的命名虚拟平台声明，不再使用已弃用的 `[system-requirements]` 表。

## 4. P1 入口与尚未取得的动态证据

1. P1 首个实现窗口加入 Gymnasium；skrl 是否安装留到 P3，不能为了 smoke test 提前
   引入训练算法。
2. 最小 headless 启动、Lunalab 加载、步进、可视化写出和退出已经通过；下一项
   smoke 需覆盖机器人加载、固定物理步数和无渲染步进。
3. Task-v0、Observation-v0、Action-v0、Reward-v0、坐标/单位和 reset 语义已经
   冻结，见 [`task_v0_spec.md`](task_v0_spec.md)。
4. benchmark protocol 和 P1 数值 Gate 已冻结，见
   [`p1_benchmark_protocol.md`](p1_benchmark_protocol.md)。
5. 正式 benchmark 在每个全新进程启动前自动选择满足连续 10 秒独占检查的空闲
   RTX 4090，不固定物理 index；每次必须记录实际 index、UUID、型号和选择快照。
