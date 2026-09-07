# CS231n Local and Cloud GPU Workflow Design

## Goal

将 Assignment 1-3 从依赖 Google Colab/Google Drive 的启动方式改造成可在本地 Jupyter 环境运行，并让需要训练加速的 Notebook 能在本地 GPU、Apple Silicon MPS、CPU 或云端 GPU 上使用同一套路径和设备配置。

## Scope

本次改造覆盖：

- 根目录 Python 环境、Jupyter 启动命令和跨 assignment 的开发说明。
- Assignment 1 的 CIFAR-10/小型 ImageNet 数据下载与本地路径。
- Assignment 2 的 CIFAR-10、COCO captioning 数据下载与本地路径。
- Assignment 3 的 CIFAR-10、COCO captioning、DDPM/CLIP/DINO 所需资源的路径与设备入口。
- 所有教学 Notebook 的 Colab 初始化单元格，使其在本地直接运行，并在检测到 Colab 时保留兼容行为。
- Submission notebook 的本地执行入口；生成 zip/PDF 的具体提交格式保持不变。

不在本次范围内：

- 修改课程题目、评分逻辑、模型结构或学生 TODO。
- 将大型 COCO、预训练模型或 DDPM 训练强制搬到本地。
- 重新设计各 assignment 的 `cs231n` 包；只做路径、环境和设备所需的最小改动。

## Architecture

根目录提供一个统一的项目环境与配置约定。`CS231N_ROOT` 指向仓库根目录，`CS231N_DATA_ROOT` 可选地指向外置数据盘；未设置时数据放在各 assignment 的 `cs231n/datasets` 下。Notebook 的第一段初始化代码调用一个轻量配置模块，计算当前 assignment 根目录、加入 `sys.path`、选择数据路径并报告设备，不再依赖固定的 `/content/drive`。

每个 assignment 继续保留自己的 `cs231n` 包，避免改变课程代码的导入语义。数据下载脚本使用仓库内的相对路径并具备幂等性；网络不可用或资源未准备好时，脚本给出可执行的下一步提示，而不是让 Notebook 在不透明的路径错误处失败。

设备选择统一遵循 `CS231N_DEVICE` 覆盖优先级：显式配置（如 `cuda`, `cuda:0`, `mps`, `cpu`）优先；未配置时按 CUDA、MPS、CPU 顺序自动检测。只有实际使用 PyTorch 的 Notebook 才依赖该设备入口，NumPy 作业仍可在 CPU 上完成。

## User workflow

本地首次使用：

1. 创建环境并安装根目录依赖。
2. 运行按 assignment 分组的数据准备命令；CIFAR-10 可本地下载，COCO/预训练模型按需要下载到数据盘。
3. 从仓库根目录启动 Jupyter，打开任意 Notebook 并从上到下运行。

云端 GPU 使用同一份 Notebook：将仓库同步到云端，设置 `CS231N_ROOT`/`CS231N_DATA_ROOT`，安装同一份依赖，并设置 `CS231N_DEVICE=cuda`（或依赖自动检测）。Notebook 不应再要求用户编辑 Drive 文件夹字符串。

## Data and path contract

- `CS231N_ROOT`: 可选。默认是配置模块根据 Notebook 当前文件位置推导出的仓库根目录。
- `CS231N_DATA_ROOT`: 可选。用于把大数据集放到外部磁盘；其下按 `assignment1`, `assignment2`, `assignment3` 分目录，未存在时回退到各 assignment 的 `cs231n/datasets`。
- `CS231N_DEVICE`: 可选。允许 `auto`, `cpu`, `mps`, `cuda`, `cuda:N`；非法或不可用设备要给出清晰错误。
- 所有路径通过 `pathlib.Path` 构造，Notebook 改变工作目录时必须使用绝对路径或配置对象提供的路径。

## Compatibility and error handling

Colab 检测只作为兼容分支：在可导入 `google.colab` 时可以挂载 Drive，但本地运行不能导入该模块。默认路径不能包含机器特定的用户名、`/content` 或 `My Drive`。数据准备脚本在目标目录已有完整数据时跳过下载，在部分下载时使用临时文件并在成功解压后清理压缩包。

如果 CUDA/MPS 不可用，轻量测试和 NumPy/CPU 作业应继续运行；需要长时间训练的 Notebook 在 CPU 上启动时显示设备和性能提示，但不改变课程代码的数值结果。缺少 COCO 或预训练权重时，Notebook 在加载资源前抛出包含实际路径和对应准备命令的错误。

## Validation

验证分三层：

1. 配置单元测试：覆盖默认根目录、外置数据根目录、设备自动检测、显式设备和非法设备。
2. 静态 Notebook 检查：确认教学 Notebook 不再包含无条件的 `google.colab`、`/content/drive` 或未定义的 `FOLDERNAME` 路径；所有 import setup 单元可在本地 Python 内核执行。
3. 最小运行检查：在不下载大型数据的环境中导入三份 `cs231n` 包，运行 Assignment 1 的 CIFAR 路径检查和 Assignment 2/3 的设备选择 smoke test；数据完整时再运行现有轻量梯度/shape 单元。

## Deliverables

- 根目录环境与启动文档、依赖定义和配置模块。
- 按 assignment 分组的数据准备脚本与路径适配。
- 更新后的 Notebook 初始化和 submission 入口。
- 自动化检查脚本/测试及运行说明。

