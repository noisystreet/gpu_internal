# 深入理解 GPU — 项目说明

## 项目概述

本项目是一份系统化的 GPU 底层原理教程，使用 reStructuredText (RST) 格式编写，可通过 Sphinx 构建为 HTML/PDF 等格式的文档。

教程内容覆盖 GPU 的六大核心领域：

1. **绪论** — GPU 发展历史、生态格局、全栈层次概览
2. **硬件结构** — GPU 芯片微架构、内存层次结构、计算单元设计
3. **执行模型** — Kernel 执行模型、线程调度、Warp/Wavefront 机制、内存访问模式
4. **驱动** — GPU 驱动架构、用户态/内核态通信、命令提交与同步
5. **运行时** — CUDA Runtime、ROCm Runtime、Vulkan Compute 等编程接口
6. **GPU 互联拓扑** — NVLink/NVSwitch、Infinity Fabric、拓扑感知编程
7. **GPU 虚拟化** — MIG、SR-IOV、vGPU、GPU 池化

## 目录结构

```
gpu_internal/
├── agents.md                 # 本文件 — 项目说明与 AI Agent 指南
├── conf.py                   # Sphinx 配置文件
├── requirements.txt          # Python 依赖
├── Makefile                  # Linux 构建脚本
├── make.bat                  # Windows 构建脚本
├── index.rst                 # 文档首页 / 入口
└── source/                   # 文档源文件
    ├── introduction/         # 绪论
    │   ├── index.rst
    │   ├── history.rst
    │   ├── ecosystem.rst
    │   ├── stack_overview.rst
    │   └── how_to_read.rst
    ├── hardware/             # 硬件结构
    │   ├── index.rst
    │   ├── architecture.rst
    │   ├── memory_hierarchy.rst
    │   ├── sm_architecture.rst
    │   ├── cuda_core.rst
    │   ├── tensor_core_architecture.rst
    │   ├── tensor_core_wmma_vs_mma.rst
    │   ├── tensor_core_numerical.rst
    │   ├── tensor_core_precision.rst
    │   ├── amd_cu.rst
    │   └── occupancy.rst
    ├── execution_model/      # 执行模型
    │   ├── index.rst
    │   ├── kernel.rst
    │   ├── warp_wavefront.rst
    │   ├── memory_access.rst
    │   └── async_pipeline.rst
    ├── driver/               # 驱动
    │   ├── index.rst
    │   ├── architecture.rst
    │   ├── firmware.rst
    │   ├── communication.rst
    │   └── driver_open_source.rst
    ├── runtime/              # 运行时
    │   ├── index.rst
    │   ├── cuda_runtime.rst
    │   ├── rocm_runtime.rst
    │   └── vulkan_compute.rst
    ├── gpu_interconnect/     # GPU 互联拓扑
    │   ├── index.rst
    │   ├── nvlink_nvswitch.rst
    │   ├── infinity_fabric.rst
    │   ├── cache_coherence.rst
    │   └── topology_awareness.rst
    ├── gpu_virtualization/   # GPU 虚拟化
    │   ├── index.rst
    │   ├── overview.rst
    │   ├── mig.rst
    │   └── sr_iov.rst
    └── appendix/             # 附录
        ├── index.rst
        ├── glossary.rst
        └── references.rst
```

## 构建方法

```bash
# 安装依赖
pip install -r requirements.txt

# 构建 HTML
make html

# 构建 PDF（需安装 LaTeX）
make latexpdf
```

## AI Agent 协作约定

- 所有文档使用 **reStructuredText** 格式，中文撰写，专业术语保留英文原文（如 `CUDA`、`Warp`、`SM`），首次出现时在括号中标注全称。
- 代码示例使用 `.. code-block::` 指令，指定语言（cuda, cpp, python, bash 等）。
- 图表使用 `.. figure::` 指令，图片存放于各章节同级的 `figures/` 子目录。
- 数学公式使用 `:math:` 或 `.. math::` 指令。
- 交叉引用使用 `:ref:` 或 `:doc:` 标签。
- 每一章都应有 `index.rst` 作为 toctree 入口。
- 文档风格：技术准确、循序渐进、配有示意图和代码示例。
- AI Agent 在新增章节时，需同时更新 `index.rst` 的 toctree 和本文件的目录结构说明。

## 内容规范

| 规范项 | 要求 |
|--------|------|
| 语言 | 中文正文，英文术语保留不翻译 |
| 格式 | RST (reStructuredText) |
| 一级标题 | `===============` |
| 二级标题 | `---------------` |
| 三级标题 | `~~~~~~~~~~~~~~~` |
| 代码块 | `.. code-block:: <lang>` |
| 图片 | `.. figure:: figures/<name>.png` |
| 引用 | `:ref:` / `:doc:` |
| 术语 | 首次出现标注英文全称，如：流多处理器（Streaming Multiprocessor, SM） |
