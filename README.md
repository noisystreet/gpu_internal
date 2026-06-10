# 深入理解 GPU

[![Documentation Status](https://readthedocs.org/projects/gpu-internal/badge/?version=latest)](https://gpu-internal.readthedocs.io/zh-cn/latest/)

系统化的 GPU 底层原理教程，使用 reStructuredText 编写，通过 Sphinx 构建为 HTML/PDF。

在线文档：https://gpu-internal.readthedocs.io/zh-cn/latest/

## 内容

教程覆盖 GPU 的六大核心领域：

| 章节 | 内容 |
|------|------|
| **绪论** | GPU 发展历史、生态格局、全栈层次概览 |
| **硬件结构** | GPU 芯片微架构、SM/CU、内存层次结构、Tensor Core |
| **执行模型** | Kernel 执行、Warp/Wavefront 调度、内存访问模式 |
| **驱动** | UMD/KMD 分层架构、命令提交、主机-设备通信 |
| **运行时** | CUDA Runtime、ROCm/HIP、Vulkan Compute |
| **GPU 互联拓扑** | NVLink/NVSwitch、Infinity Fabric、拓扑感知编程 |
| **GPU 虚拟化** | MIG、SR-IOV、vGPU、GPU 池化 |

## 快速开始

```bash
pip install -r requirements.txt
make html
```

构建产物位于 `_build/html/`，打开 `index.html` 即可浏览。

## 目录结构

```
├── index.rst                 # 文档入口
├── conf.py                   # Sphinx 配置
├── agents.md                 # AI Agent 协作指南
├── source/
│   ├── introduction/         # 绪论（5 篇）
│   ├── hardware/             # 硬件结构（4 篇）
│   ├── execution_model/      # 执行模型（4 篇）
│   ├── driver/               # 驱动（3 篇）
│   ├── runtime/              # 运行时（4 篇）
│   ├── gpu_interconnect/     # GPU 互联拓扑（4 篇）
│   ├── gpu_virtualization/   # GPU 虚拟化（4 篇）
│   └── appendix/             # 附录
```

## 许可

本教程采用 [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 许可证。
