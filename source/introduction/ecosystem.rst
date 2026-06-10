==================
GPU 生态格局
==================

.. epigraph::

   The best way to predict the future is to invent it.

   — Alan Kay

当前 GPU 计算生态由 NVIDIA、AMD 和 Intel 三大厂商主导，每个厂商在硬件架构、软件栈和生态策略上有各自的取舍。

NVIDIA 生态
================

NVIDIA 是 GPU 计算领域事实上的领导者，拥有最成熟的软硬件生态。

.. list-table::
   :header-rows: 1

   * - 层次
     - 产品/技术
     - 说明
   * - 消费级 GPU
     - GeForce RTX
     - 基于 Ada Lovelace 架构，Tensor Core + RT Core
   * - 数据中心 GPU
     - H100 / B200 / B100
     - Hopper 和 Blackwell 架构，面向 AI 训练和推理
   * - GPU 互联
     - NVLink / NVSwitch
     - 高速 GPU 间直连，拓扑全互联
   * - 虚拟化
     - MIG / vGPU
     - 硬件级 GPU 分区和虚拟化
   * - 编程模型
     - CUDA / cuDNN / TensorRT / NCCL
     - 完整的计算、AI 和通信库
   * - 开发者工具
     - Nsight Systems / Nsight Compute
     - GPU kernel 分析和性能调优

AMD 生态
================

AMD 以开源为核心策略，通过 ROCm 平台提供 NVIDIA CUDA 的替代方案。

.. list-table::
   :header-rows: 1

   * - 层次
     - 产品/技术
     - 说明
   * - 消费级 GPU
     - Radeon RX 系列
     - RDNA 3 架构
   * - 数据中心 GPU
     - MI250X / MI300X / MI350
     - CDNA 2/3 架构，配备 Matrix Core
   * - GPU 互联
     - Infinity Fabric
     - 多芯片互联，支持 Chiplet 封装
   * - 虚拟化
     - SR-IOV
     - 基于 PCIe 标准的 GPU 虚拟化
   * - 编程模型
     - ROCm / HIP / rocBLAS / RCCL
     - 开源计算平台，HIP 兼容 CUDA 语法
   * - 开发者工具
     - ROCProfiler / rocGDB / Omnitrace
     - GPU 分析和调试工具

Intel 生态
================

Intel 通过 Xe 架构进入独立 GPU 计算市场，定位 AI 推理和高性能计算。

.. list-table::
   :header-rows: 1

   * - 层次
     - 产品/技术
     - 说明
   * - 数据中心 GPU
     - Intel Max 系列 (Ponte Vecchio)
     - Xe HPC 架构，多 Tile 设计
   * - 编程模型
     - oneAPI / SYCL
     - 跨厂商统一编程模型
   * - 开发者工具
     - Intel Advisor / VTune
     - 性能分析工具

生态对比
================

三家厂商虽然在为同一个市场服务，但切入点和设计哲学差异显著。NVIDIA 追求"技术领先+生态锁定"——硬件最强、CUDA 生态最丰富，但完全封闭。AMD 走"开放追赶"路线——硬件规格亮眼、软件栈完全开源，但生态成熟度尚有差距。Intel 则押注"开放标准"——通过 oneAPI/SYCL 试图以统一编程模型打破厂商绑定。

.. list-table::
   :header-rows: 1

   * - 维度
     - NVIDIA
     - AMD
     - Intel
   * - 市场份额
     - ~90%（数据中心）
     - ~8%
     - ~2%
   * - 软件成熟度
     - 极高
     - 中（快速追赶中）
     - 低（起步阶段）
   * - AI 库生态
     - 最丰富（cuDNN, TensorRT, Triton）
     - 较丰富（MIOpen, ROCm 不断扩展）
     - 有限（oneDNN）
   * - 开源程度
     - 封闭（驱动闭源，CUDA 专有）
     - 高度开源（驱动和 ROCm 皆为开源）
     - 中等
   * - 编程语言
     - CUDA C++
     - HIP C++（类 CUDA 语法）
     - SYCL / oneAPI C++
   * - 硬件特点
     - 高性能 + 专用加速器
     - 性价比 + Chiplet 设计
     - 多核 + 统一内存

从生态对比可以看出，选择 GPU 平台不仅是选择硬件，更是选择一套技术生态。对于 AI 框架开发者而言，NVIDIA 的 CUDA 生态几乎是强制性的选择；对于希望避免供应商锁定的企业，AMD 的 ROCm 和 Intel 的 oneAPI 提供了有吸引力的替代方案，但需要在生态成熟度和独立性之间做出权衡。

AI 框架中的 GPU 支持
==============================

对于 AI 框架开发者而言，GPU 的选择与框架的 GPU 后端紧密相关：

.. list-table::
   :header-rows: 1

   * - 框架
     - NVIDIA GPU
     - AMD GPU
     - Intel GPU
   * - PyTorch
     - 原生支持（最佳）
     - ROCm 后端（社区版）
     - XPU 后端（实验性）
   * - TensorFlow
     - 原生支持（最佳）
     - ROCm 后端
     - oneAPI 后端
   * - JAX
     - 原生支持（CUDA）
     - 实验性（ROCm）
     - 不支持
   * - vLLM / TensorRT-LLM
     - 仅 NVIDIA
     - 不支持
     - 不支持

这意味着 AI 开发者如果希望使用最新的推理优化框架，NVIDIA GPU 几乎是唯一选择。这也是 NVIDIA 生态粘性最强的部分——不仅仅是硬件性能，更是上层软件栈的深度绑定。

其他参与者
================

除了三大厂商，还有一些值得关注的 GPU 计算生态参与者：

- **Apple Metal**: Apple Silicon (M1/M2/M3/M4) 集成的 GPU，通过 Metal API 访问，用于 macOS/iOS 生态
- **Qualcomm Adreno**: 移动端 GPU，通过 OpenCL/Vulkan 支持通用计算
- **Arm Mali / Immortalis**: 移动端和嵌入式 GPU，通过 OpenCL 支持
- **摩尔线程 / 壁仞 / 寒武纪**: 国产 GPU/AI 加速器，各自构建封闭或开源生态

参考与拓展阅读
====================

- Intel oneAPI Specification (https://www.oneapi.io/spec/) — Intel oneAPI 跨厂商统一编程规范
- AMD ROCm Documentation (https://rocm.docs.amd.com/) — AMD ROCm 开源计算平台文档
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 第 1 章提供 GPU 生态的系统性背景介绍
