==============
GPU 发展简史
==============

.. epigraph::

   The most dangerous phrase in the language is: "We've always done it this way."

   — Grace Hopper

GPU 从专用的图形渲染硬件演变为通用并行计算引擎，经历了三十多年的技术积累和架构变革。

萌芽期：固定功能管线（1990s）
================================

早期的图形加速卡仅实现固定功能的图形管线。在那个时代，显卡的作用仅仅是加速 CPU 的图形输出——程序员通过 DirectX 或 OpenGL 驱动接口配置各个固定功能阶段，但无法改变其内部行为。真正意义上的 GPU 概念尚未诞生。

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 1996
     - 3dfx Voodoo
     - 首款消费级 3D 加速卡
   * - 1999
     - NVIDIA GeForce 256
     - 首次提出 GPU（Graphics Processing Unit）概念
   * - 2001
     - NVIDIA GeForce 3
     - 引入可编程顶点着色器

GeForce 256 的出现具有标志性意义——它首次将"图形处理器"这个概念从 CPU 中独立出来。但此时的 GPU 仍然是一台专用的图形机器，离通用计算还很遥远。转折点来自可编程着色器的引入，它打开了 GPU 通用化的第一道门。

可编程着色器时代（2000s）
==============================

可编程着色器让 GPU 从一个"黑盒"变为"可编程的并行处理器"。最初的着色器只能运行短小的汇编程序，功能受限。随着统一着色器架构（Unified Shader Architecture）的引入，顶点和片段着色器共用 ALU 资源，GPU 的通用计算潜力开始受到学术界和工业界的关注。

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 2004
     - NVIDIA GeForce 6800 (NV40)
     - 引入可编程片段着色器
   * - 2006
     - NVIDIA GeForce 8800 (G80)
     - 统一着色器架构
   * - 2007
     - NVIDIA CUDA 1.0
     - 首次将 GPU 通用计算推向开发者
   * - 2008
     - Khronos OpenCL 1.0
     - 跨平台异构计算标准
   * - 2009
     - AMD Stream SDK
     - AMD 的首个 GPGPU 方案

2007 年是 GPU 计算历史上最重要的一年。NVIDIA 推出了 CUDA 1.0，允许开发者用 C 语言编写 GPU 程序，而不再需要通过图形 API（OpenGL/DirectX）的"曲线救国"方式。同年，AMD 也发布了 Close to Metal（CTM）接口。GPU 通用计算的时代正式开启。

GPU 计算爆发期（2010s）
==============================

2010 年代是 GPU 计算技术飞速成熟的十年。每一代架构都在解决上一代的问题，同时为即将到来的 AI 浪潮做准备。Fermi 带来了完整的 C++ 支持和 ECC 内存，Kepler 引入了动态并行，而 2012 年 AlexNet 在 GPU 上的成功训练则彻底改变了 GPU 的发展轨迹——从图形加速卡变为 AI 计算的基石。

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 2010
     - NVIDIA Fermi (GF100)
     - 完整的 ECC、L1/L2 缓存、C++ 支持
   * - 2012
     - NVIDIA Kepler (GK104)
     - SMX 设计、动态并行、Hyper-Q
   * - 2012
     - AlexNet 在 GPU 上训练成功
     - 深度学习革命的开端
   * - 2014
     - NVIDIA Maxwell (GM200)
     - 改进的共享内存架构
   * - 2016
     - NVIDIA Pascal (GP100)
     - 16nm FinFET、NVLink 1.0、HBM2
   * - 2017
     - NVIDIA Volta (GV100)
     - Tensor Core 引入，独立线程调度
   * - 2017
     - AMD ROCm 1.0 发布
     - AMD 开源 GPU 计算平台
   * - 2018
     - NVIDIA Turing (TU102)
     - RT Core、Mesh Shader、异步计算
   * - 2019
     - AMD CDNA 架构发布
     - AMD 将计算和图形架构分离

Volta 架构的 Tensor Core 是这十年最重要的硬件创新。它意识到深度学习中的矩阵运算可以用专用硬件加速到传统 CUDA Core 的数倍乃至数十倍。从此，GPU 的设计方向从"通用并行处理器"逐步转向"AI 加速器"。

AI 加速时代（2020s）
==============================

进入 2020 年代，AI 工作负载成为 GPU 设计的核心驱动力。Tensor Core 从 FP16 扩展到 BF16、TF32、FP8、FP4 等多种精度；Transformer 模型的主导地位催生了 Transformer Engine 和更大的 L2 缓存；多 GPU 系统从可选项变为标配，NVLink 和 NVSwitch 持续演进。

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 2020
     - NVIDIA Ampere (GA100)
     - 第三代 Tensor Core、MIG、L2 翻倍
   * - 2020
     - AMD MI200 (CDNA 2)
     - Matrix Core、Infinity Fabric 2.0
   * - 2022
     - NVIDIA Hopper (GH100)
     - Transformer Engine、DPX 指令、NVLink Switch
   * - 2023
     - AMD MI300 (CDNA 3)
     - 小芯片设计、CPU+GPU 统一封装
   * - 2024
     - NVIDIA Blackwell (B200)
     - NVLink 5.0、FP4/FP6 支持
   * - 2024
     - Intel Max 系列
     - Intel 进入独立 GPU 计算市场
   * - 2025
     - NVIDIA Rubin 架构
     - 第六代 Tensor Core，架构持续演进

关键架构演进脉络
======================

.. code-block:: text

   固定功能管线 → 统一着色器 → GPGPU → AI 加速器

   GPU 架构演进的核心趋势：

   1.通用化：专用电路 → 可编程 ALU → 灵活计算单元
   2.并行化：核心数持续增长，线程管理硬件日趋复杂
   3. **专用化**：Tensor Core、RT Core、Transformer Engine 等专用加速器

理解 GPU 的发展脉络，有助于预测未来的方向——更通用的计算能力、更强大的专用加速器、更紧密的系统集成。下一章将从硬件结构出发，深入 GPU 芯片的每一个角落。
   4.互联化：NVLink、Infinity Fabric 实现 GPU 间高速通信
   5.虚拟化：MIG、SR-IOV 实现硬件级资源隔离

参考与拓展阅读
====================

- NVIDIA Volta Architecture - IEEE Micro 2018 — Volta 架构详细分析
- NVIDIA Hopper Architecture - Hot Chips 34 — Hopper 架构 Hot Chips 报告
- AMD CDNA 2 Architecture - HPCA 2023 — AMD CDNA 2 架构论文
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 附录 A 提供了 GPU 发展的完整时间线
