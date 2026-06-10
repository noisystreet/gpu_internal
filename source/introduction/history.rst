==============
GPU 发展简史
==============

.. epigraph::

   The most dangerous phrase in the language is: "We've always done it this way."

   — Grace Hopper

GPU 从专用的图形渲染硬件演变为通用并行计算引擎，经历了三十多年的技术积累和架构变革。

萌芽期：固定功能管线（1990s）
================================

早期的图形加速卡仅实现固定功能的图形管线，包括顶点变换、光栅化、片段处理等阶段，每个阶段由专用的硬件单元完成，程序员无法改变其行为。

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

可编程着色器时代（2000s）
==============================

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 2004
     - NVIDIA GeForce 6800 (NV40)
     - 引入可编程片段着色器，完整支持 Shader Model 3.0
   * - 2006
     - NVIDIA GeForce 8800 (G80)
     - 统一着色器架构：顶点和片段着色器共用 ALU
   * - 2007
     - NVIDIA CUDA 1.0
     - 首次将 GPU 通用计算推向开发者
   * - 2008
     - Khronos OpenCL 1.0
     - 跨平台异构计算标准
   * - 2009
     - AMD Stream SDK / ATI Stream
     - AMD 的首个 GPGPU 方案

GPU 计算爆发期（2010s）
==============================

.. list-table::
   :header-rows: 1

   * - 年份
     - 里程碑
     - 意义
   * - 2010
     - NVIDIA Fermi (GF100)
     - 完整的 ECC、L1/L2 缓存、C++ 支持，面向 HPC
   * - 2012
     - NVIDIA Kepler (GK104)
     - SMX 设计、动态并行、Hyper-Q
   * - 2012
     - AlexNet 在 GPU 上训练成功
     - 深度学习革命的开端
   * - 2014
     - NVIDIA Maxwell (GM200)
     - 改进的共享内存架构、动态并行增强
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

AI 加速时代（2020s）
==============================

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
     - NVLink 5.0、FP4/FP6 支持、第二代 Transformer Engine
   * - 2024
     - Intel Xe / Max 系列
     - Intel 进入独立 GPU 计算市场

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
