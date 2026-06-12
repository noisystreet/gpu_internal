========================
Tensor Core 精度与代际
========================

支持的精度与代际演进
========================

.. list-table::
   :header-rows: 1

   * - 代际
     - 架构
     - 支持的精度
     - 每 SM Tensor Core 数
     - 峰值 TFLOPS (FP16)
   * - 第 1 代
     - Volta (V100)
     - FP16
     - 8
     - 125
   * - 第 2 代
     - Turing (T4)
     - FP16, INT8, INT4
     - 8
     - 65
   * - 第 3 代
     - Ampere (A100)
     - FP16, BF16, TF32, INT8, INT4
     - 4
     - 312
   * - 第 4 代
     - Hopper (H100)
     - FP16, BF16, TF32, FP8, INT8
     - 4
     - 989
   * - 第 5 代
     - Blackwell (B200)
     - FP16, BF16, TF32, FP8, FP6, FP4, INT8
     - N/A
     - 2250

精度格式详解
================

.. list-table::
   :header-rows: 1

   * - 格式
     - 位宽
     - 指数/尾数
     - 引入架构
     - 典型用途
   * - FP16
     - 16
     - 5/10
     - Volta
     - 训练中间精度
   * - BF16
     - 16
     - 8/7
     - Ampere
     - 训练（更大动态范围）
   * - TF32
     - 19
     - 8/10
     - Ampere
     - 无需改代码的加速
   * - FP8 E4M3
     - 8
     - 4/3
     - Hopper
     - 推理/训练（范围有限）
   * - FP8 E5M2
     - 8
     - 5/2
     - Hopper
     - 推理/训练（范围更大）
   * - INT8
     - 8
     - -/7+符号
     - Turing
     - 推理量化
   * - FP4
     - 4
     - 3/0 (E3M0)
     - Blackwell
     - 推理（极限压缩）

Tensor Core vs CUDA Core 吞吐对比
======================================

以 H100 SXM 为例，不同精度下的峰值吞吐：

.. figure:: /source/figures/tensor_vs_cuda_throughput.svg
   :width: 85%
   :align: center
   :alt: Tensor Core vs CUDA Core 吞吐对比

   H100 SXM 上 Tensor Core 与 CUDA Core 在各种精度下的峰值吞吐对比。
   Tensor Core 在 FP16/BF16/TF32 精度下提供 ~15x 的吞吐提升。注意 TF32 无需修改代码即可获得大幅加速。

使用场景
============

.. list-table::
   :header-rows: 1

   * - 场景
     - 运算类型
     - 推荐精度
     - 性能收益
   * - 深度学习训练
     - GEMM (全连接/卷积)
     - FP16/BF16
     - 5-10x vs FP32 CUDA Core
   * - AI 推理
     - GEMM + Attention
     - FP8/INT8/INT4
     - 10-30x vs FP32
   * - HPC 科学计算
     - 矩阵运算
     - FP64 (少量 Tensor Core)
     - 1-2x
   * - 图形渲染
     - 神经网络渲染
     - FP16
     - 3-5x

Hopper Tensor Core 增强
============================

Hopper 架构引入了几项重要改进：

.. code-block:: text

   1. DPX (Dynamic Programming Accelerator) 指令
      - 加速动态规划算法（如 DNA 序列比对）
      - 在 Tensor Core 中实现

   2. Tensor Memory Accelerator (TMA)
      - 硬件单元，独立管理 Tensor Core 的数据加载
      - 减少 warp 参与数据搬运，提升利用效率
      - 支持异步多播（asynchronous multicast）

   3. Transformer Engine
      - 自动选择 FP8/FP16 精度
      - 当检测到超出 FP8 范围时自动回退到 FP16
      - 对 LLM 训练提供约 2x 吞吐提升

   4. Warpgroup MMA (``wgmma``)
      - 允许 4 个 warp（128 线程）协作执行单次 MMA 操作
      - 相比 warp 级 mma，单次操作的计算量翻倍
      - 减少指令发射次数，提升整体吞吐

参考与拓展阅读
====================

- NVIDIA H100 Tensor Core GPU Architecture — Hopper 白皮书
- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Tensor Core 编程参考
