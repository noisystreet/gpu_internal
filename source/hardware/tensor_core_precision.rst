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

Tensor Core 代际设计演进
========================

Tensor Core 从 Volta 到 Blackwell 经历了五代演进，每一代都在精度、灵活性和性能上做出了关键改进。

**第 1 代：Volta（2017）—— 从零到一**

Volta (SM 7.0) 首次引入 Tensor Core，专为深度学习矩阵乘法设计。其核心设计决策是 **warp 级协作 MMA**——整个 warp（32 线程）共同完成一次 ``D = A × B + C`` 的矩阵分块运算。

.. code-block:: text

   关键限制:
   - 仅支持 FP16 输入 → FP32 累加，单一矩阵形状 16×16×16
   - 编程接口仅有 WMMA API（高层 fragment 抽象）
   - 仅用于 GEMM，卷积需要 im2col 转换
   - 精度约 1-2 ULP，不遵循 IEEE 754

   Volta 的 Tensor Core 实现了 ~125 TFLOPS (FP16)，
   相比 CUDA Core FP32 的 ~15 TFLOPS 提升约 8x。

第 2 代：Turing（2018）—— 整数量化

Turing (SM 7.5) 将 Tensor Core 推向推理场景，首次引入 INT8 和 INT4 量化支持：

.. code-block:: text

   Turing 的设计重点:
   - INT8/INT4 精度服务于推理量化
   - 每 SM Tensor Core 数不变（8 个），但利用 INT8 吞吐翻倍
   - Turing 在设计上更侧重图形和实时光线追踪
   - 并未增加新的矩阵形状或稀疏支持

   T4 GPU 的 INT8 Tensor Core 峰值 ~130 TOPS，
   成为云端推理的首选硬件。

**第 3 代：Ampere（2020）—— 灵活性与 BF16**

Ampere (SM 8.0) 是 Tensor Core 架构最重要的迭代，解决了 Volta/Turing 的多个核心限制：

.. code-block:: text

   主要改进:
   1. 灵活矩阵形状 — 不再限于 16×16×16
      - m16n8k16: 主流 GEMM 优化
      - m16n8k8: BF16 更小内积维度
      - m8n8k4: 小矩阵快速计算
   2. BF16 支持 — 保留 FP32 动态范围，无需混合精度缩放
   3. TF32 格式 — 19 位精度，输入 FP32 自动截断，无需改代码
   4. 2:4 稀疏支持 (mma.sp) — 结构化稀疏 2x 加速
   5. 低精度 PTX (mma API) — 替代 WMMA 的底层接口

   代价: 每 SM Tensor Core 从 8 减至 4 个，
   但总 TFLOPS 从 125 提升到 312 (FP16)。

**第 4 代：Hopper（2022）—— 更大的规模**

Hopper (SM 9.0) 的目标是突破 Ampere 的吞吐天花板，通过增加操作粒度降低指令发射开销：

.. code-block:: text

   关键改进:
   1. Warpgroup MMA (wgmma) — 4 个 warp 协作完成更大的矩阵分块
      减少指令发射次数，提升每指令的计算密度
   2. FP8 支持 (E4M3 / E5M2) — 进一步降低内存带宽需求
      专为 LLM 训练设计，支持混合 FP8 训练
   3. TMA (Tensor Memory Accelerator) — 独立硬件单元管理数据加载
      warp 不再需要参与数据搬运，专注计算
   4. DPX 指令 — 在 Tensor Core 中加速动态规划算法
      (如 Smith-Waterman DNA 序列比对)

   H100 的 Tensor Core FP16 峰值 ~989 TFLOPS，
   相比 A100 提升 ~3.2x。

**第 5 代：Blackwell（2024）—— 极限压缩**

Blackwell 延续了精度越降越低的趋势，首次引入 FP6 和 FP4：

.. code-block:: text

   关键改进:
   1. FP6 支持 — 在 FP8 和 FP4 之间的折中精度
      适用于对精度敏感但需要超过 FP8 压缩比的场景
   2. FP4 支持 (E3M0) — 极限压缩，推理场景
      单卡可加载更大模型，减少 GPU 间通信
   3. 第二代 Transformer Engine — 自动管理 FP4/FP6/FP8 精度切换
   4. 每 SM Tensor Core 架构未公开，但总吞吐大幅提升

   B200 的 FP4 Tensor Core 峰值 ~9000 TFLOPS，
   相比 H100 FP8 提升 ~4.5x。

**代际演进的核心矛盾**：

.. list-table::
   :header-rows: 1

   * - 矛盾维度
     - 早期（Volta/Turing）
     - 中期（Ampere）
     - 近期（Hopper/Blackwell）
   * - 精度策略
     - 少数固定精度
     - 多种精度可选
     - 自动精度管理
   * - 编程抽象
     - WMMA 高层 API
     - mma PTX 底层指令
     - wgmma + TMA 硬件管理
   * - 矩阵形状
     - 单一 16x16x16
     - 灵活多形状
     - Warpgroup 聚合
   * - 使用方式
     - 仅 cuBLAS/cuDNN
     - 开放给 CUTLASS/WMMA
     - TMA 异步数据流
   * - 应用场景
     - 深度学习训练
     - 训练 + 推理 + HPC
     - LLM + 科学计算 + 推理

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
