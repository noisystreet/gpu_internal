=============================
特殊函数单元（SFU）
=============================

.. epigraph::

   The purpose of computing is insight, not numbers.

   — Richard Hamming, 数学家、计算机科学先驱

特殊函数单元（Special Function Unit, SFU）是 SM 中专用于计算**超越函数**（sin、cos、log、exp、rsqrt 等）的硬件单元。与 CUDA Core 处理通用算术运算不同，SFU 通过专用的算法和硬件实现这些复杂函数，在精度可接受的前提下实现远高于软件模拟的吞吐。

SFU 的微架构
=====================

SFU 不使用通用的 FMA 流水线来计算超越函数。相反，它采用**多项式逼近**（polynomial approximation）结合**查找表**（lookup table）的专用硬件设计：

.. mermaid::

   flowchart LR
       OP["操作数 x"] --> CONF["范围规约<br/>Range Reduction"]
       CONF --> LUT["查找表<br/>系数 ROM"]
       CONF --> POLY["多项式求值<br/>(minimax 多项式)"]
       LUT --> POLY
       POLY --> RECON["范围重建<br/>Range Reconstruction"]
       RECON --> RES["结果"]

       style OP fill:#e3f2fd,color:#1565c0
       style LUT fill:#fff3e0,color:#e65100
       style POLY fill:#f3e5f5,color:#7b1fa2
       style RES fill:#e8f5e9,color:#1b5e20

计算过程（以 ``sin(x)`` 为例）：

.. code-block:: text

   1. 范围规约（Range Reduction）
      将 x 映射到 [0, π/2] 区间内：
      x' = x mod (π/2)
      同时记录象限信息

   2. 查找表访问
      从 ROM 中读取该区间上的 minimax 多项式系数：
      P0, P1, P2, ..., Pn

   3. 多项式求值
      通过 FMA 流水线计算：
      result = P0 + P1·x' + P2·x'² + ... + Pn·x'ⁿ
      （通常使用 4-8 阶多项式）

   4. 范围重建
      根据象限信息调整符号和函数（sin ↔ cos）

**关键设计特点**：

- **minimax 多项式**：最小化最大绝对误差（而非均方误差），保证所有输入下的误差上界
- **系数 ROM**：存储预计算的系数，每个函数对应一组系数
- **与 CUDA Core 解耦**：SFU 的 FMA 流水线与 CUDA Core 的 FMA 独立，可并行发射

SFU 指令集
=================

NVIDIA GPU 中可通过 SFU 执行的指令：

.. list-table::
   :header-rows: 1

   * - PTX 指令
     - 函数
     - 最大 ULP 误差
     - 说明
   * - ``sin.approx.f32``
     - 正弦
     - ~2 ULP
     - [-π, π] 范围
   * - ``cos.approx.f32``
     - 余弦
     - ~2 ULP
     - [-π, π] 范围
   * - ``lg2.approx.f32``
     - 以 2 为底的对数
     - ~1 ULP
     - 输入 > 0
   * - ``ex2.approx.f32``
     - 2 的幂次
     - ~1 ULP
     - 指数范围 [-128, 128]
   * - ``rsqrt.approx.f32``
     - 平方根倒数
     - ~2 ULP
     - 输入 > 0
   * - ``rcp.approx.f32``
     - 倒数
     - ~1 ULP
     - 非零输入

.. note::

   SFU 的结果并非 IEEE 754 精确舍入，而是**近似值**。大多数 SFU 指令的误差在 1-2 ULP（Unit in the Last Place）范围内——对于大多数图形和计算场景已足够，但某些数值敏感的算法需要额外处理。

吞吐与延迟
=================

SFU 的吞吐远低于 CUDA Core 的基本算术指令，但远高于通过软件循环模拟超越函数。以下为 Ampere A100 的数据（每 subcore）：

.. list-table::
   :header-rows: 1

   * - 指令类型
     - 每 subcore / 周期
     - 延迟（周期）
     - 对比：FP32 FMA
   * - sin / cos
     - 1 条
     - ~16
     - 4
   * - log / exp
     - 1 条
     - ~16
     - 4
   * - rsqrt
     - 1 条
     - ~8
     - 4
   * - rcp
     - 1 条
     - ~4
     - 4

.. note::

   ``rsqrt`` 和 ``rcp`` 的延迟更低，因为它们实现较简单、多项式阶数更少。
   每 SM 每周期的 SFU 吞吐 = 每 subcore 1 条 × 4 subcore = 4 条指令。

SFU 的代际演进
=====================

SFU 的设计在 GPU 各代架构中持续演进：

.. list-table::
   :header-rows: 1

   * - 架构
     - 每 SM SFU 数
     - 精度变化
     - 新增函数
   * - Maxwell (GM200)
     - 4
     - ~2 ULP
     - 基础四则
   * - Pascal (GP100)
     - 4
     - ~2 ULP
     - -
   * - Volta (GV100)
     - 4
     - ~1-2 ULP
     - 改进的 log 精度
   * - Turing (TU102)
     - 4
     - ~1-2 ULP
     - -
   * - Ampere (GA100)
     - 4
     - ~1-2 ULP
     - - 
   * - Hopper (GH100)
     - 4
     - ~1-2 ULP
     - DPX 指令（TMA 协助）

Hopper 架构新增的 DPX 指令（Dynamic Programming Accelerator）在 SFU 的基础上扩展了更多动态规划算法支持（如 Smith-Waterman 序列比对），但这不是传统的 SFU 职能，而是 SFU 资源的再利用。

SFU 的编程模型
=====================

**直接使用内建函数**：

CUDA 提供 ``__sinf()``、``__cosf()``、``__logf()``、``__expf()`` 等内建函数，编译器将它们映射到 SFU 指令：

.. code-block:: cuda

   // CUDA SFU 内建函数（映射到 SFU 硬件）
   float s = __sinf(x);   // SFU sin.approx
   float c = __cosf(x);   // SFU cos.approx
   float l = __logf(x);   // SFU lg2.approx * log(2) 换算
   float e = __expf(x);   // SFU ex2.approx * log2(e) 换算
   float r = __rsqrtf(x); // SFU rsqrt.approx

.. warning::

   这些以双下划线开头的内建函数对应的是**低精度 SFU 近似**（~1-2 ULP）。
   如果需要 IEEE 754 精确舍入的结果，应使用标准 ``sinf()``、``cosf()`` 等（精度更高但速度慢 ~5-10x）。

**精度 vs 性能的权衡**：

.. list-table::
   :header-rows: 1

   * - 选项
     - 精度
     - 性能（相对）
     - 适用场景
   * - 标准数学库 ``sinf()``
     - IEEE 754 精确
     - 1x（基准）
     - 数值计算、物理模拟
   * - SFU 内建 ``__sinf()``
     - ~1-2 ULP
     - ~5-10x 更快
     - 图形着色器、实时渲染
   * - 自定义多项式近似
     - 可定制（0.5-4 ULP）
     - ~2-5x 更快
     - 特定区间优化、AI 激活函数

**编程示例：SFU 与标准库性能对比**：

.. code-block:: cuda
   :linenos:

   __global__ void sfu_vs_stdlib(float* in, float* sfu_out, float* std_out, int N) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx >= N) return;

       float x = in[idx];

       // SFU 路径（~4 周期延迟）
       float sfu_result = __sinf(x);

       // 标准数学库路径（~20-40 周期延迟）
       float std_result = sinf(x);

       sfu_out[idx] = sfu_result;
       std_out[idx] = std_result;
   }

   // 分析：使用 Nsight Compute 对比两个路径的吞吐差异
   // ncu --kernel-name sfu_vs_stdlib --set full ./my_app

**AI 激活函数中的 SFU**：

SFU 在深度学习推理中扮演了重要角色——许多激活函数可以直接或间接利用 SFU：

.. code-block:: cuda

   // GELU 激活函数（LLM 中广泛使用）——借助 SFU 加速
   __device__ float gelu_sfu(float x) {
       // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
       // 利用 SFU rsqrt 和 exp 加速 erf 近似
       float rsqrt2 = __rsqrtf(2.0f);        // SFU 倒数平方根
       float x_norm = x * rsqrt2;
       float exp_val = __expf(-0.5f * x * x); // SFU exp 近似
       // erf 多项式近似
       float erf_approx = 1.0f - exp_val;
       return 0.5f * x * (1.0f + erf_approx);
   }

SFU 与 CUDA Core 的并行
===============================

SFU 和 CUDA Core 是 SM 中**独立**的执行单元，可并行发射：

.. code-block:: text

   每个周期，warp 调度器可以发射两条指令（Volta+）:
   指令 1: CUDA Core 算术运算（FMA、ADD、MUL）
   指令 2: SFU 超越函数 或 LD/ST 内存指令

   这意味着在优化良好的 kernel 中，SFU 计算可以和 CUDA Core 运算重叠，
   不增加额外的周期开销。

这个特性对性能优化非常重要：
- 如果 kernel 中有足够的 CUDA Core 算术操作来"遮掩"SFU 的长延迟（~16 周期），SFU 指令的延迟可以完全被隐藏
- 最佳实践：在 SFU 指令前后穿插不依赖其结果的 CUDA Core 算术指令

SFU 的精度特性详解
=========================

以下展示 SFU 结果与 IEEE 754 标准结果在不同输入区间的误差分布：

.. code-block:: text

   函数       输入范围        最大 ULP 误差    误差分布
   ─────────────────────────────────────────────────
   __sinf     [-π, π]        ~2 ULP          均匀分布
   __cosf     [-π, π]        ~2 ULP          均匀分布
   __logf     [1, 10]        ~1 ULP          偏向负向
   __expf     [-10, 10]      ~1 ULP          偏向正向
   __rsqrtf   [1, 100]       ~2 ULP          均匀分布

   对于 __sinf/__cosf，当输入接近 0 时误差最小（多项式在 0 附近精度最高）
   对于 __logf，输入接近 1 时误差最小

   Nsight Compute 提供了 SFU 指令的精度的分析：
   ncu --metrics sm__inst_executed.avg.pct_of_peak_sustained_elapsed \
       --kernel-name my_kernel ./my_app

SFU 常用内建函数速查
==============================

.. list-table::
   :header-rows: 1

   * - CUDA 内建函数
     - 对应的 SFU 指令
     - 延迟（周期）
     - 误差（ULP）
   * - ``__sinf(x)``
     - ``sin.approx.f32``
     - ~16
     - ~2
   * - ``__cosf(x)``
     - ``cos.approx.f32``
     - ~16
     - ~2
   * - ``__sincosf(x, s, c)``
     - sin + cos（一次计算）
     - ~20
     - ~2
   * - ``__logf(x)``
     - ``lg2.approx.f32`` + 乘法换算
     - ~16
     - ~1
   * - ``__log2f(x)``
     - ``lg2.approx.f32``
     - ~16
     - ~1
   * - ``__expf(x)``
     - ``ex2.approx.f32`` + 乘法换算
     - ~16
     - ~1
   * - ``__exp10f(x)``
     - ``ex2.approx.f32`` + 乘法换算
     - ~16
     - ~2
   * - ``__powf(x, y)``
     - log + exp 组合
     - ~32
     - ~2
   * - ``__rsqrtf(x)``
     - ``rsqrt.approx.f32``
     - ~8
     - ~2
   * - ``__frcp_rz(x)``
     - rcp.approx.f32（向零舍入）
     - ~4
     - ~1

参考与拓展阅读
====================

- 深入理解 :doc:`cuda_core` — CUDA Core 流水线和指令吞吐
- 深入理解 :doc:`tensor_core_architecture` — Tensor Core 矩阵运算单元
- Parallel Thread Execution ISA — PTX 中 sin/cos/log/exp 指令的规格说明
- Nsight Compute 性能分析工具 — 查看 SFU 指令的吞吐和延迟指标
- 超越函数快速算法：Polynomial approximations and implementation — Minimax 多项式的设计原理
