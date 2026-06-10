========================
计算单元：SM 与 CU
========================

.. epigraph::

   The principle of "divide and conquer" — when you have a large and complex system, break it into smaller independent pieces that can be worked on separately.

   — John von Neumann

流多处理器（Streaming Multiprocessor, SM）是 NVIDIA GPU 的核心计算单元，AMD GPU 中对应的概念称为计算单元（Compute Unit, CU）。理解 SM/CU 的内部结构是掌握 GPU 性能调优的基础。

SM 内部结构
===============

以 NVIDIA Ampere 架构的 SM 为例：

.. code-block:: text

   SM (Streaming Multiprocessor)
   +--------------------------------------------------+
   |  分区 0              分区 1              分区 2   分区 3  |
   |  +----------------+  +----------------+ ...       |
   |  | Warp Scheduler |  | Warp Scheduler |           |
   |  | Dispatch Unit  |  | Dispatch Unit  |           |
   |  |                |  |                |           |
   |  | INT32 核心 x16 |  | INT32 核心 x16 |           |
   |  | FP32 核心 x16  |  | FP32 核心 x16  |           |
   |  | Tensor Core x4 |  | Tensor Core x4 |           |
   |  |                |  |                |           |
   |  | 寄存器文件     |  | 寄存器文件     |           |
   |  +----------------+  +----------------+           |
   |                                                    |
   |  L1 / 共享内存 (128 KB)                            |
   |  一级缓存 / 纹理缓存                                |
   |  加载/存储单元 (LD/ST)                              |
   +--------------------------------------------------+

SM Subcore（子核心）
=====================

SM 内部被划分为多个 ``subcore``（也称分区、处理块），每个 ``subcore`` 拥有独立的指令调度和执行资源。这是现代 GPU 提升指令级并行（ILP）和线程级并行（TLP）的关键设计。

**每个 subcore 包含的独立资源**:

- **Warp Scheduler**: 负责维护和发射 warp 指令
- **Dispatch Unit**: 将指令分发到执行单元
- **CUDA Core 子集**: 一组 FP32 和 INT32 核心
- **Tensor Core 子集**: 部分 Tensor Core
- **寄存器文件**: subcore 私有的寄存器存储

**Subcore 的独立调度能力**:

.. code-block:: text

   时钟周期:  |  0  |  1  |  2  |  3  |  4  |  5  |
   -------------------------------------------------------
   Subcore 0: | I0  | I0  | I0  |     | I1  |     |
              | W0  | W1  | W2  |     | W0  |     |
   Subcore 1: |     | I0  | I0  | I0  |     | I1  |
              |     | W3  | W4  | W5  |     | W3  |

   每个 subcore 的 warp scheduler 独立选择就绪 warp 发射指令，
   四个 subcore 每个周期最多发射 4 条独立指令。

**Subcore 设计随架构的演进**:

.. list-table::
   :header-rows: 1

   * - 架构
     - Subcore 数
     - 每 Subcore CUDA Core
     - 每 Subcore Tensor Core
     - 调度器特点
   * - Maxwell (GM200)
     - 4
     - 32 (FP32)
     - 无
     - 每 subcore 1 调度器，2 发射
   * - Pascal (GP100)
     - 2
     - 32 (FP32)
     - 无
     - 每 subcore 1 调度器，2 发射
   * - Volta (GV100)
     - 4
     - 16 FP32 + 16 INT32
     - 2
     - 独立 FP32/INT32 管线
   * - Turing (TU102)
     - 4
     - 16 FP32 + 16 INT32
     - 2
     - 与 Volta 类似
   * - Ampere (GA100)
     - 4
     - 16 FP32 + 16 INT32
     - 4
     - 第三代数 Tensor Core
   * - Hopper (GH100)
     - 4
     - 16 FP32
     - 8
     - DPX 指令，Transformer Engine

**Subcore 设计的核心优势**:

1. **减少资源争用** — warp 调度、寄存器访问等操作局限在 subcore 内，避免全局竞争
2. **提升指令吞吐** — 多个 subcore 可并行发射指令，每个周期最多 4 条（Ampere）
3. **更好的局部性** — subcore 私有的寄存器文件减少了布线延迟
4. **细粒度电源门控** — 不活跃的 subcore 可独立关闭降低功耗

**编程视角的影响**:

虽然程序员不直接控制 subcore 的分配，但 subcore 结构影响性能优化策略：

- 同一 subcore 内的 warp 共享寄存器文件，寄存器分配不当会降低 subcore 的 warp 驻留量
- 不同 subcore 的 warp 之间几乎没有资源竞争，因此线程块划分到不同 subcore 可获得更好的隔离性
- subcore 结构解释了为什么每个 SM 的 warp 调度器数量（等于 subcore 数）是性能调优的重要参数

关键组件
============

**Warp Scheduler（束调度器）**
    每个 SM 有多个 warp 调度器（如 Ampere 有 4 个），每个调度器负责维护和发射 warp。调度器在每个周期选择一条就绪的 warp 指令发射到执行单元。

**CUDA 核心（CUDA Core）**
    每个 SM 包含数百个 CUDA 核心，是执行算术和逻辑运算的主要执行单元。CUDA 核心围绕 **FMA（Fused Multiply-Add）** 流水线设计，执行 ``D = A * B + C`` 运算。

CUDA Core 微架构
""""""""""""""""""

CUDA core 本质上是经过优化的流水线化 ALU，其核心执行路径如下：

.. code-block:: text

   指令 → Warp Scheduler → Dispatch → Operand Collect → FMA Pipeline → Writeback
                                       │
                                       ↓
                              ┌──────────────────┐
                              │  寄存器文件读取     │
                              │  (每个线程 2 个 op)  │
                              └──────────────────┘

**流水线深度**: FP32 FMA 通常为 4-6 个周期（取决于架构代际），意味着一个 warp 的指令发射后需要 4-6 周期才能得到结果。

**FP32 与 INT32 并行**:

从 Volta (SM 7.0) 开始，每个 SM subcore 内的 FP32 核心和 INT32 核心可以**独立并行**执行：

.. code-block:: text

   Volta/Ampere 每个 subcore:
   +-------------------------------------------+
   |  FP32 Core x16: 负责单精度浮点运算          |
   |  INT32 Core x16: 负责整数运算               |
   |  内存操作 (LD/ST): 独立单元                  |
   |                                            |
   |  调度器在一个周期可发射:                     |
   |  [1条 FP32 指令] + [1条 INT32 或 内存指令]  |
   +-------------------------------------------+

.. list-table::
   :header-rows: 1

   * - 架构
     - FP32/INT32 关系
     - 每周期每 subcore 最大指令数
     - FP32 FMA 延迟
   * - Maxwell
     - FP32 可执行 INT32（互斥）
     - 1 条（FP32 或 INT32）
     - 6 周期
   * - Pascal
     - FP32 可执行 INT32（互斥）
     - 1 条
     - 6 周期
   * - Volta
     - FP32 + INT32 独立
     - 2 条（1 FP32 + 1 INT32/LD/ST）
     - 4 周期
   * - Turing
     - FP32 + INT32 独立
     - 2 条
     - 4 周期
   * - Ampere
     - FP32 + INT32 独立
     - 2 条
     - 4 周期
   * - Hopper
     - FP32 + INT32 独立
     - 2 条
     - 4 周期

**指令吞吐（Instruction Throughput）**

以下为每个 SM 每周期的指令吞吐（以 Ampere 为例，4 subcore）：

.. list-table::
   :header-rows: 1

   * - 指令类型
     - 每 SM / 周期
     - 说明
   * - FP32 FMA
     - 64 条
     - 4 subcore x 16 FP32 Core
   * - FP32 ADD/MUL
     - 64 条
     - 同上（不含乘加）
   * - INT32 ADD/SUB
     - 64 条
     - 4 subcore x 16 INT32 Core
   * - INT32 MUL
     - 32 条
     - 只有部分 INT32 core 支持
   * - FP64
     - 32 条
     - 每 2 个 FP32 Core 共享 1 个 FP64
   * - SFU (sin/cos/log/exp)
     - 16 条
     - 每 subcore 1 个 SFU

**Warp 在 CUDA Core 上的执行**

当 warp 的 FP32 指令被调度时，32 个线程的运算被分配到 2 个 subcore 上（每个 subcore 16 个 FP32 Core），2 个周期完成一个 warp：

.. code-block:: text

   周期 0: Subcore 0 执行线程 0-15
           Subcore 1 执行线程 16-31
           但实际执行中，一个 warp 发射到一个 subcore，用 2 周期完成：
   周期 0: warp N → Subcore 0 → 线程 0-15 的 FP32 指令
   周期 1: warp N → Subcore 0 → 线程 16-31 的 FP32 指令

   或者两个 subcore 各处理半个 warp：
   周期 0: warp N → Subcore 0 (线程 0-15) + Subcore 1 (线程 16-31)
   该 warp 在 1 周期内完成

   Ampere 架构中，每个 warp 调度器每周期发射一条指令到其关联 subcore，
   4 个 subcore 并行，每周期最多 4 条 warp 指令被发射执行。

理解了 CUDA Core 如何执行标量运算，我们自然会问：对于深度学习中的大规模矩阵运算，是否可以用更高效的方式实现？答案就是 Tensor Core——专为矩阵乘法累加设计的专用硬件单元。

**Tensor Core（张量核心）**
    专为矩阵乘法累加（MMA）操作设计的专用硬件单元。与 CUDA Core 执行标量运算不同，Tensor Core 在 warp 级别执行**矩阵分块乘加**操作，单条指令完成一个子矩阵的 ``D = A * B + C``。

Tensor Core 工作原理
"""""""""""""""""""""""

Tensor Core 的操作单位是 **warp（32 线程）**。整个 warp 协作完成一个矩阵分块的乘法累加，而不是每个线程独立运算：

.. code-block:: text

   Warp (32 线程) 协作执行一次 MMA 操作：

   A (16x16)       B (16x16)        C (16x16)         D (16x16)
   ┌──────┐        ┌──────┐         ┌──────┐          ┌──────┐
   │      │        │      │         │      │          │      │
   │      │   ×    │      │    +    │      │    =     │      │
   │      │        │      │         │      │          │      │
   └──────┘        └──────┘         └──────┘          └──────┘

   - A 和 C 矩阵存储在每个线程的寄存器中
   - B 矩阵存储在共享内存中（或寄存器，取决于 tile 大小）
   - 一次 MMA 指令：所有 32 线程协作，在 1 个周期内完成
   - 等价于 4096 次 FMA 运算（16x16x16）

**指令级操作**:

Tensor Core 操作在 PTX 指令层面表示为 ``mma.sync``：

.. code-block:: text

   // Ampere 架构的 mma 指令示例
   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
   { d[0..3] }, { a[0..3] }, { b[0..3] }, { c[0..3] }

   参数含义:
   - m16n8k16: A 矩阵 16x16, B 矩阵 16x8, 内积维度 k=16
   - row.col: A 行主序, B 列主序
   - f16.f16.f16.f16: A/B/C/D 的精度

**编程接口**:

1. **WMMA API（warp matrix multiply-accumulate）** — 用户友好的高层抽象

.. code-block:: cuda

   #include <mma.h>
   using namespace nvcuda;

   __global__ void tensor_core_example(half* A, half* B, float* C, float* D) {
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

       wmma::load_matrix_sync(a_frag, A, 16);
       wmma::load_matrix_sync(b_frag, B, 16);
       wmma::load_matrix_sync(c_frag, C, 16, wmma::mem_row_major);

       wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

       wmma::store_matrix_sync(D, d_frag, 16, wmma::mem_row_major);
   }

2. **CUTLASS 库** — NVIDIA 的开源 GEMM 模板库，封装了 mma 指令的详细调度

3. **cuBLAS/cuDNN** — 通过库 API 自动调用 Tensor Core

WMMA API 与 mma API 的差异
""""""""""""""""""""""""""""""""

NVIDIA 提供了两套编程接口操作 Tensor Core，它们在灵活性和性能上存在显著差异（Sun 等, 2022）：

.. list-table::
   :header-rows: 1

   * - 特性
     - WMMA API (legacy)
     - mma API (PTX)
   * - 抽象层次
     - 高层封装（fragment 抽象）
     - 底层 PTX 指令直接操控
   * - 支持的 tile 形状
     - 仅 16x16x16 (FP16)
     - m16n8k16, m16n8k8, m8n8k4 等多种
   * - 稀疏矩阵支持
     - 不支持
     - Ampere+ 支持 2:4 稀疏 (mma.sp)
   * - 数据加载
     - load_matrix_sync（自动布局）
     - ldmatrix（手动布局，精确控制）
   * - 性能（同条件下）
     - 基准
     - 略优（减少 fragment 布局开销）
   * - 可编程性
     - 更简单
     - 更复杂

.. code-block:: cuda

   // mma API (PTX) 直接调用 Tensor Core 指令
   __global__ void tensor_core_mma(const half* A, const half* B, float* D) {
       // 使用 ldmatrix 指令加载数据到寄存器
       uint32_t a_reg[4], b_reg[4];  // 寄存器碎片
       asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
           : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
           : "r"(shared_addr_A));

       uint32_t c_reg[4] = {0};  // 累加器初始化为 0
       uint32_t d_reg[4];

       // mma 指令：16x8x16 FP16 → FP32
       asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
           "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15};\n"
           : "=r"(d_reg[0]), "=r"(d_reg[1]), "=r"(d_reg[2]), "=r"(d_reg[3])
           : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
             "r"(b_reg[0]), "r"(b_reg[1]), "r"(b_reg[2]), "r"(b_reg[3]),
             "r"(c_reg[0]), "r"(c_reg[1]), "r"(c_reg[2]), "r"(c_reg[3]));
   }

Tensor Core 的数值行为
""""""""""""""""""""""""""""""""

Tensor Core 的浮点运算**不遵循 IEEE 754 标准**，其内部实现采用了多种与标准不同的优化策略（Fasi 等, 2021; Sun 等, 2022）。理解这些数值特性对于科学计算和高精度场景至关重要。

**关键数值特性**：

.. list-table::
   :header-rows: 1

   * - 特性
     - 描述
     - 与 IEEE 754 的差异
   * - 乘积精度
     - A×B 的乘法结果以至少单精度（FP32）计算
     - 符合 NVIDIA 文档
   * - 累加精度
     - 中间累加以至少单精度（FP32）执行
     - 符合 NVIDIA 文档
   * - 舍入模式
     - RTZ（Round-to-Zero，截断）
     - IEEE 754 默认为 RNE
   * - 保护位（guard bits）
     - 无保护位，直接截断
     - IEEE 754 RTZ 需要保护位
   * - 中间和归一化
     - 不归一化（intermediate sums not normalized）
     - IEEE 754 要求归一化
   * - 加法器对齐
     - 尾数仅按最大幅度一次对齐
     - IEEE 754 逐对对齐
   * - 非规格化数
     - 清零（flush-to-zero）
     - IEEE 754 支持非规格化数
   * - NaN/Inf 处理
     - 符合 IEEE 754 规范
     - 一致

**加法器微架构分析**：

Fasi 等人（2021）通过精心设计的数值实验，揭示了 Tensor Core 内部的加法器结构：

.. code-block:: text

   d11 = a11·b11 + a12·b21 + a13·b31 + a14·b41 + c11

   在 Tensor Core 内部的加法树中：

   5 个乘积项同时进入多操作数加法器
          │
          ↓
   尾数对齐：基于 5 项中的最大指数一次性对齐
   （而非 IEEE 754 的逐对对齐）
          │
          ↓
   对齐后的尾数直接相加（不归一化）
   使用 3 个进位位处理超范围结果
          │
          ↓
   结果截断（Round-to-Zero）
          │
          ↓
   输出最终累加值

**这一设计的后果**：

1. **非单调性**：由于中间和不归一化，多操作数加法可能出现非单调性——即增加一项的值可能反而使结果变小。Fasi 等人指出，这在科学计算中需要特别注意。

2. **舍入误差模型**：不同于 IEEE 754 的标准舍入误差分析，Tensor Core 的截断舍入导致误差分布非对称（偏向零），且不满足标准浮点分析中的常见假设。

3. **运算顺序无关**：因为所有乘积项的对齐基于最大指数一次完成，各项的排列顺序不影响最终结果——这与 IEEE 754 的非结合性形成鲜明对比。

**精度恢复技术**：

研究人员提出了多种方法来恢复 Tensor Core 的精度损失（Ootomo 和 Yokota, 2022; Markidis 等, 2018）：

.. code-block:: text

   Markidis 方法（2018）：
   1. 使用 Tensor Core 计算低精度结果
   2. 计算残差矩阵 R = D - C - A·B（使用 CUDA Core 以 FP32 计算）
   3. 用 Tensor Core 在残差上修正结果

   Ootomo-Yokota 改进（2022）：
   1. 识别 Tensor Core 的舍入误差来源是 RTZ 而非 IEEE-754
   2. 显式补偿 RTZ 误差项
   3. 在保持 Tensor Core 高吞吐的同时恢复单精度精度
   4. 效果：接近 FP32 精度，超过 FP32 CUDA Core 的吞吐

**指令吞吐与延迟数据**：

Sun 等人（2022）通过微基准测试测得的 Tensor Core 指令性能：

.. list-table::
   :header-rows: 1

   * - 指令（Ampere A100）
     - 延迟（周期）
     - 吞吐（每 SM/周期）
     - 说明
   * - mma.m16n8k16 (FP16)
     - ~8 周期
     - 4 条
     - 主流矩阵尺寸
   * - mma.m16n8k8 (BF16)
     - ~8 周期
     - 4 条
     - BF16 格式
   * - mma.m16n8k4 (TF32)
     - ~8 周期
     - 4 条
     - 19 位精度
   * - mma.m16n8k16 (INT8)
     - ~8 周期
     - 4 条
     - INT8 量化
   * - mma.sp (2:4 稀疏)
     - ~8 周期
     - 4 条
     - 2x 加速比
   * - ldmatrix (加载)
     - ~12-16 周期
     - 2 条
     - 数据加载到寄存器

.. note::

   上述数据基于 Ampere A100 通过 CUDA PTX 指令级别的微基准测试（Sun 等, 2022; Abdelkhalik 等, 2022）。
   实际应用中的端到端吞吐还受数据加载、共享内存 bank 冲突和寄存器压力等因素影响。

支持的精度与代际演进
"""""""""""""""""""""""""""""""

.. list-table::
   :header-rows: 1

   * - 代际
     - 架构
     - 支持的精度
     - 每 SM Tensor Core 数
     - 峰值 TFLOPS (FP16, A100)
   * - 第 1 代
     - Volta (V100)
     - FP16
     - 8
     - 125
   * - 第 2 代
     - Turing (T4)
     - FP16, INT8, INT4
     - 8
     - 65 (INT8: 130)
   * - 第 3 代
     - Ampere (A100)
     - FP16, BF16, TF32, INT8, INT4
     - 4
     - 312 (TF32: 156)
   * - 第 4 代
     - Hopper (H100)
     - FP16, BF16, TF32, FP8 (E5M2/E4M3), INT8
     - 4
     - 989 (FP8: 1979)
   * - 第 5 代
     - Blackwell (B200)
     - FP16, BF16, TF32, FP8, FP6, FP4, INT8
     - N/A
     - 2250 (FP4: 9000)

**精度格式详解**:

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

**Tensor Core vs CUDA Core 吞吐对比**:

以 H100 SXM 为例，不同精度下的峰值吞吐：

.. code-block:: text

   精度        Tensor Core TFLOPS   CUDA Core TFLOPS   加速比
   ──────     ──────────────────   ────────────────   ──────
   FP64               67                 67             1x
   FP32               67                 67             1x
   TF32              989                 —              —
   FP16              989                —               —
   BF16              989                —               —
   FP8 (E4M3)       1979                —               —
   INT8             1979                —               —
   FP4                —                 —               —

   "—" 表示该格式在该单元上不支持。GPU 计算主要通过 Tensor Core 完成，
   CUDA Core 现在更多执行地址计算、控制流和少量标量运算。

Tensor Core 使用场景
"""""""""""""""""""""""""

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

**Hopper Tensor Core 增强**:

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

**特殊函数单元（SFU）**
    处理 ``sin``、``cos``、``log``、``exp``、``rsqrt`` 等超越函数。

**加载/存储单元（LD/ST）**
    处理全局内存和共享内存的加载/存储指令，负责地址生成和内存合并。

了解完 NVIDIA 的 CUDA Core 和 Tensor Core 之后，我们再来看 AMD 的对应设计。尽管 AMD 的 Compute Unit 在整体功能上对标 SM，但其内部架构和线程模型有显著不同。

AMD Compute Unit (CU) 结构
===============================

AMD CDNA 3 架构的 CU 通过两个 SIMD 单元（也称为 Wavefront 槽位）执行指令：

.. code-block:: text

   Compute Unit (CU)
   +--------------------------------------------------+
   |  SIMD 槽 0          SIMD 槽 1                     |
   |  +----------------+ +----------------+            |
   |  | Wavefront      | | Wavefront      |            |
   |  | Scheduler      | | Scheduler      |            |
   |  |                | |                |            |
   |  | Vector ALU x16 | | Vector ALU x16 |            |
   |  | Matrix Core    | | Matrix Core    |            |
   |  +----------------+ +----------------+            |
   |                                                    |
   |  共享内存 (128 KB)                                 |
   |  L1 数据缓存                                        |
   |  标量 ALU                                           |
   +--------------------------------------------------+

NVIDIA SM vs AMD CU 对比
==============================

.. list-table::
   :header-rows: 1

   * - 特征
     - NVIDIA SM (Ampere)
     - AMD CU (CDNA 3)
   * - 线程束大小
     - 32 线程 (warp)
     - 64 线程 (wavefront)
   * - 调度器 / SM
     - 4 个 warp 调度器
     - 2 个 wavefront 调度器
   * - FP32 核心 / 单元
     - 64
     - 32
   * - Tensor / Matrix Core
     - 16
     - 4
   * - 共享内存
     - 128 KB
     - 128 KB
   * - 寄存器文件
     - 65536
     - 约 48000

占用率（Occupancy）
=====================

占用率指每个 SM 中活跃 warp 数量与最大 warp 数量的比值。高占用率有助于隐藏内存延迟，但并非唯一性能指标。

**影响占用率的因素**:

1. **每线程寄存器数量** — 寄存器越多，可驻留线程越少
2. **共享内存使用量** — 每个线程块使用的共享内存越多，SM 中并行线程块越少
3. **线程块大小** — 过小或过大的块大小都会限制占用率

.. code-block:: cuda
   :linenos:

   // 使用占用率 API 计算最大理论占用率
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
       &numBlocks,        // 输出：每 SM 活跃块数
       my_kernel,         // kernel 函数
       blockSize,         // 线程块大小
       sharedMemPerBlock  // 每块共享内存 (bytes)
   );

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — SM 架构和占用率计算器的官方说明
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX 指令集手册中 mma.sync 指令的完整规格
- CUTLASS: CUDA Templates for Linear Algebra (https://github.com/NVIDIA/cutlass) — CUTLASS 库实现 Tensor Core GEMM 的模板代码
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 占用率分析和寄存器优化
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — Ampere SM 结构和指令吞吐的微基准测试
- Numerical Behavior of NVIDIA Tensor Cores (https://peerj.com/articles/cs-330/) — Tensor Core 浮点运算的舍入模式、加法器结构和非单调性分析
- Dissecting Tensor Cores via Microbenchmarks (https://arxiv.org/abs/2206.02874) — WMMA/mma API 延迟、吞吐和低精度数值行为
