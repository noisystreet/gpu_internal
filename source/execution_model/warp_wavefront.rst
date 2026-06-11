========================
Warp 与 Wavefront
========================

.. epigraph::

   One of my most productive days was throwing away 1,000 lines of code.

   — Ken Thompson

Warp 是 NVIDIA GPU 最基本的硬件调度单位，AMD GPU 中的对应概念为 Wavefront。

Warp 的概念
===============

Warp 是一组 **32 个连续的线程**，在 SM 上以 SIMD（Single Instruction, Multiple Data）方式执行。一个线程块中的线程按连续的 threadIdx 分组为 warp。

.. code-block:: text

   线程块 (Block Size = 256)
   +--------------------------------------------------+
   | Warp 0:  线程  0 -  31                           |
   | Warp 1:  线程 32 -  63                           |
   | Warp 2:  线程 64 -  95                           |
   | ...                                               |
   | Warp 7:  线程 224 - 255                          |
   +--------------------------------------------------+

Warp 调度策略
=================

每个 SM 包含多个 warp 调度器（Ampere 有 4 个，每个 subcore 对应一个）。调度器的核心任务是：在每个周期从就绪 warp 中选择一条指令发射到执行单元。

**主流调度策略**:

.. list-table::
   :header-rows: 1

   * - 策略
     - 原理
     - 特点
     - 适用场景
   * - Greedy Oldest Ready
     - 优先发射等待最久的 warp 指令
     - 单 warp 延迟低，公平性一般
     - 延迟敏感型 kernel
   * - Round Robin (循环)
     - 轮转选择就绪 warp
     - 公平性好，整体吞吐稳定
     - 吞吐量敏感型 kernel
   * - Age-based Two-Level
     - 将 warp 分为活跃/待命两级，优先发射活跃 warp
     - 兼顾延迟和吞吐
     - 混合负载

.. code-block:: text

   Greedy Oldest Ready 调度示例（4 warp, 每周期 1 发射）:

   周期:   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   ------------------------------------------------
   Warp 0: | I |   |   |   | I |   |   |   |
   Warp 1: |   | I |   |   |   | I |   |   |
   Warp 2: |   |   |   |   |   |   |   |   |  ← 停顿（等待内存）
   Warp 3: |   |   | I | I |   |   | I | I |

   就绪状态: warp 0 和 warp 1 连续发射，warp 3 在其间穿插

**指令发射宽度**:

每个 warp 调度器每周期可发射指令的数量称为发射宽度：

- **Maxwell/Turing**: 每调度器每周期 1 条指令（1-issue），但可同时发射到不同的执行管线
- **Volta**: 每调度器每周期 2 条指令（2-issue），可同时发射一条计算指令和一条内存指令
- **Ampere/Hopper**: 每调度器仍为 1-issue，但通过更多 subcore 并行提升总吞吐

.. code-block:: text

   Volta 2-issue 示例:
   周期 0: warp 3 发射 [计算指令] + [内存加载]  ← 并行
   周期 1: warp 1 发射 [计算指令] + [符号指令]

**零成本线程切换（Zero-overhead Thread Switching）**
    GPU 的上下文切换成本接近于零。当当前 warp 因等待内存访问结果而停顿时，调度器立即切换到另一个就绪的 warp。多个就绪 warp 帮助隐藏长延迟操作（如全局内存访问，200-800 周期）。

**指令发射延迟隐藏**:

.. code-block:: text

   时钟周期: | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | ...
   --------------------------------------------------
   Warp 0:    | I0 |    |    |    | I1 |    |    | ...
   Warp 1:    |    | I0 |    |    |    | I1 |    | ...
   Warp 2:    |    |    | I0 |    |    |    | I1 | ...
   Warp 3:    |    |    |    | I0 |    |    |    | I1

   需要足够的 active warp 来隐藏延迟。例如，对于 4 周期延迟，至少需要 4 个 warp 达到满吞吐。

Warp 发散（Warp Divergence）
===============================

上一节我们提到了 warp 调度器的"就绪"状态——当 warp 内所有线程都沿着同一路径执行时，调度器可以顺利发射下一条指令。但现实中的程序充满条件分支，当 warp 内的线程面临不同的执行路径时，就发生了 warp 发散。

SIMT 栈管理
-----------------

GPU 使用硬件 **SIMT 栈** 管理发散分支。每个 warp 维护一个栈，记录活跃线程掩码（active mask）和收敛点（reconvergence point）：

.. mermaid::

   flowchart TD
       A["if (condition)"] --> B["then-path<br/>线程 0-7 活跃<br/>线程 8-31 屏蔽"]
       A --> C["else-path<br/>线程 8-31 活跃<br/>线程 0-7 屏蔽"]
       B --> D["push mask<br/>栈深度 +1"]
       C --> D
       D --> E["then 路径<br/>执行完"]
       D --> F["else 路径<br/>执行完"]
       E --> G["pop mask<br/>栈深度 -1<br/>恢复完整 warp"]
       F --> G
       G --> H["收敛点<br/>所有线程重新同步"]

       style A fill:#f3e5f5,color:#7b1fa2
       style G fill:#e8f5e9,color:#1b5e20
       style H fill:#fff8e1,color:#e65100

**多级发散** 嵌套分支时，栈深度增加，每个嵌套级别产生额外的 push：

.. code-block:: cuda

   if (cond1) {             // 1 级发散
       if (cond2) {         // 2 级发散
           // ...
       }
   }

**关键性能影响**:

- 每级发散导致 push/pop 开销，约 4-8 周期
- 深度嵌套分支（6 层以上）可能溢出 SIMT 栈，跌落到慢路径
- Warp 内分支越少、越平衡，性能越好

谓词执行（Predication）
--------------------------

编译器可将短小的条件分支转换为**谓词指令**，避免 SIMT 栈发散：

.. code-block:: cuda

   // 发散版本
   if (tid < 16) {
       a[tid] = b[tid] * 2.0f;
   } else {
       a[tid] = b[tid] + 1.0f;
   }

   // 编译器优化为谓词版本:
   // p0 = (tid < 16)
   // @!p0  a[tid] = b[tid] + 1.0f   (只有 p0=0 的线程执行)
   // @p0   a[tid] = b[tid] * 2.0f   (只有 p0=1 的线程执行)

**谓词化** 将分支延迟转换为 ALU 资源使用：两条指令都被发射，但只有条件匹配的 lane 写入结果。它避免了分支的 SIMT 栈开销，适用于条件简单的短路径。但对于条件复杂或分支体很大的情况，实际发散仍优于谓词化。

**何时使用谓词**:

.. list-table::
   :header-rows: 1

   * - 条件
     - 推荐使用
     - 理由
   * - 简单条件、短路径 (< 8 条指令)
     - 谓词化
     - 避免 SIMT 栈开销
   * - 复杂条件、长路径
     - 分支发散
     - 减少无效指令发射
   * - 线程束内所有线程路径相同
     - 任意
     - 不发散，无条件开销
   * - 路径间指令差异很大
     - 分支发散
     - 谓词化会浪费发射带宽

.. code-block:: cuda
   :linenos:

   __global__ void divergent_kernel(float* data, int N) {
       int tid = blockIdx.x * blockDim.x + threadIdx.x;
       if (tid < N / 2) {
           data[tid] = data[tid] * 2.0f;     // 路径 A
       } else {
           data[tid] = data[tid] + 1.0f;     // 路径 B
       }
   }

当 ``N/2`` 不是 32 的倍数时，至少有一个 warp 同时包含走路径 A 和路径 B 的线程，导致发散。

**避免发散的策略**:

1. 确保线程块大小为 32 的倍数
2. 使用 `#pragma unroll` 减少循环发散
3. 将条件判断移出 warp，如使用 ``__ballot_sync`` 和 ``__activemask()``

.. code-block:: cuda

   // 使用谓词（predication）避免发散
   // 条件判断被转换为谓词指令，所有线程都执行
   float val = data[tid];
   float result = (tid < N / 2) ? (val * 2.0f) : (val + 1.0f);
   data[tid] = result;

AMD Wavefront
=================

AMD GPU 的 Wavefront 大小为 **64 个线程**，比 warp 大两倍。这意味着：

- 更大的发散惩罚：一个 wavefront 中 64 个线程必须统一执行路径
- 相同数量的线程占用更多寄存器
- 编译器需要更谨慎地处理分支

.. list-table::
   :header-rows: 1

   * - 特性
     - NVIDIA Warp
     - AMD Wavefront
   * - 大小
     - 32 线程
     - 64 线程
   * - 发散惩罚
     - 较小（32 线程串行化）
     - 较大（64 线程串行化）
   * - 占用率灵活性
     - 更高
     - 较低

Warp 级原语
===============

理解了 warp 的集体执行特性后，CUDA 提供的一系列 warp 级指令就变得非常自然——这些指令允许 warp 内线程直接交换数据，而不需要通过共享内存和同步操作。这就像同一小组的成员可以直接对话，而不必每次都要写便签贴在公告板上。

.. code-block:: cuda
   :linenos:

   // Warp 内规约：__shfl_down_sync
   __global__ void warp_reduce(float* data) {
       int tid = threadIdx.x;
       float val = data[tid];

       // Warp 0 执行规约
       unsigned mask = __activemask();
       for (int offset = 16; offset > 0; offset >>= 1) {
           val += __shfl_down_sync(mask, val, offset);
       }

       if (tid % 32 == 0) {
           data[tid / 32] = val;
       }
   }

常用 warp 级函数可分为几个功能类别：

**数据交换（Shuffle）**：允许 warp 内线程直接读写其他线程的寄存器值：

- ``__shfl_sync(mask, var, srcLane)`` — 从指定 lane 获取值。适用于取邻接数据，如每个线程从其左邻获取结果
- ``__shfl_up_sync(mask, var, delta)`` — 从 ``lane - delta`` 获取值。常用于前缀和（prefix sum）等自左向右的传播操作
- ``__shfl_down_sync(mask, var, delta)`` — 从 ``lane + delta`` 获取值。常用于规约（reduction）等自右向左的汇聚

**线程投票（Vote）**：收集 warp 内线程的布尔状态：

- ``__ballot_sync(mask, predicate)`` — 收集 32 个线程的谓词值，打包为一个 32 位整型。每一位代表一个线程的 true/false，是 warp 发散分析和自适应算法的核心工具
- ``__all_sync(mask, predicate)`` — 若 **所有** 活跃线程的 predicate 均为真，返回真。等价于 ``__ballot_sync(mask, predicate) == mask``
- ``__any_sync(mask, predicate)`` — 若 **任意** 活跃线程的 predicate 为真，返回真。可用于提前退出或快速收敛
- ``__popc(mask)`` — 统计 32 位掩码中 1 的个数，即活跃线程数。配合 ``__ballot_sync`` 可实现在不知道活跃线程数量的情况下动态调整算法

**选择建议**：

- 需要 warp 内数据共享时，优先使用 ``__shfl_*_sync`` 而非共享内存，延迟更低（约 5 周期 vs 30 周期）
- ``__ballot_sync`` 的 ``mask`` 参数应传 ``__activemask()``，确保仅作用于当前活跃的 lane
- Volta+ 架构要求显式传入 ``mask`` 参数（相比前代的隐式 mask 更明确但更复杂）

参考与拓展阅读
====================

- 深入理解 :doc:`../hardware/occupancy` — 占用率与 warp 调度的关系
- 深入理解 :doc:`../execution_model/kernel` — 线程块和 Grid 的配置策略
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX 指令集手册中 shfl.sync、vote.sync 等 warp 级指令的完整规范
- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中关于 warp 级原语和 Cooperative Groups 的章节
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 分支发散和谓词化的优化策略
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 第 6 章深入讲解 Warp 调度和 SIMT 机制
