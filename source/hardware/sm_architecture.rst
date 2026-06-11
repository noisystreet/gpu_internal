========================
SM 架构与 Subcore
========================

.. epigraph::

   The principle of "divide and conquer" — when you have a large and complex system, break it into smaller independent pieces that can be worked on separately.

   — John von Neumann

流多处理器（Streaming Multiprocessor, SM）是 NVIDIA GPU 的核心计算单元，AMD GPU 中对应的概念称为计算单元（Compute Unit, CU）。理解 SM/CU 的内部结构是掌握 GPU 性能调优的基础。

SM 内部结构
===============

以 NVIDIA Ampere 架构的 SM 为例：

.. mermaid::

   flowchart TB
       subgraph SM["SM (Streaming Multiprocessor)"]
           direction TB
           subgraph P0["分区 0"]
               WS0["Warp Scheduler<br/>Dispatch Unit"]
               CU0["INT32 核心 x16<br/>FP32 核心 x16<br/>Tensor Core x4"]
               RF0["寄存器文件"]
           end
           subgraph P1["分区 1"]
               WS1["Warp Scheduler<br/>Dispatch Unit"]
               CU1["INT32 核心 x16<br/>FP32 核心 x16<br/>Tensor Core x4"]
               RF1["寄存器文件"]
           end
           subgraph P2["分区 2"]
               WS2["Warp Scheduler<br/>Dispatch Unit"]
               CU2["INT32 核心 x16<br/>FP32 核心 x16<br/>Tensor Core x4"]
               RF2["寄存器文件"]
           end
           subgraph P3["分区 3"]
               WS3["Warp Scheduler<br/>Dispatch Unit"]
               CU3["INT32 核心 x16<br/>FP32 核心 x16<br/>Tensor Core x4"]
               RF3["寄存器文件"]
           end
       end
       SM --- L1["L1 / 共享内存 (128 KB)<br/>一级缓存 / 纹理缓存<br/>加载/存储单元 (LD/ST)"]

       style SM fill:#f5f5f5,color:#1a1a1a
       style P0 fill:#e3f2fd,color:#1565c0
       style P1 fill:#e3f2fd,color:#1565c0
       style P2 fill:#e3f2fd,color:#1565c0
       style P3 fill:#e3f2fd,color:#1565c0
       style L1 fill:#f3e5f5,color:#7b1fa2

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

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — SM 架构的官方说明
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — SM 指令级微基准测试
