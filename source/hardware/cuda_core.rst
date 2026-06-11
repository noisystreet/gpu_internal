========================
CUDA Core
========================

CUDA Core 是 NVIDIA SM 中的标量计算单元，负责执行 FP32、INT32、FP64 等算术运算。每个 SM 包含数百个 CUDA Core 分布在多个 subcore 中。

CUDA Core 微架构
====================

CUDA core 本质上是经过优化的流水线化 ALU，围绕 **FMA（Fused Multiply-Add）** 流水线设计，执行 ``D = A * B + C`` 运算。其核心执行路径如下：

.. mermaid::

   flowchart LR
       I["指令"] --> WS["Warp Scheduler"]
       WS --> D["Dispatch"]
       D --> OC["Operand Collect"]
       OC --> RF["寄存器文件读取<br/>(每个线程 2 个 op)"]
       RF --> FMA["FMA Pipeline"]
       FMA --> WB["Writeback"]

       style I fill:#e3f2fd,color:#1565c0
       style WS fill:#e3f2fd,color:#1565c0
       style FMA fill:#f3e5f5,color:#7b1fa2
       style WB fill:#e8f5e9,color:#1b5e20

**流水线深度**: FP32 FMA 通常为 4-6 个周期（取决于架构代际），意味着一个 warp 的指令发射后需要 4-6 周期才能得到结果。

FP32 与 INT32 并行
====================

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
     - 1 条
     - 6 周期
   * - Pascal
     - FP32 可执行 INT32（互斥）
     - 1 条
     - 6 周期
   * - Volta
     - FP32 + INT32 独立
     - 2 条
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

指令吞吐（Instruction Throughput）
======================================

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

Warp 在 CUDA Core 上的执行
============================

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

:doc:`sfu` 是 SM 中专门处理超越函数的硬件单元，详见单独章节。

**加载/存储单元（LD/ST）**
    处理全局内存和共享内存的加载/存储指令，负责地址生成和内存合并。

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 核心架构说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 指令吞吐优化
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — 指令级延迟分析
