========================
GPU 芯片架构总览
========================

.. epigraph::

   Simplicity is the ultimate sophistication.

   — Leonardo da Vinci, 艺术家、科学家、工程师

GPU 是一种**吞吐量优先**（throughput-oriented）的处理器，与 CPU 的**延迟优先**（latency-oriented）设计哲学形成鲜明对比。GPU 将大量晶体管用于计算单元而非缓存和控制逻辑，从而支持大规模线程级并行。

GPU vs CPU 架构对比
======================

.. list-table::
   :header-rows: 1

   * - 维度
     - CPU
     - GPU
   * - 设计目标
     - 降低单线程延迟
     - 提升总吞吐量
   * - 核心数量
     - 少数高性能核心（~10）
     - 大量简单核心（~1000+）
   * - 缓存
     - 大缓存（~50% 芯片面积）
     - 小缓存（~5% 芯片面积）
   * - 控制逻辑
     - 复杂（分支预测、乱序执行）
     - 简单（顺序执行、硬件多线程）
   * - 并行模式
     - 指令级并行（ILP）
     - 线程级并行（TLP）

典型 GPU 芯片结构
=====================

现代 GPU 芯片采用层次化设计：

.. code-block:: text

   +-------------------------------------------------------+
   |                      GPU 芯片                          |
   |  +--------+  +--------+  +--------+  +--------+       |
   |  |  GPC   |  |  GPC   |  |  GPC   |  |  GPC   | ...  |
   |  | +----+ |  | +----+ |  | +----+ |  | +----+ |      |
   |  | | SM | |  | | SM | |  | | SM | |  | | SM | |      |
   |  | +----+ |  | +----+ |  | +----+ |  | +----+ |      |
   |  +--------+  +--------+  +--------+  +--------+       |
   |  +--------+  +--------+  +--------+  +--------+       |
   |  |  HBM   |  |  HBM   |  |  HBM   |  |  HBM   |      |
   |  |  Stack |  |  Stack |  |  Stack |  |  Stack |      |
   |  +--------+  +--------+  +--------+  +--------+       |
   +-------------------------------------------------------+

主要组件
============

**GPC（Graphics Processing Cluster）**
    NVIDIA 架构中的高层次分组，包含多个 SM 和固定功能单元。GPC 是 GPU 芯片布局中的物理组织单位，每个 GPC 包含 1-2 个 TPC（Texture Processing Cluster）和一组固定功能硬件。

GPC 内部结构
----------------

.. code-block:: text

   Graphics Processing Cluster (GPC)
   +--------------------------------------------------+
   |  TPC 0                    TPC 1                    |
   |  +-------------------+   +-------------------+    |
   |  | SM 0     SM 1     |   | SM 2     SM 3     |    |
   |  | 2 个 SM 共享一个   |   | 2 个 SM 共享一个   |    |
   |  | 纹理缓存 (L1T)    |   | 纹理缓存 (L1T)    |    |
   |  +-------------------+   +-------------------+    |
   |                                                    |
   |  Raster Engine (光栅化引擎)                         |
   |  - 三角形设置、光栅化、Z-cull                      |
   |  - 仅在图形工作负载中活跃                           |
   |                                                    |
   |  Ray Intersection Engine (RT Core, Turing+)        |
   |  - BVH 遍历、ray-triangle 求交                     |
   |  - 仅在光线追踪工作负载中活跃                      |
   +--------------------------------------------------+

**SM（Streaming Multiprocessor）**
    NVIDIA GPU 的核心计算单元，包含 CUDA 核心、共享内存、寄存器文件等。
    在 AMD GPU 中称为 Compute Unit (CU)。

**TPC（Texture Processing Cluster）**
    TPC 是 GPC 内部的中间层，包含 2 个 SM 和共享的纹理缓存（L1 Texture Cache）。纹理单元处理纹理采样和过滤操作，在通用计算中也可用作通用的只读数据缓存。

GPU 芯片物理布局
----------------------

现代 GPU 芯片采用层次化的物理布局，不同功能区在芯片上的面积分配反映了 GPU 的设计重点：

.. code-block:: text

   Ampere GA100 芯片面积的近似分配：

   +------------------------------------------+
   |  SM / CUDA Core       ~45%               |  ← 计算单元（占近一半面积）
   |  SRAM (L2 + Register) ~20%               |  ← 片上存储
   |  HBM PHY + I/O        ~15%               |  ← 内存接口
   |  NVLink / PCIe PHY    ~8%                |  ← 互联接口
   |  固定功能硬件         ~7%                |  ← 光栅化、视频编解码等
   |  其他 (clock, scan)   ~5%                |  ← 时钟、扫描链、测试
   +------------------------------------------+

**对比**: CPU 的芯片面积分配通常为：缓存 ~50%、控制逻辑 ~25%、计算 ~25%。GPU 将超过 45% 的面积用于计算单元，充分体现了吞吐优先的设计哲学。

**Chiplet（小芯片）设计**:

从 AMD MI300 和 NVIDIA Grace Hopper 开始，GPU 也在向 chiplet 设计演进：

.. code-block:: text

   AMD MI300X Chiplet 布局:
   +--------+--------+--------+--------+
   |  GCD 0  |  GCD 1  |  GCD 2  |  GCD 3  |
   |  (GPU)  |  (GPU)  |  (GPU)  |  (GPU)  |
   +--------+--------+--------+--------+
   |  GCD 4  |  GCD 5  |  GCD 6  |  GCD 7  |
   |  (GPU)  |  (GPU)  |  (GPU)  |  (GPU)  |
   +--------+--------+--------+--------+
   |  CCD 0  |  CCD 1  |  CCD 2  |  CCD 3  |
   |  (CPU)  |  (CPU)  |  (CPU)  |  (CPU)  |
   +--------+--------+--------+--------+
   |         I/O Die (Infinity Fabric)       |
   +----------------------------------------+

   Chiplet 优点: 更高的芯片良率、灵活的配置组合、更低的研发成本
   Chiplet 缺点: 跨 chiplet 通信延迟增加、功耗略高、热管理更复杂

内存控制器架构
------------------

GPU 的内存控制器负责管理高带宽内存（HBM）的访问：

.. code-block:: text

   GPU 芯片
   +----------------------------------------------------+
   |                        Crossbar                     |
   |   +-------+-------+-------+-------+-------+-------+ |
   |   | Mem   | Mem   | Mem   | Mem   | Mem   | Mem   | |
   |   | Ctrl 0| Ctrl 1| Ctrl 2| Ctrl 3| Ctrl 4| Ctrl 5| |
   |   +---+---+---+---+---+---+---+---+---+---+---+---+ |
   |       |       |       |       |       |       |     |
   |   +---+---+---+---+---+---+---+---+---+---+---+----+|
   |   | HBM  | HBM  | HBM  | HBM  | HBM  | HBM  |     ||
   |   | Phy  | Phy  | Phy  | Phy  | Phy  | Phy  |     ||
   |   +------+------+------+------+------+------+     ||
   |       |       |       |       |       |       |     |
   +-------+-------+-------+-------+-------+-------+-----+
           |       |       |       |       |       |
        HBM0    HBM1    HBM2    HBM3    HBM4    HBM5
        Stack   Stack   Stack   Stack   Stack   Stack

**内存分区（Memory Partition）**:

L2 缓存和内存控制器被划分为多个独立分区。每个分区管理一段连续的显存地址范围：

.. list-table::
   :header-rows: 1

   * - 架构
     - 内存分区数
     - 每分区 L2 大小
     - 总 HBM 容量
     - HBM 堆栈数
   * - V100 (Volta)
     - 32
     - 128 KB
     - 16/32 GB
     - 4
   * - A100 (Ampere)
     - 40
     - 1 MB (40 MB total)
     - 40/80 GB
     - 5
   * - H100 (Hopper)
     - 60
     - 667 KB (40 MB total)
     - 80 GB
     - 6
   * - MI300X (CDNA 3)
     - 8 (每 GCD 1 个)
     - 16 MB (per GCD)
     - 192 GB
     - 8

跨分区访问的延迟差异称为 **分区不平衡（partition camping）**。当大量线程访问同一内存分区的地址时，该分区成为瓶颈，而其他分区空闲。这是 GPU 内存级并行性优化的一个重要考量。

片上互联网络
--------------------

GPU 芯片内部各个组件之间通过片上网络（Network-on-Chip, NoC）互联：

.. code-block:: text

   Ampere GA100 片上互联拓扑：

   SM Cluster  ──→  Crossbar (L2切片)  ──→  Memory Partition
        ↑                      ↓
   SM Cluster  ──→  Crossbar (L2切片)  ──→  Memory Partition
        ↑                      ↓
   SM Cluster  ──→  Crossbar (L2切片)  ──→  Memory Partition

   特点:
   - Crossbar 拓扑: 任何 SM 可访问任何内存分区
   - 非均匀访问: 不同 SM 到不同内存分区的延迟略有差异
   - 带宽均衡: 通过交叉开关实现负载均衡

**NVLink / Infinity Fabric**
    GPU 之间的高速互联，用于多 GPU 通信和跨节点数据传输。

主流架构演进
================

.. list-table::
   :header-rows: 1

   * - 厂商
     - 架构
     - 特点
   * - NVIDIA
     - Volta (SM 7.0)
     - Tensor Core 引入、独立线程调度
   * - NVIDIA
     - Turing (SM 7.5)
     - RT Core、异步计算、MIG
   * - NVIDIA
     - Ampere (SM 8.0)
     - 第三代 Tensor Core、L2 翻倍
   * - NVIDIA
     - Hopper (SM 9.0)
     - DPX 指令、Transformer Engine、NVLink Switch
   * - AMD
     - CDNA 2/3
     - Matrix Core、Infinity Fabric、3D V-Cache
   * - AMD
     - RDNA 3
     - 小芯片（chiplet）设计、统一计算单元

参考与拓展阅读
====================

- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — 通过微基准测试分析 Ampere 芯片布局和内存延迟
- NVIDIA H100 Tensor Core GPU Architecture (https://www.nvidia.com/en-us/data-center/h100/) — NVIDIA Hopper 架构白皮书
- NVIDIA Volta Architecture - IEEE Micro 2018 — Volta 架构微架构分析
- AMD CDNA 3 Architecture Whitepaper (https://www.amd.com/en/products/accelerators/instinct/cdna-3.html) — AMD CDNA 3 架构详细规格
- AMD CDNA 2 Architecture - HPCA 2023 — CDNA 2 Chiplet 设计分析
