==========================
AMD Infinity Fabric
==========================

.. epigraph::

   Where there is unity there is always victory.

   — Publilius Syrus

Infinity Fabric 是 AMD 推出的多芯片互联技术，用于连接同一封装内的 CCD（Core Compute Die）、I/O Die 以及 GPU 模块，也用于 GPU 之间的机架级互联。

Infinity Fabric 架构概览
==============================

Infinity Fabric 由两个逻辑层组成：

.. code-block:: text

   +------------------------------------------+
   |           Infinity Fabric                 |
   |  +------------------+  +--------------+   |
   |  | 数据传输层 (SDF)  |  | 控制管理层   |   |
   |  | Scalable Data     |  | (SCF)        |   |
   |  | Fabric            |  | Scalable     |   |
   |  |                   |  | Control      |   |
   |  | 负责数据包路由     |  | Fabric       |   |
   |  | 重排序、协议转换   |  | 负责电源管理  |  |
   |  +------------------+  | 时钟同步      |   |
   |                         | 初始化/配置   |   |
   |                         +--------------+   |
   +------------------------------------------+

Infinity Fabric 的**关键特性**:

- **统一协议**：同一条链路可承载内存访问、I/O 事务、缓存一致性协议
- **CXL 兼容**：Infinity Fabric 协议层与 CXL（Compute Express Link）互通
- **分布式路由**：通过集成在 Die 上的路由表实现多跳拓扑
- **自适应带宽**：根据链路质量动态调节传输速率

AMD GPU 互联拓扑
======================

**MI300X 八路 GPU 互联**:

.. code-block:: text

   +-------------------+    +-------------------+
   |   GCD 0 (GPU)     |    |   GCD 1 (GPU)     |
   |   +-----------+   |    |   +-----------+   |
   |   | XCD + HBM |   |    |   | XCD + HBM |   |
   |   +-----------+   |    |   +-----------+   |
   +--------+----------+    +----------+--------+
            |                          |
   +--------+----------+    +----------+--------+
   |   GCD 2 (GPU)     |    |   GCD 3 (GPU)     |
   |   +-----------+   |    |   +-----------+   |
   |   | XCD + HBM |   |    |   | XCD + HBM |   |
   |   +-----------+   |    |   +-----------+   |
   +-------------------+    +-------------------+

   每个 GCD 通过 6 条 Infinity Fabric 链路互联
   总聚合带宽 ~896 GB/s

Infinity Fabric vs NVLink
==============================

.. list-table::
   :header-rows: 1

   * - 特性
     - Infinity Fabric 4.x
     - NVLink 4.0
   * - 单链路带宽
     - ~64 GB/s (双向)
     - 50 GB/s (双向)
   * - 拓扑
     - 分布式路由 (多跳)
     - 直连 + NVSwitch (单跳)
   * - 缓存一致性
     - 支持 (x86 统一寻址)
     - 不支持
   * - 协议兼容
     - CXL 互通
     - 私有协议
   * - 芯片内互联
     - 支持 (Chiplet)
     - 不支持
   * - 典型 GPU 互联
     - 6 链路 / GCD
     - 18 链路 / GPU

**Infinity Architecture** 的核心设计理念是统一 CPU 和 GPU 的内存语义。在 MI300 系列中，CPU 和 GPU 共享同一内存地址空间，通过 Infinity Fabric 的缓存一致性协议实现真正的统一内存访问。

分布式路由机制
-----------------------

Infinity Fabric 采用**分布式路由**而非集中式交换机，每个 Die 上的路由表决定数据包的转发路径：

.. code-block:: text

   数据包从 GCD 0 到 GCD 5 的路由路径:

   GCD 0 → GCD 1 → GCD 3 → GCD 5
   |        |        |        |
   跳1      跳2      跳3      跳4
   本地      XCD      远程     目标
   FIFO     转发      FIFO     GCD

   每跳延迟 ~20-50 ns（视数据大小和链路负载）

   多跳拓扑的相对带宽衰减:
   单跳:  ~64 GB/s (100%)
   双跳:  ~55 GB/s (~86%)
   三跳:  ~42 GB/s (~66%)

**路由表配置**:

.. code-block:: text

   每跳节点的路由表中存储了目标 GCD 到出端口的映射：
   +----------------+------------------+
   | 目标 GCD       | 出端口 (Link ID) |
   +----------------+------------------+
   | GCD 0 (本地)   | N/A (片上互联)     |
   | GCD 1 (邻居)   | Link 0           |
   | GCD 2 (2跳)    | Link 0 → Link 1  |
   | ...            | ...              |
   | GCD 7 (远端)   | Link 0 → ...     |
   +----------------+------------------+

   Infinity Fabric 的路由表在初始化时由固件建立，
   通信时硬件根据数据包目标地址自动选择路径。

**自适应带宽管理**:

Infinity Fabric 支持链路状态动态调整：

- 高负载时自动升级链路速率（P-state 调节）
- 检测到链路错误时降级并重新训练
- 空闲链路进入低功耗状态（L0p → L1 → L2）

Infinity Fabric vs NVLink 性能对比
==========================================

.. list-table::
   :header-rows: 1

   * - 性能指标
     - Infinity Fabric 4.x (MI300X)
     - NVLink 4.0 (H100)
   * - 单链路带宽
     - ~64 GB/s (双向)
     - 50 GB/s (双向)
   * - 每 GPU 链路数
     - 6 (MI300X 每 GCD)
     - 18
   * - 总聚合带宽
     - ~896 GB/s (8 GCD)
     - 900 GB/s
   * - GPU间延迟 (P2P)
     - ~100-200 ns (单跳)
     - ~50-100 ns (直连)
   * - 缓存一致性延迟
     - ~200-400 ns
     - 不支持
   * - 实测 AllReduce (1GB)
     - ~200 us (32 ranks)
     - ~180 us (32 ranks)

.. code-block:: cpp

   // AMD 统一内存架构：通过 HIP 使用细粒度共享内存
   // 不需要显式 cudaMemcpy，CPU/GPU 直接访问同一指针

   int* shared_data;
   // 分配 CPU-GPU 共享内存 (MI300 统一寻址)
   hipExtMallocWithFlags(
       (void**)&shared_data, N * sizeof(int),
       hipDeviceMallocUncached | hipMallocSignalMemory
   );

   // CPU 填充数据
   for (int i = 0; i < N; i++) shared_data[i] = i;

   // GPU 直接访问 (缓存一致性保证)
   kernel<<<grid, block>>>(shared_data, N);
   hipDeviceSynchronize();

参考与拓展阅读
====================

- AMD Infinity Architecture (https://www.amd.com/en/products/accelerators/instinct.html) — AMD Infinity Architecture 白皮书
- AMD ROCm Documentation (https://rocm.docs.amd.com/) — ROCm 中 Infinity Fabric 和统一内存的 API 文档
- AMD CDNA 3 Architecture Whitepaper (https://www.amd.com/en/products/accelerators/instinct/cdna-3.html) — CDNA 3 Chiplet 互联架构的详细说明
