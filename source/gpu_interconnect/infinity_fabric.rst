==========================
AMD Infinity Fabric
==========================

.. epigraph::

   Where there is unity there is always victory.

   — Publilius Syrus, 拉丁格言作者

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
