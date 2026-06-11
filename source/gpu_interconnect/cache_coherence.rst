========================
CPU-GPU 缓存一致性
========================

.. epigraph::

   A man's errors are his portals of discovery.

   — James Joyce, 作家

CPU 和 GPU 各自拥有独立的缓存层次结构。当两者共享同一数据时，一个处理器对数据的修改必须对另一个处理器可见——这就是**缓存一致性（cache coherence）** 问题。不同厂商在 CPU-GPU 一致性上采取了截然不同的设计哲学，这直接影响编程模型和性能。

一致性问题的根源
======================

在典型的异构系统中，CPU 和 GPU 各自维护独立的缓存：

.. mermaid::

   flowchart TB
       subgraph CPU_SIDE["CPU 侧"]
           CPU["CPU 核心"]
           L1C["L1 缓存"]
           L2C["L2 缓存"]
           CPU --> L1C --> L2C
       end
       subgraph GPU_SIDE["GPU 侧"]
           SM["SM"]
           L1G["L1 / 共享内存"]
           L2G["L2 缓存"]
           SM --> L1G --> L2G
       end
       L2C ---|PCIe / CXL| L2G
       MEM["主存 (DRAM / HBM)"]
       L2C --- MEM
       L2G --- MEM

       style CPU fill:#e3f2fd,color:#1565c0
       style GPU fill:#f3e5f5,color:#7b1fa2
       style MEM fill:#fff3e0,color:#e65100

**核心问题**：

1. CPU 在 L1/L2 中缓存了地址 X 的数据，修改后尚未写回主存
2. GPU 从主存读取地址 X → 读到过期值
3. 反之亦然：GPU 修改了缓存中的数据，CPU 读到的是旧版本

解决方法分为两大阵营：硬件一致性（由互联协议自动保证）和软件一致性（由程序员/驱动显式同步）。

NVIDIA 方案：无硬件一致性
==============================

NVIDIA 的 GPU 互联方案（PCIe、NVLink）**不提供**硬件缓存一致性。这是 NVIDIA 的故意设计——一致性协议会带来延迟开销和实现复杂度，而 GPU 的工作负载通常不需要 CPU 和 GPU 频繁共享数据。

**设计哲学**：

.. code-block:: text

   NVIDIA 的观点：
   - GPU 工作负载通常是批量处理：CPU 准备数据 → 传输到 GPU → GPU 计算 → 结果回传
   - 在此模式中，CPU 和 GPU 很少同时修改同一数据
   - 硬件一致性的延迟开销 > 显式同步的开销
   - 因此：不实现硬件一致性，由驱动和运行时负责同步

**编程模型的影响**：

.. code-block:: cuda
   :linenos:

   // NVIDIA 方案：显式同步保证数据可见性
   float* data;
   cudaMallocManaged(&data, N * sizeof(float));

   // CPU 填充数据
   for (int i = 0; i < N; i++) data[i] = i;

   // 预取到 GPU（避免缺页延迟）
   cudaMemPrefetchAsync(data, N * sizeof(float), device, stream);

   // GPU 计算
   kernel<<<grid, block, 0, stream>>>(data, N);
   cudaDeviceSynchronize();  // 隐式同步点

   // CPU 读取结果（自动缺页回迁到 CPU）
   float sum = 0;
   for (int i = 0; i < N; i++) sum += data[i];

**Unified Memory 的一致性保证**：

CUDA Unified Memory 提供的是**页粒度的一致性**，通过页错误机制实现：

.. code-block:: text

   写入流程：
   CPU 写 data[i] = i
     → 页面标记为"CPU 修改"，TLB 条目在 GPU 侧失效
   
   GPU 读 data[i]
     → GPU TLB 缺失 → 页错误 → 驱动发现页面在 CPU 侧
     → 将页面迁移到 GPU 显存 → GPU 页表指向新位置
     → 恢复 GPU 线程

   关键语义：
   - cudaDeviceSynchronize() 保证所有 GPU 操作完成后，CPU 可见
   - 同一页面上 CPU 和 GPU 交替访问会产生大量缺页
   - 页面迁移粒度 = 64KB（小页面）或 2MB（大页面）

.. note::

   NVIDIA 的 Unified Memory 不是硬件缓存一致性——它依赖于驱动层的页迁移和 GPU TLB shootdown。这意味着 CPU 修改一个页面后，GPU 需要等到页错误处理完成后才能看到新值。

AMD Infinity Architecture：硬件一致性
============================================

AMD 的 Infinity Architecture 走了一条不同的路——它支持 CPU 和 GPU 间的缓存一致性协议。在 MI300 系列 APU 中，CPU 和 GPU 共享同一内存地址空间，并通过 Infinity Fabric 的一致性协议保证数据可见性。

.. mermaid::

   flowchart TB
       subgraph AMD_CPU["AMD CPU (CCD)"]
           C0["核心 0"] --> L1C0["L1"]
           C1["核心 1"] --> L1C1["L1"]
           L1C0 --> L2C0["L2"]
           L1C1 --> L2C0
           L2C0 --> L3["L3 (共享)"]
       end
       subgraph AMD_GPU["AMD GPU (GCD)"]
           G0["CU 0"] --> L1G["L1 缓存"]
           G1["CU 1"] --> L1G
           L1G --> L2G["L2 缓存"]
       end
       L3 ---|Infinity Fabric<br/>一致性协议| L2G
       L3 --- MEM["HBM3 (统一内存)"]
       L2G --- MEM

       style C0 fill:#e3f2fd,color:#1565c0
       style C1 fill:#e3f2fd,color:#1565c0
       style G0 fill:#f3e5f5,color:#7b1fa2
       style G1 fill:#f3e5f5,color:#7b1fa2
       style MEM fill:#fff3e0,color:#e65100

**一致性协议要点**：

- Infinity Fabric 使用**目录式一致性协议**（directory-based coherence）
- 每个缓存行有一个归属目录（home agent），跟踪该行被哪些处理器缓存
- 当 CPU 写入一个缓存行时，通过探针（probe）使 GPU 侧的副本失效
- 一致性域涵盖 CPU L3 缓存和 GPU L2 缓存

**编程模型的差异**：

.. code-block:: cpp

   // AMD 方案：硬件一致性，无需显式同步
   // 在 MI300 上，CPU 和 GPU 共享同一物理内存

   int* shared_data;
   hipExtMallocWithFlags(
       (void**)&shared_data, N * sizeof(int),
       hipDeviceMallocUncached | hipMallocSignalMemory
   );

   // CPU 写入
   for (int i = 0; i < N; i++) shared_data[i] = i;

   // GPU 直接读取（硬件一致性保证数据可见）
   kernel<<<grid, block>>>(shared_data, N);
   hipDeviceSynchronize();

   // CPU 读取结果（无需缺页，数据已经一致）
   printf("result: %d\n", shared_data[0]);

**Infinity Fabric 一致性 vs NVLink 非一致性**：

.. list-table::
   :header-rows: 1

   * - 特性
     - Infinity Fabric (AMD)
     - NVLink (NVIDIA)
   * - 硬件一致性
     - 支持（目录式协议）
     - 不支持
   * - 一致性域
     - CPU L3 + GPU L2
     - 无（GPU 无 CPU 侧一致性）
   * - 数据同步方式
     - 硬件探针自动失效
     - 驱动页迁移 + TLB shootdown
   * - 共享粒度
     - 缓存行（64 字节）
     - 内存页（64 KB / 2 MB）
   * - CPU 写 GPU 读延迟
     - ~100-300 ns（硬件探针）
     - ~10-50 us（页错误+迁移）
   * - GPU 写 CPU 读延迟
     - ~100-300 ns
     - ~10-50 us
   * - 编程模型
     - 共享内存语义
     - Unified Memory + 显式同步
   * - 功耗开销
     - 较高（探针流量）
     - 较低（按需迁移）

NVLink-C2C：Grace Hopper 的一致性互联
=============================================

NVIDIA 的 Grace Hopper 架构通过 **NVLink-C2C（Chip-to-Chip）** 首次实现了 CPU 和 GPU 之间的硬件一致性互联：

.. mermaid::

   flowchart LR
       subgraph GH["Grace Hopper 超级芯片"]
           direction LR
           subgraph CPU_DIE["Grace CPU"]
               NC["Neoverse V2 核心"] --> CC["一致性控制器<br/>(CHI 协议)"]
           end
           subgraph GPU_DIE["Hopper GPU"]
               SM_D["SM 核心"] --> GUC["GPU 一致性单元"]
           end
           CC ---|NVLink-C2C<br/>~900 GB/s| GUC
           CC --- CPU_MEM["LPDDR5X<br/>~500 GB/s"]
           GUC --- GPU_MEM["HBM3<br/>~3 TB/s"]
       end

       style CPU_DIE fill:#e3f2fd,color:#1565c0
       style GPU_DIE fill:#f3e5f5,color:#7b1fa2

**关键技术细节**：

- NVLink-C2C 基于 Arm **CHI（Coherent Hub Interface）** 协议，这是 Arm AMBA 5 标准中的一致性协议
- Grace CPU 的一致性控制器（CHI Node）和 Hopper GPU 的一致性单元通过 C2C 链路直接通信
- 一致性域覆盖 CPU 的 L3 缓存和 GPU 的 L2 缓存
- 与完全一致的 Infinity Architecture 不同，NVLink-C2C 提供的是**选择性一致性**——显存中的部分区域可以标记为一致性域，其他区域仍按传统显存管理

.. code-block:: cuda

   // Grace Hopper：使用 NVLink-C2C 一致性内存
   // 只有通过特定 API 分配的内存才进入一致性域

   int* coherent_data;
   // 分配 CPU-GPU 一致性内存（NVLink-C2C 保证一致）
   cudaMallocManaged(&coherent_data, N * sizeof(int));
   // Grace Hopper 上的 Unified Memory 可以利用 C2C 硬件一致性

   // CPU 写入 → GPU 硬件探针使其缓存失效
   for (int i = 0; i < N; i++) coherent_data[i] = i;
   // 无需 cudaMemPrefetchAsync，硬件一致性保证可见性

   kernel<<<grid, block>>>(coherent_data, N);
   cudaDeviceSynchronize();

   // CPU 读取 → 无需缺页
   int sum = 0;
   for (int i = 0; i < N; i++) sum += coherent_data[i];

.. note::

   Grace Hopper 的 NVLink-C2C 是一致性和性能的折中方案。它提供了比传统 NVIDIA 方案更细粒度的共享（~100 ns 级延迟），但仍保留了选择性一致性以减少一致性探针带来的带宽开销。

CXL：开放标准的一致性互联
================================

CXL（Compute Express Link）是开放标准的缓存一致性互联协议，得到 Intel、AMD、Arm 等厂商支持。Infinity Fabric 3+ 已与 CXL 互通。

.. list-table::
   :header-rows: 1

   * - CXL 子协议
     - 功能
     - GPU 相关性
   * - CXL.io
     - I/O 语义（类似 PCIe）
     - GPU 枚举和 DMA
   * - CXL.cache
     - 设备缓存 CPU 侧的缓存
     - GPU 缓存 CPU 数据
   * - CXL.mem
     - 设备访问 CPU 主存
     - GPU 直接读取 CPU DRAM

一致性对编程的影响
========================

**编程模式对比**：

.. list-table::
   :header-rows: 1

   * - 场景
     - 无硬件一致性（NVIDIA 传统）
     - 有硬件一致性（AMD IF / NVLink-C2C）
   * - 数据初始化
     - CPU 填充 → 必须 ``cudaMemPrefetchAsync`` 或触发缺页
     - CPU 填充后直接 GPU 读取，硬件保证数据可见
   * - 细粒度共享
     - 代价高：每次切换访问方都触发缺页
     - 代价低：缓存行级失效
   * - 同步要求
     - 必须调用 ``cudaDeviceSynchronize`` 或 event
     - 需要轻量级 fence 但不一定需要全设备同步
   * - 多 GPU 共享
     - NVLink P2P 需显式 ``cudaDeviceEnablePeerAccess``
     - 一致性域扩展至多 GPU
   * - 错误使用后果
     - 读取过期数据（无一致性协议无法检测）
     - 自动检测，但可能性能下降

**性能权衡**：

.. code-block:: text

   无硬件一致性（NVIDIA PCIe/NVLink 传统方案）:
     优点: 无探针流量，GPU 带宽充分用于计算
     缺点: 页迁移延迟 ~10-50 us，不适合细粒度共享
     适合: 批量数据处理、训练、推理

   有硬件一致性（Infinity Fabric / NVLink-C2C）:
     优点: 共享延迟 ~100-300 ns，支持细粒度数据共享
     缺点: 一致性探针占用带宽 ~5-15%
     适合: 实时系统、工作站、CPU-GPU 协同计算

   CXL 方案:
     优点: 开放标准，多厂商互操作
     缺点: 当前带宽受限（与 NVLink 和 IF 相比）
     适合: 专用加速器、异构集群

参考与拓展阅读
====================

- AMD Infinity Architecture Whitepaper (https://www.amd.com/en/products/accelerators/instinct.html) — Infinity Fabric 缓存一致性协议说明
- NVIDIA Grace Hopper Architecture Whitepaper (https://www.nvidia.com/en-us/data-center/grace-hopper/) — NVLink-C2C 一致性方案
- Compute Express Link Specification (https://www.computeexpresslink.org/) — CXL 开放标准协议
- Understanding GPU Page Migration - IISWC 2022 — GPU 页迁移 vs 硬件一致性的实验对比
- ARM AMBA 5 CHI Specification (https://developer.arm.com/architectures/system-architectures/amba/amba-5) — CHI 一致性协议
