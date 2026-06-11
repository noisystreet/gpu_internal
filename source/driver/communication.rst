========================
主机-设备通信
========================

.. epigraph::

   Information is the resolution of uncertainty.

   — Claude Shannon

主机（CPU）与设备（GPU）之间的通信是 GPU 计算的关键瓶颈之一。理解通信机制和优化数据传输是提升 GPU 应用性能的核心。

.. _driver-communication-pcie:

PCIe 通信
==============

GPU 通常通过 PCI Express 总线连接到主机。

.. list-table::
   :header-rows: 1

   * - PCIe 版本
     - x16 单向带宽
     - x16 双向带宽
   * - PCIe 3.0
     - ~16 GB/s
     - ~32 GB/s
   * - PCIe 4.0
     - ~32 GB/s
     - ~64 GB/s
   * - PCIe 5.0
     - ~64 GB/s
     - ~128 GB/s

命令提交机制
================

CPU 提交命令到 GPU 的核心机制是通过 **PUSH 缓冲区（Push Buffer）**。当应用程序调用 CUDA API 时，背后的过程远不止一条指令——它是一条从用户空间到硬件寄存器的长链。

.. mermaid::

   sequenceDiagram
       participant App as 应用程序
       participant UMD as 用户态驱动 (UMD)
       participant KMD as 内核态驱动 (KMD)
       participant MMIO as 硬件 MMIO/Doorbell
       participant GPU as GPU 固件
       participant SM as SM/CU

       App->>UMD: (1) 构造命令
       UMD->>UMD: 构建命令缓冲区
       UMD->>KMD: (2) 调用 ioctl
       KMD->>KMD: (3) 验证并映射
       KMD->>KMD: (4) 写 PUSH 缓冲区
       KMD->>MMIO: (5) 写入 GPU 寄存器
       MMIO->>GPU: Doorbell 通知
       GPU->>GPU: (6) 解析命令
       GPU->>GPU: (7) 调度到执行单元
       GPU->>SM: 执行

**Doorbell 机制**:

驱动通过写入 GPU 的 MMIO 空间（称为 Doorbell 寄存器）通知 GPU 有新命令可用。Doorbell 写入操作非常轻量，只需一次内存映射写入。

**Ring Buffer（环形缓冲区）**:

命令缓冲区通常组织为环状结构，GPU 通过读取生产/消费指针获取新的命令：

.. mermaid::

   flowchart LR
       subgraph Ring["环形缓冲区"]
           CMD0["cmd 0"] --- CMD1["cmd 1"] --- CMD2["cmd 2"]
           CMD2 --- CMD3["cmd 3"] --- DOTS["..."] --- CMDN["cmd N"]
       end
       CP["消费指针 &#40GPU 更新&#41"] -.-> CMD2
       PP["生产指针 &#40CPU 更新&#41"] -.-> CMDN

       style Ring fill:#e8eaf6,color:#283593
       style CP fill:#f3e5f5,color:#7b1fa2
       style PP fill:#e3f2fd,color:#1565c0

同步机制
============

有了命令提交机制之后，下一个关键问题是如何协调主机和设备之间的执行顺序。GPU 是异步执行的——CPU 提交命令后立即返回，而 GPU 可能还在执行之前的命令。同步机制就是解决这种"各说各话"问题的桥梁。

**Fence（栅栏）**
    Fence 是 GPU 和 CPU 之间同步的基本原语。CPU 写入一个值到 GPU 内存，GPU 在处理完命令后写入信号值（signal），CPU 轮询或等待该值变化。

.. code-block:: c
   :linenos:

   // Fence 操作简化的伪代码
   // CPU 端
   uint64_t* fence = mmap_gpu_memory();
   *fence = 0;
   submit_command(gpu, "完成后将 fence 设为 1");
   while (*fence != 1) {
       // 忙等待或 yield
       _mm_pause();
   }

**CUDA 事件（Event）**
    CUDA Event 封装了硬件 fence/timestamp 操作：

.. code-block:: cuda
   :linenos:

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start, stream);
   kernel<<<grid, block, 0, stream>>>();
   cudaEventRecord(stop, stream);

   cudaEventSynchronize(stop);
   float ms;
   cudaEventElapsedTime(&ms, start, stop);

**Stream 与并发**
    CUDA Stream 是 GPU 操作的命令序列。不同流的操作可并发执行：

.. code-block:: cuda

   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   // 两个 kernel 可在不同流中并发执行
   kernel1<<<grid, block, 0, stream1>>>(...);
   kernel2<<<grid, block, 0, stream2>>>(...);

DMA 引擎
============

前面的命令提交和同步机制解决了"如何"和"何时"执行的问题，但 GPU 计算中另一个关键操作是数据传输。现代 GPU 包含专用的 DMA 引擎来处理主机-设备数据传输，让 CPU 得以从数据搬运中解放出来。

- **双向 DMA**: 支持同时上传和下载
- **DMA 重叠**: 数据传输可以与 kernel 执行重叠
- **Peer-to-Peer**: 通过 NVLink/NVSwitch 的 GPU 间直接 DMA

.. code-block:: cuda
   :linenos:

   // 使用流实现传输与计算重叠
   cudaStream_t stream;
   cudaStreamCreate(&stream);

   // 异步数据传输
   cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

   // kernel 在相同流中排队，等待传输完成
   kernel<<<grid, block, 0, stream>>>(d_data);

   // 异步结果回传
   cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream);

统一内存（Unified Memory）
==============================

前面的数据传输需要程序员显式管理——决定什么时候上传数据、什么时候下载结果。统一内存试图消除这种负担，让 CPU 和 GPU 共享同一地址空间。

统一内存（CUDA 6+）提供 CPU 和 GPU 之间的自动数据迁移：

.. code-block:: cuda

   // 统一内存分配
   float* data;
   cudaMallocManaged(&data, N * sizeof(float));

   // CPU 访问 — 自动触发 page fault 迁移到主机
   for (int i = 0; i < N; i++) data[i] = i;

   // GPU 访问 — 自动触发 page fault 迁移到设备
   kernel<<<grid, block>>>(data, N);
   cudaDeviceSynchronize();

**机制**: 通过 GPU 页错误和驱动页迁移引擎实现。首次访问缺失页面时触发页错误，驱动将页面迁移到访问方的内存。

**优化**: ``cudaMemAdvise()`` 和 ``cudaMemPrefetchAsync()`` 可以提前提示驱动进行迁移，减少页错误开销。

GPU 页错误与内存迁移
=============================

统一内存的实现依赖于 GPU 页错误（page fault）和驱动页迁移引擎。

**页错误流程**:

.. code-block:: text

   GPU 执行 kernel:
   1. 访问统一内存地址 A
   2. GPU MMU 查找页表 → 页面不在本地显存
   3. GPU 触发页错误，暂停访问线程
   4. 驱动捕获页错误：
      a. 定位页面当前位置（CPU DRAM / 其他 GPU 显存 / 磁盘交换区）
      b. 如果页面在 CPU 端：
         - 通过 PCIe DMA 将页面拷贝到 GPU 显存
      c. 如果页面在其他 GPU：
         - 通过 NVLink P2P 直接迁移
      d. 更新 GPU 页表 → 指向新位置
   5. 恢复等待的线程，继续执行

**迁移策略**:

.. list-table::
   :header-rows: 1

   * - 策略
     - 触发方式
     - 优点
     - 缺点
   * - Demand Paging（按需）
     - 访问缺失触发
     - 透明、无需干预
     - 每次缺页有 ~10us 延迟
   * - Prefetch（预取）
     - ``cudaMemPrefetchAsync``
     - 提前迁移、无缺页延迟
     - 需提前知道访问模式
   * - Advise（提示）
     - ``cudaMemAdvise``
     - 驱动可根据提示智能迁移
     - 仅作为参考，不保证
   * - Read Duplication
     - ``cudaMemAdviseSetReadMostly``
     - 只读页面复制到多个位置
     - 写操作需同步

**GPU TLB 管理**:

GPU 的 MMU 包含多级 TLB（Translation Lookaside Buffer）。缓存的页表条目在页迁移后需要失效：

.. code-block:: text

   全局 TLB（L2 TLB）: 所有 SM 共享，~2K-4K 条目
                 |
     ┌───────┬───┴───┬───────┐
     │       │       │       │
   SM 0    SM 1    SM 2    SM 3
   L1 TLB  L1 TLB  L1 TLB  L1 TLB  (~128-256 条目)

   TLB shootdown: 页迁移后，所有 SM 的 TLB 条目需要失效
   - 全局 shootdown 延迟约 1-3 us
   - 频繁页迁移会导致 TLB 颠簸，性能急剧下降

**页错误对性能的影响**:

.. code-block:: cuda

   // 页错误优化的最佳实践
   // 1. 已知访问模式 — 显式预取
   float* data;
   cudaMallocManaged(&data, N * sizeof(float));
   // 在 kernel 启动前预取到 GPU
   cudaMemPrefetchAsync(data, N * sizeof(float), deviceId, stream);
   kernel<<<grid, block, 0, stream>>>(data, N);
   cudaDeviceSynchronize();

   // 2. 只读数据 — 使用读取复制
   cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetReadMostly, deviceId);

   // 3. 避免频繁 CPU-GPU 交替访问 — 尽量批量处理
   //    每次切换访问位置都会触发批量页迁移

**Linux 上的页迁移开销实测参考**:

.. code-block:: text

   迁移 4 KB 页面 (PCIe 4.0 x16):        ~1.5 us
   迁移 2 MB 大页 (PCIe 4.0 x16):        ~50 us
   GPU MMU TLB shootdown (全局):          ~1-3 us
   批处理页迁移 (512 个页面, 2 MB):       ~200 us
   NVLink P2P 页迁移 (4 KB):               ~0.3 us

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中关于统一内存和流同步的章节
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 主机-设备数据传输优化
- Understanding GPU Page Migration - IISWC 2022 — GPU 页错误和迁移机制的实验分析
