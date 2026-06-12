========================
GPUDirect 技术族
========================

.. epigraph::

   The fastest I/O is the one you don't have to do.

   — 计算机系统设计格言

GPUDirect 是 NVIDIA 的一系列直接数据传输技术族，旨在消除 GPU 通信路径中的 CPU 中转瓶颈。整个技术族围绕一个核心理念——**让数据绕过 CPU，直接从源设备到达 GPU 显存**。

GPUDirect 技术族包含四个主要子技术：

.. list-table::
   :header-rows: 1

   * - 技术
     - 功能
     - 替代的路径
     - 典型场景
   * - GPUDirect RDMA
     - 网卡直接读写 GPU 显存
     - GPU→CPU→NIC 两次 PCIe 传输
     - 多节点分布式训练
   * - GPUDirect Storage (GDS)
     - 存储设备直接读写 GPU 显存
     - GPU→CPU→NVMe 两次拷贝
     - 数据加载、检查点恢复
   * - GPUDirect P2P
     - GPU 之间直接显存访问
     - GPU→CPU→GPU 中转
     - 单节点多 GPU 通信
   * - GPUDirect PeerSync
     - GPU 间直接同步（无需 CPU 轮询）
     - CPU 轮询 GPU 标志位
     - 多 GPU 细粒度协作

.. mermaid::

   flowchart TB
       subgraph Legacy["传统路径（多次拷贝）"]
           direction LR
           GPU["GPU 显存"] -->|PCIe| CPU["CPU DRAM"]
           CPU -->|PCIe| NIC["NIC / NVMe"]
       end

       subgraph Direct["GPUDirect（零拷贝）"]
           direction LR
           GPU2["GPU 显存"] -.->|DMA 直读| NIC2["NIC / NVMe"]
           NIC2 -.->|DMA 直写| GPU2
       end

       style Legacy fill:#fce4ec,color:#b71c1c
       style Direct fill:#e8f5e9,color:#1b5e20

GPUDirect RDMA
===================

GPUDirect RDMA 允许 InfiniBand 或 RoCE 网卡通过 DMA 直接读写 GPU 显存，完全绕过 CPU 和系统内存。这是在多节点训练中实现高效通信的基础。

**数据路径对比**：

.. code-block:: text

   传统路径（无 GPUDirect）:
   GPU 显存 → CPU DRAM (cudaMemcpy) → 系统内存 → NIC → 网络
   三次 PCIe 传输，CPU 参与拷贝

   GPUDirect RDMA:
   GPU 显存 → NIC → 网络
   一次 PCIe 传输，CPU 完全不参与

**实现机制**：

GPUDirect RDMA 的关键在于**内存注册（Memory Registration）**——将 GPU 显存的物理地址映射到 NIC 的 DMA 地址空间：

.. code-block:: text

   1. UMD 调用 cudaMalloc 分配 GPU 显存
   2. KMD 为该分配建立 GPU 页表，获得物理地址
   3. 通过 PCIe BAR 将物理地址暴露给 NIC
   4. NIC 驱动调用 nvidia_p2p_get_pages() 获取 GPU 页面的物理地址
   5. NIC 将物理地址注册到自己的 DMA 引擎
   6. 后续 NIC 可直接读写这些 GPU 页面，无需 CPU 参与

.. code-block:: c

   // NIC 驱动注册 GPU 显存为 RDMA 缓冲区
   struct nvidia_p2p_page_table* page_table;
   nvidia_p2p_get_pages(0, gpu_virt_addr, size, &page_table);

   struct ibv_mr* mr = ibv_reg_mr(pd, page_table->pages, size,
                                   IBV_ACCESS_LOCAL_WRITE);
   // 传输完成后释放
   ibv_dereg_mr(mr);
   nvidia_p2p_put_pages(page_table);

**性能对比**：

.. list-table::
   :header-rows: 1

   * - 指标
     - 传统 (cudaMemcpy + send)
     - GPUDirect RDMA
   * - 单向延迟 (4KB)
     - ~10-15 us
     - ~2-4 us
   * - 带宽 (256MB, IB HDR100)
     - ~85 GB/s
     - ~95 GB/s（接近线速）
   * - CPU 利用率
     - 30-60%
     - < 5%
   * - 内存拷贝次数
     - 2 次 (GPU→CPU, CPU→NIC)
     - 0 次

**网卡选型：InfiniBand vs RoCE**：

.. list-table::
   :header-rows: 1

   * - 特性
     - InfiniBand
     - RoCE
   * - 标准
     - IBTA
     - IEEE 802.1
   * - 典型带宽
     - NDR 400/800 Gbps
     - 200/400 Gbps
   * - 延迟
     - < 1 us
     - ~1-3 us
   * - 流控
     - 基于信用的链路层
     - PFC (802.1Qbb)
   * - GPU 生态
     - 最成熟（ConnectX）
     - 较成熟
   * - 典型部署
     - DGX/HGX 节点间
     - 通用数据中心

GPUDirect Storage (GDS)
============================

GDS 将 GPUDirect 的思路扩展到存储领域，允许 GPU 直接从 NVMe SSD 读写数据，无需经过 CPU 内存中转。

**数据路径对比**：

.. code-block:: text

   传统路径（无 GDS）:
   NVMe SSD → CPU DRAM (DMA) → CPU 内存拷贝 → GPU 显存 (cudaMemcpy)
   两次 PCIe 传输 + 一次 CPU 拷贝

   GDS:
   NVMe SSD → GPU 显存 (DMA)
   一次 PCIe 传输，CPU 零参与

**cuFile API 使用示例**：

.. code-block:: cuda
   :linenos:

   // GDS: 从文件直接读取到 GPU 显存
   #include <cufile.h>

   CUfileDescr_t desc = { .type = CU_FILE_HANDLE_TYPE_OPAQUE_FD };
   desc.handle.fd = open("large_dataset.bin", O_RDONLY);
   CUfileHandle_t handle;
   cuFileHandleRegister(&handle, &desc);

   // 直接从文件读取到 GPU 显存（绕过 CPU）
   cuFileRead(handle, d_gpu_buffer, size, 0, 0);

   // 传统方式对比:
   // fread(h_buf, 1, size, fp);           // NVMe → CPU DRAM
   // cudaMemcpy(d_gpu, h_buf, size, ...); // CPU → GPU

**适用场景**：

- **训练数据加载**：大数据集直接从 SSD 加载到 GPU，减少加载时间 2-5x
- **检查点（checkpoint）**：大模型检查点保存/恢复，绕过 CPU 内存限制
- **推理缓存**：模型权重直接从存储读取到 GPU，减少预热时间

**限制**：

.. code-block:: text

   1. 需要 NVMe SSD 支持 PCIe P2P（大多数数据中心 SSD 支持）
   2. 文件必须位于支持 DAX（Direct Access）的文件系统上
   3. 对于随机小 I/O 场景收益有限（GDS 对小 I/O 的优化不如大块 I/O）
   4. 需要 cuFile API（CUDA 11.4+），与标准 POSIX I/O 不兼容

GPUDirect P2P
==================

GPUDirect P2P（Peer-to-Peer）允许一台机器上的 GPU 之间直接读写对方显存，无需 CPU 中转。

**数据路径对比**：

.. code-block:: text

   传统 P2P:
   GPU 0 显存 → CPU DRAM (cudaMemcpyDeviceToHost)
            → GPU 1 显存 (cudaMemcpyHostToDevice)
   两次 PCIe 传输，CPU 中转

   GPUDirect P2P:
   GPU 0 显存 → GPU 1 显存 (cudaMemcpyPeer / __ldg() 直接读)
   一次 PCIe 传输（或 NVLink），无 CPU 参与

**编程示例**：

.. code-block:: cuda
   :linenos:

   // 启用 P2P 访问
   cudaSetDevice(0);
   cudaDeviceEnablePeerAccess(1, 0);

   // 方案 A: 使用 cudaMemcpyPeer
   cudaMemcpyPeer(dev1_ptr, 1, dev0_ptr, 0, size);

   // 方案 B: 直接在 kernel 中通过指针访问
   // GPU 0 上的 kernel 可直接读取 GPU 1 的显存
   kernel<<<grid, block>>>(dev0_ptr, dev1_ptr);

   __global__ void kernel(float* local, float* peer) {
       // 通过 NVLink 直接读取邻居 GPU 的数据
       float val = peer[threadIdx.x];  // 自动触发 P2P 传输
       local[threadIdx.x] += val;
   }

.. warning::

   P2P 直接指针访问（方案 B）虽然延迟低，但如果频繁跨 GPU 访问会导致大量 P2P 流量。
   最佳实践是将数据聚合后再一次性传输，而不是逐元素访问。

GPUDirect PeerSync
======================

PeerSync 是 GPUDirect 技术族中最晚引入的成员，解决多 GPU 协作时的同步开销问题。

**问题背景**：

.. code-block:: text

   传统多 GPU 同步方式:
   GPU 0 完成计算 → 写入标志位到显存 → CPU 轮询该标志位
   → CPU 通知 GPU 1 开始 → GPU 1 读取标志位

   问题:
   - CPU 轮询消耗大量 CPU 周期
   - CPU→GPU 通知路径延迟 ~5 us
   - 不适合细粒度 GPU 协作

   PeerSync 方案:
   GPU 0 完成计算 → 直接向 GPU 1 发送同步信号
   → GPU 1 自动开始执行

   PeerSync 完全在 GPU 间直接完成同步，CPU 零参与。

**使用场景**：

- **流水线并行**：GPU 0 完成前向传播后直接通知 GPU 1 开始下一阶段
- **多 GPU 流处理**：一个 GPU 完成中间结果后直接触发下游 GPU 的计算
- **NCCL 内部**：NCCL 在某些通信算法中利用 PeerSync 减少同步延迟

与 NCCL 的集成
====================

在多节点训练中，GPUDirect 各子技术协同工作：

.. mermaid::

   flowchart TB
       subgraph Node0["节点 0"]
           direction LR
           G0["GPU 0"] ---|NVLink| G1["GPU 1"]
           G0 ---|NVLink| G2["GPU 2"]
           G1 ---|NVLink| G3["GPU 3"]
           G2 ---|NVLink| G3
           G0 ---|P2P| G1
           G2 ---|P2P| G3
           G0 -.->|GDS| SSD0["NVMe SSD"]
       end

       subgraph Node1["节点 1"]
           direction LR
           G4["GPU 4"] ---|NVLink| G5["GPU 5"]
           G4 ---|NVLink| G6["GPU 6"]
           G5 ---|NVLink| G7["GPU 7"]
           G6 ---|NVLink| G7
       end

       Node0 -->|RDMA / NIC| NET["网络"]
       Node1 -->|RDMA / NIC| NET

       style Node0 fill:#e3f2fd,color:#1565c0
       style Node1 fill:#e3f2fd,color:#1565c0
       style NET fill:#fff3e0,color:#e65100

**通信路径总结**：

.. list-table::
   :header-rows: 1

   * - 路径
     - 技术
     - 带宽
     - 延迟
     - CPU 参与
   * - 单 GPU 读存储
     - GDS
     - PCIe 带宽
     - ~10 us
     - 否
   * - 节点内 GPU↔GPU
     - NVLink + P2P
     - 600-900 GB/s
     - ~1 us
     - 否
   * - 节点内多 GPU 同步
     - PeerSync
     - N/A
     - sub-us
     - 否
   * - 节点间 GPU↔GPU
     - RDMA + GPUDirect
     - 25-200 GB/s
     - ~2-4 us
     - 否
   * - 无 GPUDirect（传统）
     - cudaMemcpy + send
     - 受限于 CPU 带宽
     - ~10-15 us
     - 是（30-60%）

参考与拓展阅读
====================

- NVIDIA GPUDirect Documentation (https://docs.nvidia.com/cuda/gpudirect-rdma/) — GPUDirect RDMA 官方文档
- NVIDIA GPUDirect Storage (https://docs.nvidia.com/gpudirect-storage/) — GDS 用户指南和最佳实践
- NVIDIA GPUDirect PeerSync (https://developer.nvidia.com/gpudirect) — PeerSync 功能介绍
- NCCL Documentation (https://docs.nvidia.com/deeplearning/nccl/) — NCCL 与 GPUDirect 集成说明
