==========================
NVLink 与 NVSwitch
==========================

.. epigraph::

   The whole is greater than the sum of its parts.

   — Aristotle

NVLink 是 NVIDIA 开发的高带宽 GPU 间直接互联协议，旨在突破 PCIe 的带宽瓶颈。NVSwitch 是 NVLink 的全连接交换机，实现 GPU 之间的任意拓扑互联。在深入 NVLink 的具体规格之前，先从一个关键问题出发：为什么 PCIe 不够用？

NVLink 演进
=================

NVLink 自首次推出以来经历了多代演进，每一代的带宽和链路数都在增长：

.. list-table::
   :header-rows: 1

   * - 版本
     - 单链路带宽
     - 每 GPU 链路数
     - 总带宽 (双向)
   * - NVLink 1.0 (P100)
     - 20 GB/s
     - 4
     - 160 GB/s
   * - NVLink 2.0 (V100)
     - 25 GB/s
     - 6
     - 300 GB/s
   * - NVLink 3.0 (A100)
     - 50 GB/s
     - 12
     - 600 GB/s
   * - NVLink 4.0 (H100)
     - 50 GB/s
     - 18
     - 900 GB/s
   * - NVLink 5.0 (B100)
     - 100 GB/s
     - 18
     - 1800 GB/s

与 PCIe 对比
=================

先看一组直观的数字对比：

.. code-block:: text

   传输 1 GB 数据的理论延迟对比:

   PCIe 5.0 x16:   64 GB/s   → ~16 ms
   NVLink 4.0 x18: 900 GB/s  → ~1.1 ms (双向)
   NVLink 5.0 x18: 1800 GB/s → ~0.6 ms (双向)

   NVLink 的优势不仅在于更高带宽，还在于更低的通信延迟
   和更少的 CPU 参与（GPU 可直接读写对端显存）。

NVSwitch 拓扑
=================

NVSwitch 实现**全互联（All-to-All）** 拓扑：

.. mermaid::

   flowchart TB
       subgraph PCIe["传统 PCIe 拓扑"]
           direction TB
           GPU0[GPU 0] --- GPU1[GPU 1]
           GPU0 --- PCIeSW1[PCIe SW]
           GPU1 --- PCIeSW2[PCIe SW]
           PCIeSW1 --- CPU[CPU + DRAM]
           PCIeSW2 --- CPU
       end

       subgraph NVSwitch_Top["NVSwitch 拓扑 (DGX/HGX)"]
           direction LR
           NVSW["NVSwitch"] --- G0[GPU 0]
           NVSW --- G1[GPU 1]
           NVSW --- G2[GPU 2]
           NVSW --- G3[GPU 3]
           NVSW --- G4[GPU 4]
           NVSW --- G5[GPU 5]
           NVSW --- G6[GPU 6]
           NVSW --- G7[GPU 7]
       end

       style CPU fill:#f3e5f5,color:#7b1fa2
       style NVSW fill:#fff3e0,color:#e65100
       style PCIe fill:#e8eaf6,color:#283593

**关键特性**:

- **全带宽 All-to-All**: 任意两 GPU 之间均以 NVLink 全带宽通信
- **SHARP 技术**: 交换机内置规约引擎，支持树形规约（AllReduce）在交换机中完成，无需 GPU 参与
- **NVLink 域**: 划分 GPU 子集为独立通信域，减少广播开销

NVLink 的 GPU Direct 技术
==============================

GPU Direct 是一系列技术的统称，允许第三方设备（存储、网卡、GPU）绕过 CPU，直接与 GPU 进行数据交换：

.. list-table::
   :header-rows: 1

   * - 技术
     - 功能
   * - GPU Direct RDMA
     - IB/RoCE 网卡直接读写 GPU 显存
   * - GPU Direct P2P
     - GPU 直接访问对端 GPU 显存（通过 NVLink 或 PCIe）
   * - GPU Direct Storage
     - NVMe SSD 直接传输数据到 GPU 显存

.. code-block:: cuda
   :linenos:

   // GPU Direct P2P 访问：启用和验证
   int canAccess;
   cudaDeviceCanAccessPeer(&canAccess, device0, device1);
   if (canAccess) {
       cudaDeviceEnablePeerAccess(device1, 0);
       // 现在 GPU 0 可以直接访问 GPU 1 的显存
       // 无需 cudaMemcpy，直接指针访问
       kernel<<<grid, block>>>(d_ptr_on_gpu1);
   }

GPUDirect RDMA 深入
---------------------------

GPUDirect RDMA 允许 InfiniBand 或 RoCE 网卡通过 DMA 直接读写 GPU 显存，完全绕过 CPU 和系统内存。这是在多节点训练中实现高效通信的基础。

**数据路径对比**:

.. code-block:: text

   传统路径 (无 GPUDirect):
   GPU 显存 → CPU DRAM (cudaMemcpy) → 系统内存 → NIC → 网络
   ↓               ↓                     ↓
   1. PCIe 读取    2. CPU 内存拷贝       3. PCIe 写入
   三次 PCIe 传输，CPU 参与拷贝

   GPUDirect RDMA:
   GPU 显存 → NIC → 网络
   ↓               ↓
   1. NIC DMA 读取 2. 直接发送
   一次 PCIe 传输，CPU 完全不参与

**实现机制**:

GPUDirect RDMA 的关键在于**内存注册（Memory Registration）**——将 GPU 显存的物理地址映射到 NIC 的 DMA 地址空间：

.. code-block:: text

   1. UMD 调用 cudaMalloc 分配 GPU 显存
   2. KMD 为该分配建立 GPU 页表，获得物理地址
   3. 通过 PCIe BAR 将物理地址暴露给 NIC
   4. NIC 驱动调用 nvidia_p2p_get_pages() 获取 GPU 页面的物理地址
   5. NIC 将物理地址注册到自己的 DMA 引擎
   6. 后续 NIC 可直接读写这些 GPU 页面，无需 CPU 参与

**网卡注册 GPU 内存流程**:

.. code-block:: c

   // 伪代码：NIC 驱动注册 GPU 显存为 RDMA 缓冲区
   // 1. 获取 GPU 页面的物理地址映射
   struct nvidia_p2p_page_table* page_table;
   nvidia_p2p_get_pages(0,  // GPU 0
                        gpu_virt_addr,
                        size,
                        &page_table);

   // 2. NIC 注册这些物理页面到自己的 DMA 引擎
   struct ibv_mr* mr = ibv_reg_mr(pd,
                        page_table->pages,
                        size,
                        IBV_ACCESS_LOCAL_WRITE);

   // 3. 传输完成后释放
   ibv_dereg_mr(mr);
   nvidia_p2p_put_pages(page_table);

**GPUDirect RDMA vs 传统路径实测对比**:

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
     - ~95 GB/s (接近线速)
   * - CPU 利用率
     - 30-60%
     - < 5%
   * - 内存拷贝次数
     - 2 次 (GPU→CPU, CPU→NIC)
     - 0 次

**InfiniBand vs RoCE**:

.. list-table::
   :header-rows: 1

   * - 特性
     - InfiniBand
     - RoCE (RDMA over Converged Ethernet)
   * - 标准
     - IBTA (InfiniBand Trade Association)
     - IEEE 802.1 (以太网)
   * - 典型带宽
     - NDR 400/800 Gbps
     - 200/400 Gbps
   * - 延迟
     - < 1 us
     - ~1-3 us
   * - 流控
     - 基于信用的链路层流控
     - PFC (Priority Flow Control, 802.1Qbb)
   * - 拥塞控制
     - 内置 CC
     - DCQCN (基于 ECN)
   * - GPU 结合生态
     - 最成熟（Mellanox/NVIDIA ConnectX）
     - 较成熟（ConnectX RoCE）
   * - 典型部署
     - DGX/HGX 节点间
     - 通用数据中心

**GPUDirect Storage (GDS)**:

GDS 是 GPU Direct 技术在存储领域的扩展，允许 GPU 直接从 NVMe SSD 读写数据：

.. code-block:: cuda

   // GDS: 从文件直接读取到 GPU 显存
   #include <cufile.h>

   CUfileDescr_t desc = { .type = CU_FILE_HANDLE_TYPE_OPAQUE_FD };
   desc.handle.fd = open("large_dataset.bin", O_RDONLY);
   CUfileHandle_t handle;
   cuFileHandleRegister(&handle, &desc);

   // 直接从文件读取到 GPU 显存（绕过 CPU）
   cuFileRead(handle, d_gpu_buffer, size, 0, 0);

   // 传统方式需要: fread → cudaMemcpy 两步
   // GDS 直接: GPU 显存 ← 文件（单步）

**与 NCCL 的关系**:

在多节点训练中，节点内使用 NVLink（通过 NCCL），节点间使用 RDMA：

.. code-block:: text

   单节点训练:
   GPU 0 ──NVLink── GPU 1 ──NVLink── GPU 2 ──NVLink── GPU 3
                        ↑
                    (NCCL 自动选择 NVLS/Simple 协议)

   多节点训练:
   节点 0: GPU 0─NVLink─GPU 1         节点 1: GPU 4─NVLink─GPU 5
               │         │                    │         │
            NCCL-Net                    NCCL-Net
               │         │                    │         │
            ┌──┴─────────┴──┐          ┌──────┴─────────┴─┐
            │ IB/RoCE NIC   │          │   IB/RoCE NIC    │
            └───────┬───────┘          └────────┬─────────┘
                    │                           │
                    └─────────── RDMA ──────────┘
                    (NCCL 通过 GPUDirect RDMA 通信)

GPU 间通信性能模型
=========================

**Bidirectional Ring AllReduce** 的带宽-延迟模型：

.. math::

   T = \alpha \times 2(N-1) + \frac{2(N-1)}{N} \times \frac{S}{B}

其中：

- :math:`N` — GPU 数量
- :math:`S` — 数据量
- :math:`B` — NVLink 带宽
- :math:`\alpha` — 每次通信的固定开销（延迟）

NVSwitch 内部 Fabric 管理
===============================

NVSwitch 内部运行专用的固件，管理端口的连接和路由。每个 NVSwitch 包含多个端口，每个端口连接到一块 GPU：

.. code-block:: text

   NVSwitch 内部架构 (H100 NVSwitch 为例):

   +----------------------------------------------------+
   |                    NVSwitch                           |
   |                                                      |
   |  +----------+  +----------+  +----------+           |
   |  | Crossbar |  | Crossbar |  | Crossbar |  ...      |
   |  |  0       |  |  1       |  |  2       |           |
   |  +-----+----+  +-----+----+  +-----+----+           |
   |        |             |             |                  |
   |  +-----+----+  +-----+----+  +-----+----+           |
   |  | SerDes  |  | SerDes  |  | SerDes  |  ...         |
   |  |       0|  |       1|  |       2|                 |
   |  +--------+  +--------+  +--------+                  |
   |                                                      |
   |  +----------------------------------------------+   |
   |  | Fabric Manager (固件)                        |   |
   |  | - 路由表管理                                  |   |
   |  | - 链路训练和错误恢复                          |   |
   |  | - SHARP 规约引擎                              |   |
   |  | - 遥测和计数器                                |   |
   |  +----------------------------------------------+   |
   +----------------------------------------------------+

**NVLink 域（NVLink Domain）**:

可在 NVSwitch 上划分多个逻辑域，每个域包含一组 GPU，域内 GPU 间全带宽通信，域间带宽受限：

.. code-block:: bash

   # 创建 NVLink 域
   nvidia-smi nvlink -d 0 -i 0 -m 0 -g 0,1,2,3   # 域 0: GPU 0-3
   nvidia-smi nvlink -d 1 -i 0 -m 1 -g 4,5,6,7   # 域 1: GPU 4-7

SHARP 编程接口
======================

NVSwitch 的 SHARP（Scalable Hierarchical Aggregation Protocol）允许在交换机内完成规约操作。NCCL 中的 NVLS 协议即基于此技术：

.. code-block:: cpp

   // NCCL 中使用 NVLS 协议（自动选择）
   ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream);

   // 强制使用 NVLS
   // export NCCL_ALGO=NVLS

   // 查看 SHARP 统计
   // export NCCL_DEBUG=INFO
   // export NCCL_DEBUG_SUBSYS=INIT,GRAPH

**SHARP 规约类型支持**:

.. list-table::
   :header-rows: 1

   * - 操作
     - 数据类型
     - 交换机内完成
   * - SUM
     - FP32, FP16, BF16, INT32
     - 是
   * - PROD
     - FP32, FP16
     - 是
   * - MAX / MIN
     - FP32, FP16, INT32
     - 是

**多节点 NVSwitch 拓扑**:

DGX H100 的 8 路 GPU 通过 4 个 NVSwitch 互联：

.. code-block:: text

                     GPU 0-7
                   /  |  |  \
                  /   |  |   \
   NVSwitch 0 --/    |  |    \-- NVSwitch 3
   NVSwitch 1 -------/  \------- NVSwitch 2

   每 GPU 连接 18 条 NVLink 到 4 个 NVSwitch
   → 任意两 GPU 之间: 3 条 NVLink (150 GB/s)

NVLink 功耗管理
=====================

NVLink 链路的功耗可动态调节：

.. code-block:: text

   L0: 全速运行（18 链路, ~150W 每 GPU）
   L1: 降低链路速率（省电约 30%）
   L2: 关闭部分链路（省电约 60%）
   L3: 完全断电（恢复需 ~1ms）

   可通过 nvidia-smi 监控链路状态:
   nvidia-smi nvlink -s

参考与拓展阅读
====================

- 深入理解 :doc:`cache_coherence` — CPU-GPU 缓存一致性
- 深入理解 :doc:`topology_awareness` — 拓扑感知的通信优化
- NVIDIA NVLink & NVSwitch (https://www.nvidia.com/en-us/data-center/nvlink/) — NVIDIA NVLink/NVSwitch 技术白皮书
- NVIDIA NCCL Documentation (https://docs.nvidia.com/deeplearning/nccl/) — NCCL 通信库文档和算法说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 多 GPU 通信优化和 P2P 访问
