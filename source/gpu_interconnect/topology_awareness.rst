==========================
拓扑感知编程
==========================

.. epigraph::

   It is not the strongest of the species that survives, nor the most intelligent, but the one most responsive to change.

   — Charles Darwin

在多 GPU 系统中，GPU 之间的连接拓扑直接决定了通信性能和编程模型。了解并利用拓扑信息可以显著提升多 GPU 应用程序的性能。

拓扑查询
=============

CUDA 提供 API 查询设备间的拓扑关系：

.. code-block:: cuda
   :linenos:

   int nDevices;
   cudaGetDeviceCount(&nDevices);

   for (int i = 0; i < nDevices; i++) {
       for (int j = 0; j < nDevices; j++) {
           if (i == j) continue;

           // 查询 P2P 访问能力
           int canAccessPeer;
           cudaDeviceCanAccessPeer(&canAccessPeer, i, j);

           // 查询 NVLink 链路数
           int nvlinkLinks;
           cudaDeviceGetNvLinkInfo(i, &nvlinkLinks);

           // 查询 PCIe 拓扑距离
           cudaDeviceGetP2PAttribute(
               &p2pAttribute,
               cudaDevP2PAttrAccessSupported, i, j);
       }
   }

**关键拓扑属性**:

.. list-table::
   :header-rows: 1

   * - 属性
     - API
     - 说明
   * - P2P 支持
     - ``cudaDeviceCanAccessPeer``
     - 是否可直接访问对端显存
   * - NVLink 链路数
     - ``cudaDeviceGetNvLinkInfo``
     - 两 GPU 间 NVLink 连接数
   * - PCIe 距离
     - ``cudaDevP2PAttrPcieDistance``
     - PCIe 拓扑距离(跳数)
   * - 访问延迟
     - ``cudaDevP2PAttrAccessLatency``
     - P2P 访问延迟 (ns)
   * - 访问带宽
     - ``cudaDevP2PAttrNativeAtomicSupported``
     - 是否支持 P2P 原子操作

**NVIDIA 拓扑结构工具**:

.. code-block:: bash

   # 查看 GPU 拓扑
   nvidia-smi topo -m

   # 输出示例
   #       GPU0    GPU1    GPU2    GPU3
   # GPU0   X      NV4     NV4     SYS
   # GPU1  NV4      X      NV4     SYS
   # GPU2  NV4     NV4      X      SYS
   # GPU3  SYS     SYS     SYS      X

   # NV4 = 4 条 NVLink 链路
   # SYS = PCIe

**AMD ROCm 拓扑查询**:

.. code-block:: cpp

   // ROCm 中查看 GPU 可达性
   hipDeviceCanAccessPeer(&canAccess, device0, device1);

   // 通过 ROCm 工具
   // rocminfo 查看 GPU 拓扑
   // rocm-smi 查看链路信息

拓扑感知的通信优化
=========================

**策略 1: P2P 直连通信**

若 GPU 之间通过 NVLink 直连，优先使用 P2P 而非 PCIe-based 通信：

.. code-block:: cuda

   if (canAccessPeer) {
       cudaSetDevice(dev0);
       cudaDeviceEnablePeerAccess(dev1, 0);
       // 直接对 dev1 的显存读写
       kernel_p2p<<<grid, block>>>(d_peer1_ptr, data);
   } else {
       // 回退到 CPU 中转
       cudaMemcpy(h_buf, d_dev0, size, ...);
       cudaSetDevice(dev1);
       cudaMemcpy(d_dev1, h_buf, size, ...);
   }

**策略 2: 拓扑感知的 Ring AllReduce**

根据 GPU 间距离排列 AllReduce 环，让 NVLink 链路承载尽可能多的流量：

.. code-block:: text

   最优 Ring（NVLink 感知）:
   GPU0 → GPU1 → GPU2 → GPU3 → GPU0
   每步均为 NVLink 直连，带宽 ~600 GB/s (A100)

   次优 Ring（跨 PCIe）:
   GPU0 → GPU2 (PCIe) → GPU1 (NVLink) → GPU3 (PCIe)
   跨 PCIe 链路带宽 ~32 GB/s，成为瓶颈

**策略 3: NCCL 拓扑感知**

NCCL (NVIDIA Collective Communications Library) 内置拓扑感知能力：

.. code-block:: cpp

   // NCCL 自动选择最优通信路径
   ncclCommInitRank(&comm, nRanks, ncclUniqueId, rank);

   // 可通过环境变量查看 NCCL 使用的拓扑
   // export NCCL_DEBUG=INFO
   // export NCCL_DEBUG_SUBSYS=GRAPH,INIT

**策略 4: 分层通信**

常见的 GPU 互联拓扑形成清晰的层次结构：

.. code-block:: text

   节点内 (NVLink / Infinity Fabric) — 高带宽 (600-900 GB/s)
       ↓
   节点间 (NVLink Switch / Infinity Fabric) — 中带宽 (100-400 GB/s)
       ↓
   机架间 (IB / RoCE) — 低带宽 (25-200 GB/s)

编程时应在最高带宽的层级尽量聚合数据，减少低带宽层级的通信量。

常见拓扑模式与适用场景
==============================

.. list-table::
   :header-rows: 1

   * - 拓扑模式
     - 典型平台
     - 适用场景
     - 通信策略
   * - 全互联 (All-to-All)
     - DGX/HGX (NVSwitch)
     - AllReduce、AllGather
     - NCCL 自动选择
   * - 蝶形 (Dragonfly+)
     - 多机架 IB 网络
     - 大规模分布式训练
     - 分层规约
   * - 环形 (Ring)
     - PCIe 连接的 GPU
     - AllReduce、ReduceScatter
     - Ring AllReduce
   * - 树形 (Tree)
     - NCCL SHARP
     - 广播、规约
     - 交换机辅助

NCCL 通信算法深入
=========================

NCCL（NVIDIA Collective Communications Library）是 NVIDIA 的 GPU 集合通信库，实现了多种通信算法以适应不同的拓扑和场景。理解这些算法的工作原理，有助于在分布式训练和多 GPU 编程中选择最合适的通信策略。

**Ring AllReduce**

Ring AllReduce 是最广泛使用的通信算法之一。它将 N 个 GPU 连接成逻辑环，分为 scatter-reduce 和 allgather 两个阶段：

.. math::

   T_{\text{ring}} = 2 \times (N-1) \times \alpha + 2 \times \frac{N-1}{N} \times \frac{S}{B}

其中 :math:`\alpha` 是每次通信的固定延迟，:math:`S` 是数据量，:math:`B` 是带宽。

.. code-block:: text

   Ring AllReduce 示例（4 GPU, ReduceScatter 阶段）:

   第 1 步:   第 2 步:   第 3 步:
   G0→G1: C0  G0→G1: C1  G0→G1: C2
   G1→G2: C1  G1→G2: C2  G1→G2: C0
   G2→G3: C2  G2→G3: C0  G2→G3: C1
   G3→G0: C0  G3→G0: C1  G3→G0: C2

   C0/C1/C2 = 数据块。每步每个 GPU 同时发送和接收一个块。

**Tree AllReduce**

使用二叉树拓扑做规约。适用于节点间网络（延迟较高时）：

.. math::

   T_{\text{tree}} = 2 \times \log_2(N) \times \alpha + \frac{S}{B}

当 :math:`S` 很小（延迟主导）时，Tree 算法优于 Ring。

**NVLS（NVLink SHARP）**

NVSwitch 内置的 SHARP（Scalable Hierarchical Aggregation Protocol）技术，在交换机内部完成规约操作：

.. code-block:: text

   传统方式:               NVLS SHARP:
   GPU0   GPU1             GPU0   GPU1
      \   /                  |     |
      GPU0←GPU1 结果         +--+--+
                               |
                          NVSwitch
                          (内嵌规约)
                               |
                          结果分发

   NVLS 优点:
   - 规约操作在交换机中完成，GPU 仅发送/接收一次
   - 减少 GPU 参与规约的计算开销
   - 对 AllReduce 类型操作效果最显著

**算法选择策略**:

.. list-table::
   :header-rows: 1

   * - 条件
     - 推荐算法
     - 理由
   * - 同节点、NVLink 直连、大数据量
     - Ring AllReduce
     - 充分利用 NVLink 全带宽
   * - 跨节点、IB 网络、小数据量
     - Tree AllReduce
     - 减少通信步数，降低延迟影响
   * - NVSwitch 硬件、数据量大
     - NVLS
     - 交换机内规约，GPU 开销最小
   * - 混合拓扑
     - 分层（Hierarchical）
     - 节点内 Ring + 节点间 Tree

**NCCL 协议选择**:

NCCL 支持三种通信协议，可在精度和性能间权衡：

.. code-block:: bash

   # LL (Low Latency): FP16 精度，最低延迟
   # LL128 (Low Latency 128): 128 位对齐，适合小消息
   # Simple: FP32 精度，最高带宽

   # 查看 NCCL 使用的协议
   export NCCL_DEBUG=INFO
   # NCCL PROTO: Simple (大数据量默认)
   # NCCL PROTO: LL (小数据量默认)

**RCCL（ROCm Collective Communications Library）**

AMD 的集合通信库，与 NCCL API 兼容：

.. code-block:: cpp

   // RCCL API 与 NCCL 高度一致
   ncclUniqueId id;
   NCCLCHECK(ncclGetUniqueId(&id));
   NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));
   NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count,
                           ncclFloat, ncclSum, comm, stream));

参考与拓展阅读
====================

- NVIDIA NCCL Documentation (https://docs.nvidia.com/deeplearning/nccl/) — NCCL 完整文档和通信算法实现细节
- NVIDIA NVLink & NVSwitch (https://www.nvidia.com/en-us/data-center/nvlink/) — NVLink/NVSwitch 拓扑对通信的影响
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 多 GPU 通信优化章节
- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中 P2P 通信部分
