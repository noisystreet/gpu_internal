==========================
NVLink 与 NVSwitch
==========================

.. epigraph::

   The whole is greater than the sum of its parts.

   — Aristotle

NVLink 是 NVIDIA 开发的高带宽 GPU 间直接互联协议，旨在突破 PCIe 的带宽瓶颈。NVSwitch 是 NVLink 的全连接交换机，实现 GPU 之间的任意拓扑互联。在深入 NVLink 的具体规格之前，先从一个关键问题出发：为什么 PCIe 不够用？

NVLink 代际演进
====================

.. figure:: /source/figures/nvlink_evolution.svg
   :width: 85%
   :align: center
   :alt: NVLink 代际演进

   NVLink 从 1.0 到 5.0 的带宽演进。左轴为单链路带宽，右轴为 8 GPU 聚合双向带宽。

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

:doc:`gpudirect` 深入讲解了 GPUDirect RDMA、Storage、P2P 和 PeerSync 技术。

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
