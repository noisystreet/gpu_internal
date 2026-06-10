==========================
NVLink 与 NVSwitch
==========================

.. epigraph::

   The whole is greater than the sum of its parts.

   — Aristotle, 古希腊哲学家

NVLink 是 NVIDIA 开发的高带宽 GPU 间直接互联协议，旨在突破 PCIe 的带宽瓶颈。NVSwitch 是 NVLink 的全连接交换机，实现 GPU 之间的任意拓扑互联。

NVLink 演进
=================

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

.. code-block:: text

   传统 PCIe 拓扑 (PCIe Switch):
   +--------+     +--------+
   | GPU 0  |-----| GPU 1  |
   +--------+     +--------+
        |              |
   +---------+  +---------+
   | PCIe SW |  | PCIe SW |
   +---------+  +---------+
        |              |
   +-------------------------+
   |      CPU + DRAM         |
   +-------------------------+

   NVSwitch 拓扑 (DGX/HGX):
   +-------------------------------------+
   |            NVSwitch                 |
   |  +--------+  +--------+  +--------+ |
   |  | GPU 0  |  | GPU 1  |  | GPU 2  | |
   |  +--------+  +--------+  +--------+ |
   |  | GPU 3  |  | GPU 4  |  | GPU 5  | |
   |  +--------+  +--------+  +--------+ |
   |  | GPU 6  |  | GPU 7  |            |
   |  +--------+  +--------+            |
   +-------------------------------------+

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

参考与拓展阅读
====================

- NVIDIA NVLink & NVSwitch (https://www.nvidia.com/en-us/data-center/nvlink/) — NVIDIA NVLink/NVSwitch 技术白皮书
- NVIDIA NCCL Documentation (https://docs.nvidia.com/deeplearning/nccl/) — NCCL 通信库文档和算法说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 多 GPU 通信优化和 P2P 访问
