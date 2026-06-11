========================
AMD Compute Unit (CU)
========================

了解完 NVIDIA 的 CUDA Core 和 Tensor Core 之后，我们再来看 AMD 的对应设计。尽管 AMD 的 Compute Unit 在整体功能上对标 SM，但其内部架构和线程模型有显著不同。

AMD CDNA 3 架构的 CU 通过两个 SIMD 单元（也称为 Wavefront 槽位）执行指令：

.. code-block:: text

   Compute Unit (CU)
   +--------------------------------------------------+
   |  SIMD 槽 0          SIMD 槽 1                     |
   |  +----------------+ +----------------+            |
   |  | Wavefront      | | Wavefront      |            |
   |  | Scheduler      | | Scheduler      |            |
   |  |                | |                |            |
   |  | Vector ALU x16 | | Vector ALU x16 |            |
   |  | Matrix Core    | | Matrix Core    |            |
   |  +----------------+ +----------------+            |
   |                                                    |
   |  共享内存 (128 KB)                                 |
   |  L1 数据缓存                                        |
   |  标量 ALU                                           |
   +--------------------------------------------------+

设计差异
============

AMD CU 与 NVIDIA SM 的关键差异：

- **Wavefront 大小**：AMD 使用 64 线程的 wavefront 而非 32 线程的 warp，发散惩罚更大
- **SIMD 槽位**：每个 CU 包含 2 个独立 SIMD 槽，与 SM 的 4 subcore 设计不同
- **Matrix Core**：对标 Tensor Core，支持 FP16/BF16 精度
- **标量单元**：AMD CU 包含独立标量 ALU，用于地址计算和控制流

NVIDIA SM vs AMD CU 对比
==============================

.. list-table::
   :header-rows: 1

   * - 特征
     - NVIDIA SM (Ampere)
     - AMD CU (CDNA 3)
   * - 线程束大小
     - 32 线程 (warp)
     - 64 线程 (wavefront)
   * - 调度器 / SM
     - 4 个 warp 调度器
     - 2 个 wavefront 调度器
   * - FP32 核心 / 单元
     - 64
     - 32
   * - Tensor / Matrix Core
     - 16
     - 4
   * - 共享内存
     - 128 KB
     - 128 KB
   * - 寄存器文件
     - 65536
     - 约 48000

参考与拓展阅读
====================

- 深入理解 :doc:`sm_architecture` — NVIDIA SM 架构对比
- 深入理解 :doc:`tensor_core_precision` — Matrix Core 精度对比
- AMD CDNA 3 Architecture Whitepaper (https://www.amd.com/en/products/accelerators/cdna-3.html)
- AMD ROCm Documentation (https://rocm.docs.amd.com/) — HIP 编程指南
