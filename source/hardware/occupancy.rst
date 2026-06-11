========================
占用率（Occupancy）
========================

占用率指每个 SM 中活跃 warp 数量与最大 warp 数量的比值。高占用率有助于隐藏内存延迟，但并非唯一性能指标。

影响占用率的因素
===================

1. **每线程寄存器数量** — 寄存器越多，可驻留线程越少
2. **共享内存使用量** — 每个线程块使用的共享内存越多，SM 中并行线程块越少
3. **线程块大小** — 过小或过大的块大小都会限制占用率

注意，占用率并非越高越好。有时通过降低占用率（提高每线程寄存器数）可以获得更高的单线程性能，最终提升总吞吐。这体现了 GPU 编程中典型的"硬件利用率 vs 单线程效率"的权衡。

.. code-block:: cuda
   :linenos:

   // 使用占用率 API 计算最大理论占用率
   cudaOccupancyMaxActiveBlocksPerMultiprocessor(
       &numBlocks,        // 输出：每 SM 活跃块数
       my_kernel,         // kernel 函数
       blockSize,         // 线程块大小
       sharedMemPerBlock  // 每块共享内存 (bytes)
   );

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 占用率计算器官方说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 占用率分析和寄存器优化
