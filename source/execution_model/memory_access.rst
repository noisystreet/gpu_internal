========================
内存访问模式
========================

.. epigraph::

   We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil.

   — Donald Knuth, 图灵奖得主、《计算机程序设计艺术》作者

GPU 的性能高度依赖于内存访问模式。理解 GPU 的内存子系统特性并编写缓存友好的访问模式是 GPU 优化的核心。

全局内存合并访问
======================

GPU 内存控制器以 **32 字节扇区（sector）** 为单位访问显存。当同一 warp 的所有线程访问连续的 32 字节对齐地址时，硬件将这些访问合并为最少数量的内存事务。

**对齐合并访问（最优）**:

.. code-block:: text

   线程: 0  1  2  3  ... 31
        |  |  |  |      |
        v  v  v  v      v
   地址: [0] [4] [8] [12] ... [124]   (每个线程 4 字节)
        \_____________________________________/
                    一个 128 字节
                    缓存行事务

**非合并访问（低效）**:

.. code-block:: text

   线程: 0  1  2  3  ... 31
        |  |  |  |      |
        v  v  v  v      v
   地址: [0] [128] [256] [384] ... [3968]   (跨度 128 字节)
        |    |     |     |          |
        32   32    32    32         32 字节扇区 x 32 = 32 个事务

**连续访问 vs 步长访问**:

.. code-block:: cuda
   :linenos:

   // 合并访问（优）
   __global__ void coalesced(float* A, float* B, int N) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < N) {
           B[idx] = A[idx] * 2.0f;  // thread i 访问 A[i], 连续
       }
   }

   // 非合并访问（劣）
   __global__ void strided(float* A, float* B, int N, int stride) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < N) {
           B[idx] = A[idx * stride] * 2.0f;  // 步长为 stride
       }
   }

共享内存 Bank 冲突
========================

共享内存被划分为 **32 个 bank**，每个 bank 宽 **4 字节**。连续的 4 字节字映射到连续的 bank。

.. code-block:: text

   Bank:  0   1   2   3  ...  31
         +---+---+---+---+     +---+
         | 0 | 1 | 2 | 3 |     |31 |  地址: 0-127
         +---+---+---+---+     +---+
         |32 |33 |34 |35 |     |63 |  地址: 128-255
         +---+---+---+---+     +---+

**无冲突**：同一 warp 的每个线程访问不同的 bank。

**无冲突（广播）**：同一 warp 的所有线程访问同一地址，硬件广播到所有线程。

**2 路 Bank 冲突**：两个线程访问同一个 bank 中的不同地址，该 bank 被串行化。

**避免 Bank 冲突的策略**:

1. **填充（Padding）**: 在共享内存声明中增加额外元素，错开对齐：

   .. code-block:: cuda

      __shared__ float shared[32][32 + 1];  // 加 1 消除列访问冲突

2. **改变访问模式**: 根据需要选择行主序或列主序访问。

3. **利用广播**: 设计访问模式使多个线程访问同一地址。

常量内存访问
=================

常量内存具有两大优势：

1. **缓存广播**：同一 warp 的线程访问同一常量地址时仅一次内存事务，结果广播给所有线程
2. **延迟较低**：常量缓存的开启延迟低于全局内存

.. code-block:: cuda
   :linenos:

   // 常量内存声明
   __constant__ float coefficients[256];

   // 主机端设置
   cudaMemcpyToSymbol(coefficients, host_coeffs, sizeof(float) * 256);

   // kernel 内使用
   __global__ void apply_coeff(float* data, int N) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < N) {
           data[idx] = data[idx] * coefficients[idx % 256];  // 广播模式
       }
   }

只读缓存（`__ldg` 指令）
=============================

NVIDIA 的 Kepler 架构开始引入了只读缓存（Read-Only Cache），通过 ``__ldg()`` 内建函数使用。对于只读且访问模式随机的情况，``__ldg`` 可绕过 L1 而使用专用的只读缓存，提升命中率。

.. code-block:: cuda

   __global__ void read_only_kernel(const float* __restrict__ input, float* output) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       output[idx] = __ldg(&input[idx]) * 2.0f;
   }

内存访问性能分析准则
=========================

.. list-table::
   :header-rows: 1

   * - 准则
     - 说明
     - 影响
   * - 合并访问
     - 同一 warp 访问连续对齐地址
     - 充分利用显存带宽
   * - 避免 Bank 冲突
     - 共享内存访问均匀分布在 32 个 bank 上
     - 避免串行化
   * - 避免 Warp 发散
     - warp 内所有线程走同一控制流路径
     - 避免串行化分支
   * - 利用广播
     - 使用常量内存或共享内存广播
     - 减少内存事务
   * - 避免非对齐访问
     - 起始地址对齐到 128 字节
     - 减少 L2 缓存行浪费
   * - 使用向量化加载
     - ``float4`` / ``int4`` 等向量类型
     - 减少指令数量，提高带宽利用率

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中关于内存合并访问的详细说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 最佳实践指南中的内存优化章节
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — 通过微基准测试分析 Ampere 内存子系统
