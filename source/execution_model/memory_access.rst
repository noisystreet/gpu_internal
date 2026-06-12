========================
内存访问模式
========================

.. epigraph::

   We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil.

   — Donald Knuth

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

只读缓存（``__ldg`` 指令）
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

L2 缓存分区与 Partition Camping
=========================================

L2 缓存被划分为多个独立分区（Ampere 有 40 个，Hopper 有 60 个），每个分区管理一段连续的显存地址范围。当大量线程同时访问同一个 L2 分区的地址时，就会产生**分区不平衡（partition camping）** 现象——该分区成为瓶颈而其他分区空闲。

.. code-block:: text

   存在分区冲突的地址访问模式:

   线程访问: A[0], A[32], A[64], A[96], ...
       每个地址相差 32 个字 (128 字节 = 一个缓存行)
       → 所有地址映射到同一个 L2 分区
       → 该分区带宽饱和度 ~100%, 其他分区利用率 ~30%

   避免分区冲突的技巧:

   1. 向地址加上一个偏移量，使地址范围跨越多个分区
   2. 使用大页面 (2MB 而非 64KB) 改变地址到分区的映射
   3. 利用 Hopper 架构改进的 L2 分区均衡

.. code-block:: cuda

   // Partition camping 示例：跨步访问导致所有线程命中同一 L2 分区
   __global__ void partition_camped(float* data, int N) {
       int idx = threadIdx.x;
       // data[0], data[32], data[64], ... → 同一分区
       float val = data[idx * 32];
   }

Persisting L2 Cache (Hopper)
=====================================

Hopper 架构引入了 L2 缓存持久化功能，允许将部分 L2 缓存分配给特定的内存区域，确保该区域的数据不被其他流冲刷：

.. code-block:: cuda
   :linenos:

   // 持久化 L2 分区
   cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 10 * 1024 * 1024);  // 10 MB

   // 设置流的持久化窗口
   cudaStreamAttrValue attr;
   attr.accessPolicyWindow.base_ptr = d_data;
   attr.accessPolicyWindow.num_bytes = N * sizeof(float);
   attr.accessPolicyWindow.hitRatio  = 1.0;  // 100% 命中
   cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

   // 该流中的 kernel 数据会持久驻留在 L2 中
   kernel<<<grid, block, 0, stream>>>(d_data, N);

**适用场景**: Transformer 中的注意力矩阵、卷积权重、频繁重用的查找表。

缓存行填充与 Sector
============================

GPU 的缓存行大小为 **128 字节**，但读写最小单元是 **32 字节扇区（sector）**。当 warp 只需要 4 字节数据时，硬件也会读取整个 sector（32 字节），造成带宽浪费：

.. code-block:: text

   Warp 访问 4 字节/线程 → 共同覆盖 128 字节范围
   硬件操作:
   1. 检查地址范围覆盖了几个 sector
   2. 每个 sector 发送一次内存事务
   3. 合并后的内存事务数 = 覆盖的 sector 数

   Sector 命中率优化:
   - 合并访问的 warp 覆盖 128 字节 = 4 sectors = 1 个内存事务
   - 步长 8 的 warp 覆盖 1024 字节 = 32 sectors = 8 个内存事务
   - 随机访问覆盖 N 个缓存行 = N × 4 sectors

内存延迟隐藏
=================

GPU 通过线程级并行来隐藏内存访问的长延迟（200-800 周期），而不是依赖于大缓存：

.. code-block:: text

   延迟隐藏条件:
   所需活跃 warp 数 = 内存延迟 (周期) / 发射间隔 (周期)

   以 Ampere 为例:
   - 全局内存延迟: ~400 周期
   - 每 warp 发射间隔: 4 周期 (2 周期发射 + 2 周期流水线)
   - 所需最少 warp: 400 / 4 = 100 个 warp

   每 SM 最大 warp 数: 64 (Ampere)
   → 单靠 warp 不足完全隐藏延迟
   → 需要考虑缓存命中率、计算密度协作

.. code-block:: text

   计算密集型 (compute-bound):  计算时间 > 访存延迟 → 延迟不敏感
   访存密集型 (memory-bound):  访存延迟 > 计算时间 → 需要高占用率

Roofline 性能模型
======================

Roofline 模型是将计算吞吐和内存带宽统一在一个坐标系中的性能分析方法。它回答了 GPU 性能分析中最关键的问题——**当前 kernel 是被计算能力限制还是被内存带宽限制**。

**Roofline 模型的基本形式**：

.. figure:: /source/figures/roofline.svg
   :width: 100%
   :align: center
   :alt: Roofline 模型 — H100 SXM

   Roofline 模型将计算吞吐（TFLOPS）和内存带宽（TB/s）统一在同一坐标系中。散点标注了典型 Kernel 的位置。

**核心指标：算术强度（Arithmetic Intensity）**

算术强度 = 总 FLOPS / 总内存流量（Byte），单位 FLOPS/Byte。它是判断 kernel 性质的关键：

.. code-block:: text

   算术强度 > ridge point → compute-bound（计算受限）
   算术强度 < ridge point → memory-bound（访存受限）

   典型 kernel 的算术强度:
   kernel                          FLOPS   内存流量   算术强度   bound 类型
   ─────────────────────────────────────────────────────────────────
   Vector Add (1:1)              1 op      12 Byte    0.08      内存
   SAXPY (y = a*x + y)           2 ops     16 Byte    0.125     内存
   GEMM (M=N=K=4096, FP32)        ~2T ops   ~96 MB    ~20000    计算
   Convolution (3x3, FP16)        ~50M ops  ~1 MB     ~50       计算
   Softmax (N=1024)              ~5K ops   ~8 KB     ~0.6      内存
   Attention (N=4096, head=64)   ~2M ops   ~128 KB   ~15       较均衡

**使用 Nsight Compute 获取 Roofline 数据**：

.. code-block:: bash

   # Nsight Compute 的 Roofline 分析
   ncu --set full --roofline-only ./my_kernel

   # 输出内容
   # - Kernel 的算术强度 (FLOPS/Byte)
   # - 理论峰值 FLOPS
   # - 理论峰值带宽
   # - Roofline 图表位置

   # 不依赖工具的快速估算:
   # nvprof / nsys 可获取 kernel 的 duration 和 memory traffic
   # nsys profile --stats=true ./my_app
   # 计算: FLOPS ≈ (总运算量) / (duration)
   #       带宽 ≈ (总内存流量) / (duration)
   #       算术强度 = FLOPS / 带宽

**Roofline 分析的实践步骤**：

.. code-block:: text

   1. 确定 kernel 的总运算量（FLOPS）
      矩阵乘法 MxK × KxN: 2 × M × N × K FLOPS
      逐元素操作: N 个线程 × 1 FMA/线程

   2. 确定 kernel 的总内存流量（Byte）
      输入 + 输出 + 中间结果（排除 L1/L2 命中部分）

   3. 计算算术强度 = FLOPS / Byte

   4. 与 ridge point 比较
      ridge point = 峰值带宽 / 峰值 FLOPS
      H100 SXM: 2 TB/s ÷ 67 TFLOPS ≈ 30 FLOPS/Byte

   5. 判定瓶颈并优化:
      memory-bound → 优化内存访问模式（合并访问、缓存复用）
      compute-bound → 优化计算策略（Tensor Core、降低精度）

**如何缓解 memory-bound**：

.. code-block:: text

   策略                    预期收益      代价
   ────────────────────────────────────────────
   合并访问                 2-10x        代码重构
   共享内存 tile 化          2-5x         共享内存使用量增加
   L1/共享内存配置           ~20%         需要 tune
   预取 (Prefetch)           1.5-3x       增加代码复杂度
   降低精度 (FP16/BF16)      2x           精度损失
   增大 tile 尺寸            1.5-3x       寄存器压力增加
   使用向量化加载 (float4)   1.5-2x        边界处理复杂

**如何缓解 compute-bound**：

.. code-block:: text

   策略                    预期收益      代价
   ────────────────────────────────────────────
   使用 Tensor Core         5-10x       需要 FP16/BF16/TF32
   降低精度                 2-4x         精度损失
   减少寄存器溢出           1.5-3x       修改 kernel 参数
   ILP (指令级并行)         1.5-2x       展开循环、增加寄存器
   使用 CUDA Graph          2-5x         减少启动开销

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中关于内存合并访问的详细说明
- 深入理解 :doc:`../execution_model/warp_wavefront` — Warp 内存访问模式
- 深入理解 :doc:`../hardware/memory_hierarchy` — 存储层次结构
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 最佳实践指南中的内存优化章节
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — 通过微基准测试分析 Ampere 内存子系统
