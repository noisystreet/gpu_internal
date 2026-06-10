==========================
异步操作与流水线
==========================

.. epigraph::

   No man ever steps in the same river twice, for it's not the same river and he's not the same man.

   — Heraclitus, 古希腊哲学家

现代 GPU 支持多种异步操作机制，允许多个操作重叠执行，从而最大化硬件利用率。

操作重叠（Overlap）
======================

GPU 计算中可重叠的操作类型：

.. code-block:: text

   时间 →
   ┌──────────┐
   │ 传输 H2D  │
   └──────────┘
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Kernel 0 │  │ Kernel 1 │  │ Kernel 2 │
              └──────────┘  └──────────┘  └──────────┘
              ┌──────────┐
              │ 传输 D2H  │
              └──────────┘

   H2D = Host-to-Device, D2H = Device-to-Host

**重叠要求**

1. 使用异步 API（``cudaMemcpyAsync``）
2. 传输和 kernel 位于 ``不同 CUDA 流``
3. 硬件支持重叠（Pascal+ 默认启用）

.. code-block:: cuda

   cudaStream_t stream1, stream2;
   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   // 流 1: 传输
   cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1);

   // 流 2: 计算（与传输重叠）
   kernel_before<<<grid, block, 0, stream2>>>(...);

   // 流 1: 内核计算
   kernel_main<<<grid, block, 0, stream1>>>(d_data);

   // 流 1: 结果回传
   cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, stream1);

流水线（Pipeline）
======================

CUDA 11.0+ 引入 ``cuda::pipeline``，提供共享内存的异步数据流水线机制。

**基本原理**:

.. code-block:: text

   Stage 0: 从全局内存异步加载 → 共享内存
   Stage 1: 从共享内存计算处理
   Stage 2: 异步写回 → 全局内存

   多个 stage 在流水线上重叠执行，
   用 producer-consumer 语义同步。

**流水线 API 示例**:

.. code-block:: cuda

   #include <cuda/pipeline>

   __global__ void pipeline_example(const float* input, float* output, int N) {
       // 定义流水线阶段
       __shared__ float shared_buffer[256];

       auto pipeline = cuda::make_pipeline();

       // producer: 异步加载数据到共享内存
       pipeline.producer_acquire();
       cuda::memcpy_async(shared_buffer, input + blockIdx.x * 256,
                          sizeof(float) * 256, pipeline);
       pipeline.producer_commit();

       // consumer: 从共享内存读取并计算
       pipeline.consumer_wait();
       for (int i = 0; i < 256; i++) {
           shared_buffer[i] = shared_buffer[i] * 2.0f;
       }
       pipeline.consumer_release();

       // 写回
       for (int i = threadIdx.x; i < 256; i += blockDim.x) {
           output[blockIdx.x * 256 + i] = shared_buffer[i];
       }
   }

异步拷贝（Async Copy）
============================

从 Ampere 架构开始，NVIDIA GPU 引入了硬件异步拷贝指令 ``cp.async``：

.. code-block:: cuda

   // CUDA 11+ 异步拷贝: 全局内存 → 共享内存
   __global__ void async_copy_example(const float* input, float* output) {
       __shared__ float tile[32][32];

       // 使用 cp.async 硬件指令加载数据
       // 不阻塞 warp 执行，数据由 DMA 单元搬运
       asm volatile(
           "cp.async.ca.shared.global [%0], [%1], 16;\n"
           : : "r"(&tile[threadIdx.y][threadIdx.x]),
               "l"(&input[(blockIdx.x * 32 + threadIdx.y) * N + threadIdx.x])
       );

       // 等待所有异步拷贝完成
       asm volatile("cp.async.wait_group 0;\n");

       // 确保共享内存可见
       __syncthreads();

       // 计算处理
       // ...
   }

**异步拷贝的优点**:

- 数据搬运由专用 DMA 硬件完成，不占用计算 ALU
- 搬运和计算指令可在同一个 warp 中并行发射
- 减少 ``__syncthreads()`` 等待时间

多流并发与流优先级
========================

CUDA 流支持优先级设置，高优先级流可抢占低优先级流的资源：

.. code-block:: cuda

   // 创建高/低优先级流
   int leastPriority, greatestPriority;
   cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

   cudaStream_t high_prio_stream, low_prio_stream;
   cudaStreamCreateWithPriority(&high_prio_stream, cudaStreamNonBlocking,
                                greatestPriority);
   cudaStreamCreateWithPriority(&low_prio_stream, cudaStreamNonBlocking,
                                leastPriority);

   // 高优先级 kernel 优先调度
   high_prio_kernel<<<grid, block, 0, high_prio_stream>>>(...);
   low_prio_kernel<<<grid, block, 0, low_prio_stream>>>(...);

**流并发限制**:

- PCIe 3.0/4.0 上 DMA 引擎通常为 2，双向 4 通道
- NVLink 上 DMA 引擎更多，并发度更高
- 超过硬件并发上限的操作会自动串行化

示例：完整的三流水线重叠模式
========================================

.. code-block:: cuda

   // 使用 3 个流实现传输-计算-回传流水线
   cudaStream_t h2d_stream, compute_stream, d2h_stream;
   cudaStreamCreate(&h2d_stream);
   cudaStreamCreate(&compute_stream);
   cudaStreamCreate(&d2h_stream);

   for (int i = 0; i < nChunks; i++) {
       // 当前块传输到设备
       cudaMemcpyAsync(d_chunks[i], h_chunks[i], chunkSize,
                       cudaMemcpyHostToDevice, h2d_stream);

       // kernel 计算（与下一块的传输重叠）
       kernel<<<grid, block, 0, compute_stream>>>(d_chunks[i]);

       // 结果回传（与下一块的计算重叠）
       cudaMemcpyAsync(h_results[i], d_results[i], chunkSize,
                       cudaMemcpyDeviceToHost, d2h_stream);
   }

   cudaDeviceSynchronize();

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 流、异步拷贝和 cuda::pipeline 的完整 API 参考
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 异步操作重叠的最佳实践
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — cp.async 指令的硬件级规范
