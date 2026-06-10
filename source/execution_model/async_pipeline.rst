==========================
异步操作与流水线
==========================

.. epigraph::

   No man ever steps in the same river twice, for it's not the same river and he's not the same man.

   — Heraclitus

现代 GPU 支持多种异步操作机制，允许多个操作重叠执行，从而最大化硬件利用率。

操作重叠（Overlap）
======================

GPU 计算的本质是并行。但真正的挑战不在于同时运行多个 kernel，而在于让不同类型的工作——数据传输、kernel 计算、结果回传——互相掩盖彼此的延迟。

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

满足这些要求后，GPU 的 DMA 引擎和计算引擎可以同时工作，实现"一边搬运、一边计算"的理想状态。

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

CUDA Graph 与流水线结合
==============================

CUDA Graph 支持将流水线模式捕获为图，消除启动开销：

.. code-block:: cuda
   :linenos:

   // 1. 创建流水线 stream
   cudaStream_t h2d, compute, d2h;
   cudaStreamCreate(&h2d);
   cudaStreamCreate(&compute);
   cudaStreamCreate(&d2h);

   // 2. 使用流捕获构建流水线图
   cudaGraph_t graph;
   cudaStreamBeginCapture(h2d, cudaStreamCaptureModeGlobal);

   // 此处的所有操作在性能关键的循环中将被缓存
   cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, h2d);
   // 注意：需要确保依赖关系正确（h2d → compute → d2h）
   // CUDA Graph 自动推断流间依赖
   kernel<<<grid, block, 0, compute>>>(d_data);
   cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, d2h);

   cudaStreamEndCapture(h2d, &graph);

   // 3. 实例化
   cudaGraphExec_t graphExec;
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

   // 4. 在训练循环中重复启动（0 次 CPU-GPU 交互）
   for (int i = 0; i < 10000; i++) {
       cudaGraphLaunch(graphExec, h2d);  // ~3-5 us
   }

**性能对比**：

.. list-table::
   :header-rows: 1

   * - 启动方式
     - 单次迭代开销
     - 10000 次迭代总耗时
     - 适用场景
   * - 普通 Stream 提交
     - ~20-50 us
     - ~200-500 ms
     - 通用 GPU 计算
   * - CUDA Graph
     - ~3-5 us
     - ~30-50 ms
     - 定型循环、推理服务

Stream Callback
=====================

CUDA 支持在 stream 中插入回调，用于异步通知 CPU：

.. code-block:: cuda
   :linenos:

   void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* data) {
       // 此回调在 CPU 端异步执行，不阻塞 GPU
       // GPU 操作完成后自动触发
       auto* result = static_cast<float*>(data);
       printf("Kernel 完成，第一个结果: %f\n", result[0]);
   }

   kernel<<<grid, block, 0, stream>>>(d_data, N);
   cudaLaunchHostFunc(stream, callback, h_result);

**注意事项**：

- Callback 在 CPU 端执行，会占用 CPU 周期
- 不可在 callback 中调用 CUDA API（死锁风险）
- 适合轻量通知而非繁重处理
- 推荐用于日志、指标采集、触发下一阶段

Host-side 与 Device-side Launch 对比
===========================================

GPU 操作的启动方式直接影响延迟：

.. code-block:: text

   Host-side launch (CPU 提交):
   CPU 端:
   1. 构建 Kernel 参数包     (~1 us)
   2. 内核驱动 ioctl 调用     (~5-20 us)
   3. KMD 写入 PUSH 缓冲区    (~1-5 us)
   4. Doorbell 通知 GPU       (~0.1 us)
                             合计: ~20-50 us

   Device-side launch (GPU 提交):
   在 CUDA Graph 或动态并行中:
   GPU 直接在 SM 上产生新的 Kernel 启动请求
                             延迟: ~1-5 us

**动态并行（Dynamic Parallelism）** 允许 GPU kernel 内部启动子 kernel：

.. code-block:: cuda
   :linenos:

   __global__ void parent_kernel(float* data, int N, int depth) {
       if (depth > 0 && N > 256) {
           // 从 GPU 端启动子 kernel（无需 CPU 参与）
           child_kernel<<<N/256, 256>>>(data, N, depth - 1);
       }
       // GPU 调度器负责父子 kernel 的同步
   }

   // 主机只需启动根 kernel
   parent_kernel<<<grid, block>>>(d_data, N, 10);
   cudaDeviceSynchronize();  // 等待整棵树完成

.. warning::

   动态并行有较高的调度开销（~5-10 us 每次设备侧启动），
   且会增加显存压力（需要额外的 GPU 侧调度栈），
   建议仅在递归算法或动态工作负载中使用。

多引擎并发与硬件队列
============================

GPU 内部有多个独立的硬件引擎，可以并行执行不同类型的操作：

.. code-block:: text

   GPU 硬件引擎:
   +------------+  +------------+  +------------+
   | 计算引擎    |  | 拷贝引擎    |  | 图形引擎    |
   | (SM)       |  | (DMA)      |  | (Graphics) |
   +------------+  +------------+  +------------+

   每个引擎维护独立的命令队列。
   不同流的操作如果使用不同引擎，可真正并行。

   CUDA Stream 的引擎映射（Pascal+）:
   - 计算操作 (Kernel) → 计算引擎
   - 内存操作 (cudaMemcpyAsync) → 拷贝引擎
   - 因此计算和拷贝可在不同流中并行

**超售（Oversubscription）与 MPS**:

当活跃请求超过硬件引擎容量时，GPU 的硬件调度器（HWS）进行时间片复用：

.. code-block:: text

   4 个流提交 4 个 kernel，但只有 2 个计算引擎:
   Time:  |  0  1  2  3  4  5  6  7
   ───────┼───────────────────────────
   引擎 0: | K0  K0  K2  K2  ─  ─  ─
   引擎 1: | K1  K1  K3  K3  ─  ─  ─

   每 2 个 kernel 在时间片上交替执行。
   这是 MPS 实现多进程 GPU 共享的基础机制。

**最佳实践**:

.. code-block:: cuda

   // 推荐：使用独立流实现传输和计算引擎重叠
   cudaStream_t compute_stream, transfer_stream;
   cudaStreamCreate(&compute_stream);
   cudaStreamCreate(&transfer_stream);

   // 传输引擎工作
   cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, transfer_stream);

   // 计算引擎工作（与传输并行）
   kernel<<<grid, block, 0, compute_stream>>>(d_data);

   // 结果回传（与下一轮计算并行）
   cudaMemcpyAsync(h_result, d_result, size, cudaMemcpyDeviceToHost, transfer_stream);

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 流、异步拷贝和 cuda::pipeline 的完整 API 参考
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 异步操作重叠的最佳实践
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — cp.async 指令的硬件级规范
