========================
Kernel 执行模型
========================

.. epigraph::

   We can only see a short distance ahead, but we can see plenty there that needs to be done.

   — Alan Turing

Kernel 是在 GPU 上执行的函数。GPU 采用 **SIMT（Single Instruction, Multiple Threads）** 执行模型，即同一条指令由多个线程在不同的数据上执行。

线程层次结构
=================

GPU 的线程组织为三级层次结构：

.. code-block:: text

   网格 (Grid)
   +--------------------------------------------------+
   |  线程块 (Thread Block) 0         线程块 1          |
   |  +----------------------------+  +--------------+  |
   |  | 线程 (0,0)  线程 (1,0) ...  |  | ...          |  |
   |  | 线程 (0,1)  线程 (1,1) ...  |  |              |  |
   |  +----------------------------+  +--------------+  |
   |  线程块 2                     线程块 3              |
   |  ...                           ...                 |
   +--------------------------------------------------+

**Grid (网格)**
    由用户指定的所有线程块组成的集合，对应一次 kernel 启动。

**Thread Block (线程块)**
    一组协同工作的线程，可驻留在同一 SM 上，通过共享内存通信，通过 ``__syncthreads()`` 同步。

**Thread (线程)**
    最小的执行单元，每个线程有独立的程序计数器、寄存器和本地内存。

**Warp**
    32 个连续线程组成的硬件调度单元，是 SM 上实际的指令执行粒度。

Kernel 启动语法
====================

.. code-block:: cuda
   :linenos:

   // CUDA kernel 启动语法
   kernel_name<<<grid_dim, block_dim, shared_mem, stream>>>(args...);

   // 参数说明:
   //   grid_dim   — 网格维度 (dim3 或整数)
   //   block_dim  — 线程块维度 (dim3 或整数)
   //   shared_mem — 动态共享内存大小 (字节, 可选)
   //   stream     — CUDA 流 (可选)

   // 示例: 一维向量加法
   dim3 grid(1024);          // 1024 个线程块
   dim3 block(256);          // 每块 256 个线程
   vec_add<<<grid, block>>>(d_a, d_b, d_c, N);

**内置变量**:

.. list-table::
   :header-rows: 1

   * - 变量
     - 类型
     - 说明
   * - ``gridDim``
     - dim3
     - 网格维度（每个维度上的块数）
   * - ``blockIdx``
     - dim3
     - 当前线程块在网格中的索引
   * - ``blockDim``
     - dim3
     - 线程块维度（每块中的线程数）
   * - ``threadIdx``
     - dim3
     - 当前线程在线程块中的索引
   * - ``warpSize``
     - int
     - warp 大小（始终为 32）

**全局线程 ID 计算**:

.. code-block:: cuda

   // 一维
   int tid = blockIdx.x * blockDim.x + threadIdx.x;

   // 二维
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int idy = blockIdx.y * blockDim.y + threadIdx.y;
   int linear_tid = idy * (gridDim.x * blockDim.x) + idx;

Kernel 的执行流程
======================

.. code-block:: text

   主机端                           GPU
   ========                        ===================
   1. 分配显存 (cudaMalloc)
   2. 拷贝数据 (cudaMemcpy H2D)
   3. 启动 Kernel (<<<>>>)
                                    4. Grid 分发到 GPC
                                    5. Thread Block 分配到 SM
                                    6. SM 将线程分组为 Warp
                                    7. Warp 调度器选择就绪 Warp
                                    8. 发射指令到执行单元
   9. 同步 (cudaDeviceSynchronize)
   10. 拷贝结果 (cudaMemcpy D2H)
   11. 释放显存 (cudaFree)

Kernel 同步机制
===================

**线程块内部同步**: ``__syncthreads()``
    保证同一线程块内所有线程在继续执行前都到达此同步点——作为内存栅栏，确保此前所有共享内存和全局内存访问对块内其他线程可见。

**设备级同步**: ``cudaDeviceSynchronize()``
    阻塞主机直到所有 pending 的 kernel 完成。

**流同步**: ``cudaStreamSynchronize()``
    等待特定流中所有操作完成。

深入同步原语
=================

**Warp 级同步: ``__syncwarp()``**

``__syncwarp()`` 是 warp 内的轻量级同步屏障，只同步同一 warp 内的 32 个线程。它比 ``__syncthreads()`` 开销低得多，因为不涉及跨 warp 的通信。

.. code-block:: cuda

   // __syncwarp 用于 warp 内数据交换
   __global__ void syncwarp_example(float* data) {
       int tid = threadIdx.x;
       int lane = tid % 32;
       float val = data[tid];

       // 不需要 __syncthreads，同 warp 天然按锁步执行
       // 但在某些场景（如发散后重收敛）需要显式同步
       __syncwarp();

       unsigned mask = __activemask();
       val += __shfl_down_sync(mask, val, 16);
       __syncwarp();
       val += __shfl_down_sync(mask, val, 8);
       __syncwarp();
       val += __shfl_down_sync(mask, val, 4);
       __syncwarp();
       val += __shfl_down_sync(mask, val, 2);
       __syncwarp();
       val += __shfl_down_sync(mask, val, 1);

       if (lane == 0) data[tid / 32] = val;
   }

**Cooperative Groups**

CUDA 9+ 引入 ``cooperative_groups`` 命名空间，提供灵活的线程分组和同步机制：

.. code-block:: cuda

   #include <cooperative_groups.h>

   namespace cg = cooperative_groups;

   __global__ void cooperative_groups_example(float* data) {
       // 获取当前线程所属的线程块
       cg::thread_block block = cg::this_thread_block();

       // 获取当前 warp
       cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

       // 将线程块分为 8 个 4 线程子组
       cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);

       // 在 tile 内做规约
       float val = data[threadIdx.x];
       for (int i = tile4.size() / 2; i > 0; i >>= 1) {
           val += tile4.shfl_down(val, i);
       }

       if (tile4.thread_rank() == 0) {
           data[block.thread_rank() / 4] = val;
       }
   }

**内存栅栏（Memory Fence）**

同步屏障的另一种形式是内存栅栏，保证内存操作的顺序可见性：

.. list-table::
   :header-rows: 1

   * - 栅栏函数
     - 可见性范围
     - 典型用途
   * - ``__threadfence()``
     - 所有线程（同一设备）
     - 全局内存的跨线程可见性
   * - ``__threadfence_block()``
     - 同一线程块内
     - 共享内存 + 全局内存
   * - ``__threadfence_system()``
     - 主机 + 所有设备
     - CPU-GPU 统一内存同步

.. code-block:: cuda

   __global__ void fence_example(int* flag, float* data) {
       int tid = threadIdx.x;

       if (tid == 0) {
           data[0] = compute_value();
           // 保证 data[0] 的写在线程 1 看到 flag 前可见
           __threadfence();
           flag[0] = 1;  // 通知其他线程
       }
       // 线程 1 等待 flag
       if (tid == 1) {
           while (flag[0] == 0);
           // __threadfence 保证此时 data[0] 已写入
           float val = data[0];
       }
   }

**Acquire/Release 语义**

从 CUDA 11.0 开始，GPU 支持 C++ 风格的 acquire/release 语义：

.. code-block:: cuda

   // 释放语义: 之前的写入在 others 观察到的后续操作前可见
   __threadfence_release();

   // 获取语义: 之后的读取在观察到当前操作之前的值后执行
   __threadfence_acquire();

   // 顺序一致性: 单一时钟次序
   __threadfence_seq_cst();

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中关于同步和内存栅栏的章节
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX 指令集手册中同步指令的完整说明
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 第 4-5 章详细讲解 Kernel 执行模型和线程同步
