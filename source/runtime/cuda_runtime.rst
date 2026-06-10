========================
CUDA Runtime
========================

.. epigraph::

   The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise.

   — Edsger W. Dijkstra, 图灵奖得主

CUDA（Compute Unified Device Architecture）是 NVIDIA 推出的并行计算平台和编程模型。CUDA Runtime 是面向开发者的高级 API 层。

CUDA 软件栈
================

.. code-block:: text

   应用程序 (Application)
       |
   主流编程语言: CUDA C++ / Python (CuPy, Numba) / Fortran
       |
   CUDA Runtime API (cudart)
       |
   CUDA Driver API (cuda)
       |
   用户态驱动 (libcuda.so)
       |
   内核态驱动 (nvidia.ko)
       |
   NVIDIA GPU 硬件

**Runtime API vs Driver API**:

.. list-table::
   :header-rows: 1

   * - 特性
     - Runtime API
     - Driver API
   * - 易用性
     - 高，自动初始化和管理
     - 低，手动控制
   * - 控制粒度
     - 粗粒度
     - 细粒度（上下文、模块）
   * - 初始化
     - 隐式
     - 显式 (cuInit)
   * - 适用场景
     - 大多数应用
     - 框架、调试工具
   * - 推荐使用
     - 是
     - 运行时/框架开发

CUDA 核函数（Kernel）
========================

.. code-block:: cuda
   :linenos:

   // 向量加法 Kernel
   __global__ void vec_add(float* a, float* b, float* c, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           c[idx] = a[idx] + b[idx];
       }
   }

   // 主机代码
   int main() {
       int N = 1 << 20;
       float *d_a, *d_b, *d_c;

       // 分配显存
       cudaMalloc(&d_a, N * sizeof(float));
       cudaMalloc(&d_b, N * sizeof(float));
       cudaMalloc(&d_c, N * sizeof(float));

       // 拷贝数据到设备
       cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
       cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

       // 启动 Kernel
       int blockSize = 256;
       int gridSize = (N + blockSize - 1) / blockSize;
       vec_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

       // 同步并拷贝结果
       cudaDeviceSynchronize();
       cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

       // 清理
       cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
       return 0;
   }

内存管理 API
=================

.. list-table::
   :header-rows: 1

   * - API
     - 功能
   * - ``cudaMalloc``
     - 分配设备全局内存
   * - ``cudaMallocPitch``
     - 分配 2D 对齐内存
   * - ``cudaMalloc3D``
     - 分配 3D 对齐内存
   * - ``cudaMallocManaged``
     - 分配统一内存
   * - ``cudaMemcpy``
     - 同步数据传输
   * - ``cudaMemcpyAsync``
     - 异步数据传输
   * - ``cudaMemset``
     - 内存设置
   * - ``cudaFree``
     - 释放内存

流（Stream）与事件（Event）
===============================

**Stream** 是 GPU 操作的命令队列，同一流中的操作顺序执行，不同流的操作可以并发。

.. code-block:: cuda
   :linenos:

   // 多流并发
   cudaStream_t streams[4];
   for (int i = 0; i < 4; i++) {
       cudaStreamCreate(&streams[i]);
   }

   for (int i = 0; i < 4; i++) {
       int offset = i * N / 4;
       kernel<<<grid, block, 0, streams[i]>>>(d + offset, N / 4);
   }

**Event** 用于流间同步和性能测量：

.. code-block:: cuda

   cudaEvent_t event;
   cudaEventCreate(&event);
   cudaEventRecord(event, stream);   // Stream 完成时记录事件
   cudaStreamWaitEvent(other_stream, event, 0);  // 等待事件

动态并行（Dynamic Parallelism）
===================================

从 Kepler 架构开始，CUDA 支持在 GPU kernel 中启动子 kernel：

.. code-block:: cuda
   :linenos:

   __global__ void child_kernel(float* data, int N) {
       // ...
   }

   __global__ void parent_kernel(float* data, int N) {
       if (N > 256) {
           // 在 GPU 端启动子 kernel
           child_kernel<<<N/256, 256>>>(data, N);
       }
   }

CUDA Graph
===============

CUDA 10+ 引入了 CUDA Graph，允许预先定义 kernel 和内存操作的依赖关系图，减少运行时调度开销。

.. code-block:: cuda
   :linenos:

   cudaGraph_t graph;
   cudaGraphCreate(&graph, 0);

   cudaGraphAddKernelNode(&node, graph, NULL, 0, &node_params);

   cudaGraphExec_t instance;
   cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

   // 重复使用已实例化的图
   for (int i = 0; i < 1000; i++) {
       cudaGraphLaunch(instance, stream);
   }
   cudaDeviceSynchronize();

错误处理
============

所有 CUDA Runtime API 返回 ``cudaError_t``：

.. code-block:: cuda
   :linenos:

   #define CUDA_CHECK(call) \
       do { \
           cudaError_t err = call; \
           if (err != cudaSuccess) { \
               fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                       cudaGetErrorString(err), __FILE__, __LINE__); \
               exit(EXIT_FAILURE); \
           } \
       } while(0)

   // 使用
   CUDA_CHECK(cudaMalloc(&d_ptr, size));
   CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA Runtime API 的完整参考文档
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — CUDA 编程最佳实践
- NVIDIA Nsight Compute (https://docs.nvidia.com/nsight-compute/) — Nsight Compute 性能分析工具使用指南
