========================
ROCm Runtime
========================

.. epigraph::

   The important thing is not to stop questioning. Curiosity has its own reason for existing.

   — Albert Einstein

ROCm（Radeon Open Compute）是 AMD 推出的开源 GPU 计算平台。HIP（Heterogeneous Interface for Portability）是 ROCm 的核心编程模型，提供与 CUDA 高度相似的 API，支持在 NVIDIA 和 AMD GPU 之间便捷移植。

ROCm 软件栈
================

.. code-block:: text

   应用程序 (Application)
       |
   HIP / OpenCL / Python (TensorFlow, PyTorch)
       |
   ROC Runtime (ROCr) + ROC Compiler (ROCclr)
       |
   ROCk Kernel Driver (amdgpu.ko)
       |
   AMD GPU 硬件 (CDNA / RDNA)

HIP 编程接口
=================

HIP 旨在作为 CUDA 的可移植替代。大多数 CUDA 概念和 API 在 HIP 中有直接对应：

.. csv-table::
   :header: CUDA API, HIP API
   :widths: auto

   ``cudaMalloc``, ``hipMalloc``
   ``cudaMemcpy``, ``hipMemcpy``
   ``cudaFree``, ``hipFree``
   Kernel 启动语法, 相同启动语法 (<<<grid, block>>>)
   ``__global__``, ``__global__``
   ``__shared__``, ``__shared__``
   ``threadIdx``, ``threadIdx`` (相同)
   ``__syncthreads``, ``__syncthreads`` (相同)

HIP Kernel 示例
====================

.. code-block:: cpp
   :linenos:

   // saxpy 在 HIP 中的实现
   #include <hip/hip_runtime.h>

   __global__ void saxpy(float a, float* x, float* y, int n) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < n) {
           y[idx] = a * x[idx] + y[idx];
       }
   }

   int main() {
       int N = 1 << 20;
       float *d_x, *d_y;

       hipMalloc(&d_x, N * sizeof(float));
       hipMalloc(&d_y, N * sizeof(float));

       hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice);
       hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice);

       int blockSize = 256;
       int gridSize = (N + blockSize - 1) / blockSize;
       saxpy<<<gridSize, blockSize>>>(2.0f, d_x, d_y, N);

       hipDeviceSynchronize();
       hipMemcpy(h_y, d_y, N * sizeof(float), hipMemcpyDeviceToHost);

       hipFree(d_x);
       hipFree(d_y);
       return 0;
   }

HIP 之下的运行时层
=======================

**ROC Runtime (ROCr)**
    ROCr 是 ROCm 的用户态运行时库，负责管理 GPU 上下文、队列调度和内存。它通过 libhsakmt（HSA Kernel Module Thunk）与内核驱动通信。

**HSA（Heterogeneous System Architecture）**
    AMD 的异构计算框架规范：

    - **HSA 队列**: 基于 AQL（Architected Queuing Language）包的命令提交
    - **hsa_signal_t**: 用于 CPU-GPU 同步的信号量
    - **hsa_amd_memory_pool_t**: 统一内存池管理

.. code-block:: c
   :linenos:

   // HSA AQL 包结构（简化）
   typedef struct {
       uint16_t header;        // 包类型、格式等
       uint16_t setup;         // 调度信息
       uint32_t kernel_object; // kernel 代码句柄
       void*    args;          // 参数指针
       void*    completion_signal; // 完成信号
   } aql_packet_t;

ROCm 编译器基础设施
========================

ROCm 使用多层编译：

.. code-block:: text

   HIP C++ 源码
       |
   clang (LLVM)
       |
   LLVM IR
       |
   AMDGPU LLVM Backend
       |
   GCN/CDNA 指令 (ISA)
       |
   硬件执行

hipcc 编译与 hipify 迁移
================================

**hipcc 编译命令**:

.. code-block:: bash

   # 编译 HIP 程序到 AMD GPU
   hipcc -o saxpy saxpy.cpp

   # 指定目标 GPU 架构
   hipcc --offload-arch=gfx942 -o saxpy saxpy.cpp   # MI300X
   hipcc --offload-arch=gfx90a -o saxpy saxpy.cpp   # MI250X

   # 调试模式
   hipcc -g -O0 -o saxpy saxpy.cpp

**hipify-perl: CUDA 到 HIP 的自动迁移**:

.. code-block:: bash

   # CUDA 源码 → HIP 源码 自动转换
   hipify-perl vector_add.cu > vector_add.hip.cpp

   # 包含 CUDA 头文件的目录
   hipify-perl --cuda-path=/usr/local/cuda vector_add.cu

   # 检查 hipify 报告
   hipify-perl --print-stats vector_add.cu

.. code-block:: text

   hipify 转换示例:
   cudaMalloc(&d_a, size)    → hipMalloc(&d_a, size)
   cudaMemcpyAsync(...)      → hipMemcpyAsync(...)
   cudaDeviceSynchronize()   → hipDeviceSynchronize()
   __syncthreads()           → __syncthreads() (保持不变)

**hipLaunchKernelGGL 语法**:

当 CUDA 的 ``<<<>>>`` 语法在 C++ 模板代码中受限时，HIP 提供了等价的函数式调用：

.. code-block:: cpp

   // CUDA 风格
   saxpy<<<gridSize, blockSize, 0, stream>>>(2.0f, d_x, d_y, N);

   // 等效的 HIP 风格
   hipLaunchKernelGGL(saxpy, gridSize, blockSize, 0, stream, 2.0f, d_x, d_y, N);

性能分析工具：rocprof
=========================

ROCm 提供 ``rocprof`` 用于 Profiling HIP 程序：

.. code-block:: bash

   # 基础性能分析
   rocprof --stats ./saxpy

   # 跟踪 kernel 和 API 调用
   rocprof --trace hipMemcpy,hipMalloc ./saxpy

   # 输出硬件计数器
   rocprof --hsa-trace --stats ./saxpy

   # 结果输出: results.csv / results.stats.csv

ROCm 与 CUDA 的差异详解
=========================

.. list-table::
   :header-rows: 1

   * - 差异项
     - CUDA
     - HIP / ROCm
   * - 厂商
     - NVIDIA 专有
     - AMD 开源
   * - 编译工具
     - nvcc
     - hipcc (基于 clang)
   * - ISA
     - PTX → SASS
     - GCN / CDNA ISA
   * - 硬件平台
     - 仅 NVIDIA
     - AMD + NVIDIA（通过 HIP）
   * - 驱动
     - 封闭
     - 完全开源

参考与拓展阅读
====================

- AMD ROCm Documentation (https://rocm.docs.amd.com/) — ROCm 完整文档和 HIP 编程指南
- AMD CDNA 3 Architecture Whitepaper (https://www.amd.com/en/products/accelerators/instinct/cdna-3.html) — AMD CDNA 3 架构白皮书
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 第 13 章介绍 HIP 和多平台 GPU 编程
