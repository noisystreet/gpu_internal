========================
Intel GPU Runtime
========================

.. epigraph::

   The only way to go fast is to go well.

   — Robert C. Martin, 软件工程师

Intel 通过 **oneAPI** 统一编程模型进入 GPU 计算市场。与 NVIDIA CUDA 和 AMD ROCm 不同，Intel 的策略是提供跨厂商的开放标准——开发者编写一次代码，可在 Intel GPU、CPU 甚至 FPGA 上运行。

Intel GPU 产品线与特性
==============================

Intel GPU 硬件涵盖了从集成显卡到独立数据中心 GPU 的产品系列：

.. list-table::
   :header-rows: 1

   * - 产品线
     - 架构
     - 用途
     - 计算能力
   * - Intel UHD / Iris Xe (集成)
     - Xe-LP
     - 笔记本/桌面图形
     - 96-128 EU
   * - Intel Arc A 系列
     - Xe HPG (Alchemist)
     - 消费级独立 GPU
     - 最多 512 XMX Core
   * - Intel Max 系列 (PVC)
     - Xe HPC (Ponte Vecchio)
     - 数据中心 HPC/AI
     - 最多 128 Xe-Core (Tile)
   * - Intel Max 系列 (Rialto Bridge)
     - Xe HPC Next
     - 下一代数据中心
     - TBA

**EU（Execution Unit）** 是 Intel GPU 的基本计算单元，类似于 NVIDIA 的 CUDA Core 或 AMD 的 Vector ALU。每个 EU 包含多个 ALU 和一个线程调度器，支持同时多线程（SIMT）。

oneAPI 与 SYCL
====================

oneAPI 是 Intel 主导的跨厂商统一计算平台，核心编程模型是 **SYCL**——基于 C++ 的开放标准，通过 lambda 表达式或函数对象在设备端执行计算。

**SYCL 的层次结构**：

.. code-block:: text

   应用程序
       |
   SYCL C++ (DPC++ — Data Parallel C++)
       |
   oneAPI 库: oneDNN, oneMKL, oneCCL 等
       |
   Intel oneAPI Level Zero (低层 API)
       |
   用户态驱动 (Intel Compute Runtime)
       |
   内核态驱动 (i915.ko / xe.ko)
       |
   Intel GPU 硬件

与 CUDA 的概念对应
=========================

.. list-table::
   :header-rows: 1

   * - CUDA 概念
     - SYCL / oneAPI 对应
     - 说明
   * - Grid / Block / Thread
     - ND-Range / Work Group / Work Item
     - 语义几乎一致
   * - kernel<<<grid, block>>>
     - parallel_for(nd_range, [=](nd_item<1> item))
     - C++ lambda 形式
   * - __syncthreads()
     - item.barrier()
     - 工作组内同步
   * - Shared Memory
     - local_accessor
     - 工作组本地内存
   * - cudaMalloc / cudaMemcpy
     - buffer / queue.submit([&](handler& h))
     - 通过队列管理
   * - CUDA Stream
     - queue (sycl::queue)
     - 命令执行队列
   * - cudaDeviceSynchronize
     - queue.wait()
     - 等待队列完成
   * - Warp / Subgroup
     - Sub-group (轮辐大小取决于硬件)
     - Intel GPU sub-group 大小 = 16 (Xe 架构)

SYCL 编程示例
====================

**向量加法 (Vector Add)**：

.. code-block:: cpp
   :linenos:

   #include <sycl/sycl.hpp>

   int main() {
       const int N = 1 << 20;
       float* a = new float[N];
       float* b = new float[N];
       float* c = new float[N];
       for (int i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 2.0f; }

       // 创建 SYCL 队列（自动选择 GPU 设备）
       sycl::queue q(sycl::gpu_selector_v);

       // 分配设备内存（通过缓冲区）
       sycl::buffer<float> buf_a(a, N);
       sycl::buffer<float> buf_b(b, N);
       sycl::buffer<float> buf_c(c, N);

       // 提交计算任务
       q.submit([&](sycl::handler& h) {
           // 获取缓冲区的访问器
           auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
           auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
           auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

           // 定义并行计算范围
           h.parallel_for(N, [=](sycl::id<1> idx) {
               acc_c[idx] = acc_a[idx] + acc_b[idx];
           });
       });

       // 等待 GPU 完成
       q.wait();

       // 验证结果
       std::cout << "c[0] = " << c[0] << "\n";
       delete[] a; delete[] b; delete[] c;
       return 0;
   }

**显式工作组与本地内存**：

.. code-block:: cpp
   :linenos:

   q.submit([&](sycl::handler& h) {
       auto acc = buf.get_access<sycl::access::mode::read_write>(h);

       // 本地内存（类似 __shared__）
       sycl::local_accessor<float, 1> shared(256, h);

       // 定义工作组大小和全局范围
       sycl::nd_range<1> nd_range(N, 256);  // global_size=256, local_size=256

       h.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
           int local_id = item.get_local_id(0);
           int global_id = item.get_global_id(0);

           // 加载到本地内存
           shared[local_id] = acc[global_id];
           item.barrier(sycl::access::fence_space::local_space);

           // 在本地内存上操作
           for (int s = 128; s > 0; s >>= 1) {
               if (local_id < s) {
                   shared[local_id] += shared[local_id + s];
               }
               item.barrier(sycl::access::fence_space::local_space);
           }

           if (local_id == 0) {
               acc[global_id] = shared[0];
           }
       });
   });

oneAPI 核心库
====================

对于大多数开发者而言，直接编写 SYCL kernel 并非最有效率的方式。Intel 提供了与 CUDA 生态一一对应的高性能库，每个库都针对 Intel GPU 做了深度优化：

.. list-table::
   :header-rows: 1

   * - 库名
     - 功能
     - CUDA 对应
   * - oneDNN
     - 深度学习原语（卷积、归一化等）
     - cuDNN
   * - oneMKL
     - BLAS、LAPACK、FFT 等数学库
     - cuBLAS / cuFFT
   * - oneCCL
     - 集合通信（AllReduce、Broadcast 等）
     - NCCL / RCCL
   * - oneDPL
     - 并行 STL 算法（sort、reduce 等）
     - Thrust
   * - oneTBB
     - CPU 任务并行
     - (无直接对应)

其中 **oneCCL (oneAPI Collective Communications Library)** 在多 GPU 通信中扮演关键角色——它负责在多个 Intel GPU 或多个节点之间高效地执行集合通信操作，类似于 NCCL 在 NVIDIA 生态中的作用：

.. code-block:: cpp

   #include <oneccl/ccl.hpp>

   ccl::init();
   auto comm = ccl::create_communicator();

   // AllReduce 操作
   std::vector<float> sendbuf(count, 1.0f);
   std::vector<float> recvbuf(count, 0.0f);
   ccl::allreduce(sendbuf.data(),
                  recvbuf.data(),
                  count,
                  ccl::datatype::float32,
                  ccl::reduction::sum,
                  comm);

Intel Level Zero API
============================

在 SYCL 和 oneAPI 库之下，Level Zero 是 Intel 的底层驱动接口，类似于 CUDA Driver API。它直接与硬件交互，常用于框架开发者或需要极致控制的应用。以下示例展示了 Level Zero 的初始化流程：

.. code-block:: cpp

   #include <level_zero/ze_api.h>

   // 初始化
   zeInit(ZE_INIT_FLAG_GPU_ONLY);

   // 枚举驱动和设备
   uint32_t driverCount = 1;
   zeDriverHandle_t driver;
   zeDriverGet(&driverCount, &driver);

   uint32_t deviceCount = 1;
   zeDeviceHandle_t device;
   zeDeviceGet(driver, &deviceCount, &device);

   // 创建上下文和命令队列
   ze_context_handle_t context;
   zeContextCreate(driver, &context, 0);

   ze_command_queue_desc_t queueDesc = {
       ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
       nullptr, 0, 0, 0,
       ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
       ZE_COMMAND_QUEUE_PRIORITY_NORMAL
   };
   ze_command_queue_handle_t queue;
   zeCommandQueueCreate(context, device, &queueDesc, &queue);

Intel GPU 的独特特性
==============================

**硬件 Sub-group 大小**：

Intel Xe GPU 的 Sub-group 大小为 16（而非 NVIDIA 的 32 或 AMD 的 64）。这意味着 warp-level 的同步和数据交换范围是 16 个线程：

.. code-block:: cpp

   // SYCL 中查询 sub-group 大小
   auto sg_size = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
   // Xe 架构: [16]
   // 如果是集成 GPU 可能为 [8]

**XMX Core（Xe Matrix eXtension）**：

Intel 的 XMX Core 类似于 NVIDIA 的 Tensor Core 和 AMD 的 Matrix Core，为矩阵乘法提供硬件加速：

.. code-block:: cpp

   // 在 SYCL 中启用 XMX（通过 oneDNN 自动调度）
   dnnl::engine engine(dnnl::engine::kind::gpu, 0);
   dnnl::stream stream(engine);

   // oneDNN 在 Intel GPU 上自动使用 XMX Core
   auto matmul_pd = dnnl::matmul::primitive_desc(engine, src_md, weights_md, dst_md);
   auto matmul_prim = dnnl::matmul(matmul_pd);
   matmul_prim.execute(stream, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wgt}, {DNNL_ARG_DST, dst}});

**统一内存 (Unified Shared Memory, USM)**：

SYCL 支持 USM，与 CUDA Unified Memory 概念等价：

.. code-block:: cpp

   // USM 分配（统一内存，CPU 和 GPU 共享指针）
   auto data = sycl::malloc_shared<float>(N, q);

   // CPU 侧访问
   for (int i = 0; i < N; i++) data[i] = i;

   // GPU 侧访问（自动迁移）
   q.parallel_for(N, [=](sycl::id<1> idx) {
       data[idx] *= 2.0f;
   }).wait();

编译与运行
====================

.. code-block:: bash

   # 安装 oneAPI 工具包 (Linux)
   # 从 https://www.intel.com/oneAPI 下载

   # 设置环境
   source /opt/intel/oneapi/setvars.sh

   # 编译 SYCL 程序
   dpcpp -o vector_add vector_add.cpp   # Intel 编译器
   # 或
   icpx -fsycl -o vector_add vector_add.cpp  # LLVM-based

   # 指定设备后端
   SYCL_DEVICE_FILTER=gpu ./vector_add    # GPU 运行
   SYCL_DEVICE_FILTER=cpu ./vector_add    # CPU 回退运行

   # 查看可用设备
   sycl-ls

Intel vs NVIDIA vs AMD 运行时对比
=========================================

.. list-table::
   :header-rows: 1

   * - 维度
     - NVIDIA CUDA
     - AMD ROCm
     - Intel oneAPI
   * - 编程模型
     - CUDA C++
     - HIP C++
     - SYCL / DPC++
   * - 开放标准
     - 专有
     - 半开源
     - 开放标准 (SYCL)
   * - 硬件支持
     - 仅 NVIDIA
     - AMD + NVIDIA (HIP)
     - Intel + 第三方后端
   * - 库生态
     - 最成熟
     - 较成熟
     - 发展中
   * - 调试工具
     - Nsight Compute
     - rocprof
     - Intel Advisor / VTune
   * - 编译器
     - nvcc
     - hipcc (clang)
     - dpcpp / icpx (LLVM)
   * - 成熟度
     - 极高
     - 中高
     - 中
   * - 主要优势
     - 生态完善、文档丰富
     - 开源可定制
     - 跨厂商、开放标准

参考与拓展阅读
====================

- Intel oneAPI Specification (https://www.oneapi.io/spec/) — oneAPI 规范和编程指南
- Intel DPC++ SYCL Compiler (https://github.com/intel/llvm) — Intel 的 LLVM SYCL 编译器
- oneDNN Documentation (https://oneapi-src.github.io/oneDNN/) — Intel 深度学习原语库
- Data Parallel C++: Mastering DPC++ for Programming Heterogeneous Systems — Reinders et al.
