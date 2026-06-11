======================
GPU 全栈层次概览
======================

.. epigraph::

   Science is what we understand well enough to explain to a computer. Art is everything else we do.

   — Donald Knuth

一次 GPU 计算操作从应用程序到硬件执行，经过多个软件和硬件层次。理解这些层次有助于定位性能瓶颈和理解系统行为。

完整调用栈
================

.. code-block:: text

   层次 1: 应用程序层
   +-----------------------------------------------+
   | AI 框架 (TensorFlow/PyTorch)                   |
   | 科学计算 (GROMACS/NAMD)                        |
   | 图形渲染 (Unreal/Blender)                      |
   +---------------------|-------------------------+
                         | API 调用
   层次 2: 编程模型层
   +---------------------|-------------------------+
   | CUDA Runtime / HIP / Vulkan / OpenCL          |
   | 库: cuBLAS / cuDNN / rocBLAS / MIOpen        |
   +---------------------|-------------------------+
                         | 标准 API
   层次 3: 运行时层
   +---------------------|-------------------------+
   | CUDA Driver API / ROCr / Vulkan Loader        |
   | 内存管理、上下文管理、流管理                   |
   +---------------------|-------------------------+
                         | ioctl
   层次 4: 用户态驱动 (UMD)
   +---------------------|-------------------------+
   | libcuda.so / libamdocl64.so / libvulkan.so    |
   | 命令缓冲构建、着色器编译、内存映射             |
   +---------------------|-------------------------+
                         | ioctl
   层次 5: 内核态驱动 (KMD)
   +---------------------|-------------------------+
   | nvidia.ko / amdgpu.ko / i915.ko              |
   | 硬件初始化、中断处理、MMU 管理、电源管理       |
   +---------------------|-------------------------+
                         | PCIe / MMIO / DMA
   层次 6: 固件层
   +---------------------|-------------------------+
   | GPU 微控制器固件                               |
   | 命令队列调度、电源门控、错误恢复               |
   +---------------------|-------------------------+
                         | 硬件指令
   层次 7: 硬件层
   +---------------------|-------------------------+
   | SM / CU, HBM, NVLink, Tensor Core, RT Core    |
   | 实际执行计算指令的内存加载/存储                |
   +-----------------------------------------------+

各层次关键职能
======================

**应用程序层**
    以 AI 框架或科学计算软件的形式存在，调用 GPU 计算库完成计算任务。

**编程模型层**
    提供 GPU 编程抽象。CUDA 是事实标准，HIP 提供跨厂商兼容性，Vulkan 提供底层控制。

**运行时层**
    管理 GPU 上下文、流（stream）、事件（event）、内存分配等资源。CUDA Runtime 封装了 Driver API 的复杂性。

**用户态驱动 (UMD)**
    运行在用户空间，负责将 API 调用转换为 GPU 可执行的命令缓冲区。包含着色器编译器（PTX → SASS 或 LLVM IR → GCN）。

**内核态驱动 (KMD)**
    运行在内核空间，通过 ``ioctl`` 接口与 UMD 通信。负责硬件资源管理、中断处理、虚拟内存管理等核心底层操作。

**固件层**
    运行在 GPU 芯片内的微控制器上，管理命令队列调度、电源状态切换、错误恢复等实时性要求高的任务。

**硬件层**
    GPU 芯片的实际物理实现，包括计算单元（SM/CU）、内存控制器、互联总线等。

性能视角的分层
======================

每个层次都可能成为性能瓶颈，性能分析工具针对不同层次提供了相应的观测手段：

.. list-table::
   :header-rows: 1

   * - 层次
     - 常见瓶颈
     - 分析工具
   * - 应用层
     - 框架调度开销、数据预处理
     - PyTorch Profiler, TensorBoard
   * - 运行时层
     - 内存分配开销、流同步
     - Nsight Systems
   * - 用户态驱动
     - 命令缓冲提交、着色器编译
     - Nsight Systems (API trace)
   * - 内核态驱动
     - ioctl 延迟、上下文切换
     - nvidia-smi, strace
   * - 硬件层
     - 计算吞吐、内存带宽、warp 发散
     - Nsight Compute, rocprof

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程模型中关于软件栈的描述
- AMD ROCm Documentation (https://rocm.docs.amd.com/) — ROCm 软件栈层次结构
- Vulkan 1.3 Specification (https://registry.khronos.org/vulkan/specs/1.3/html/) — Vulkan 分层架构的完整规范
