========
术语表
========

.. epigraph::

   Language is the road map of a culture. It tells you where its people come from and where they are going.

   — Rita Mae Brown, 作家

.. glossary::

   SM
      流多处理器（Streaming Multiprocessor）。NVIDIA GPU 的核心计算单元，包含 CUDA 核心、共享内存、寄存器文件和调度逻辑。

   CU
      计算单元（Compute Unit）。AMD GPU 中与 SM 对应的概念。

   GPC
      图形处理集群（Graphics Processing Cluster）。NVIDIA GPU 中多个 SM 及固定功能单元的组合。

   Warp
      NVIDIA GPU 中 32 个线程组成的硬件调度单位，是 SM 上的最小执行粒度。

   Wavefront
      AMD GPU 中 64 个线程组成的硬件调度单位。

   SIMT
      单指令多线程（Single Instruction, Multiple Threads）。GPU 的执行模型，同一指令在多个线程上以不同数据并行执行。

   SIMD
      单指令多数据（Single Instruction, Multiple Data）。CPU 和 GPU ALU 的执行方式，单一指令在向量数据上并行操作。

   Tensor Core
      NVIDIA GPU 中的专用矩阵运算单元，支持混合精度矩阵乘法累加操作。

   Matrix Core
      AMD GPU 中与 Tensor Core 对应的矩阵运算单元。

   PTX
      并行线程执行（Parallel Thread Execution）。NVIDIA 的中间指令集架构（ISA），位于 CUDA C++ 和 SASS 之间。

   ISA
      指令集架构（Instruction Set Architecture）。处理器执行的机器指令格式。

   HBM
      高带宽内存（High Bandwidth Memory）。通过硅通孔（TSV）技术堆叠的 3D DRAM，GPU 的主流显存方案。

   DMA
      直接内存访问（Direct Memory Access）。由硬件控制器在无需 CPU 参与下执行的数据传输。

   MMU
      内存管理单元（Memory Management Unit）。GPU 中管理虚拟地址到物理地址转换的硬件单元。

   IOMMU
      输入输出内存管理单元（Input-Output Memory Management Unit）。系统层面将 GPU/PCIe 设备的 DMA 访问映射到系统物理地址。

   PCIe
      PCI Express。GPU 与主机通信的标准高速串行总线。

   NVLink
      NVIDIA 的 GPU 间高速互联协议，提供比 PCIe 高数倍的带宽。

   Infinity Fabric
      AMD 的多芯片互联技术，用于 GPU 间通信和 CPU-GPU 统一内存访问。

   SR-IOV
      单根输入输出虚拟化（Single Root I/O Virtualization）。允许单个物理 GPU 被多个虚拟机共享的技术。

   VF
      虚拟功能（Virtual Function）。SR-IOV 中的轻量级虚拟设备实例。

   SPIR-V
      Standard Portable Intermediate Representation - Vulkan。Vulkan 使用的二进制中间表示格式。

   SASS
      NVIDIA GPU 的原生机器码（Streaming ASSembly）。PTX 经过编译后生成的芯片原生指令。

   Occupancy
      占用率。SM 中活跃 warp 数量与最大支持 warp 数量的比值。

   Coalescing
      合并访问。GPU 内存控制器将多个线程的内存访问合并为少量大粒度事务的能力。

   Bank Conflict
      Bank 冲突。多个线程同时访问共享内存同一 bank 中不同地址导致的串行化现象。

   Unified Memory
      统一内存。CUDA 6+ 引入的 CPU-GPU 统一虚拟地址空间，支持自动页迁移。

   GPGPU
      通用计算图形处理器（General-Purpose Graphics Processing Unit）。用于非图形计算的 GPU。
