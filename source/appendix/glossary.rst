========
术语表
========

.. epigraph::

   Language is the road map of a culture. It tells you where its people come from and where they are going.

   — Rita Mae Brown

.. glossary::

   SM
      流多处理器（Streaming Multiprocessor）。NVIDIA GPU 的核心计算单元，包含 CUDA 核心、共享内存、寄存器文件和调度逻辑。多个 SM 构成一个 GPC。

   CU
      计算单元（Compute Unit）。AMD GPU 中与 SM 对应的概念，包含 Vector ALU、Matrix Core 和共享内存。

   GPC
      图形处理集群（Graphics Processing Cluster）。NVIDIA GPU 中多个 SM 及固定功能单元的组合，是芯片物理布局中的高层次分组。

   TPC
      纹理处理集群（Texture Processing Cluster）。GPC 内部的中间层，通常包含 2 个 SM 和共享的纹理缓存。

   Warp
      NVIDIA GPU 中 32 个线程组成的硬件调度单位，是 SM 上的最小执行粒度。同一 warp 中的线程按 SIMT 模式执行。

   Wavefront
      AMD GPU 中 64 个线程组成的硬件调度单位，比 warp 大两倍，发散惩罚也更大。

   SIMT
      单指令多线程（Single Instruction, Multiple Threads）。GPU 的执行模型，同一指令在多个线程上以不同数据并行执行。与 SIMD 不同，每个线程拥有独立的程序计数器。

   SIMD
      单指令多数据（Single Instruction, Multiple Data）。CPU 和 GPU ALU 的执行方式，单一指令在向量数据上并行操作。GPU 在 warp/wavefront 内部实际采用 SIMD 执行。

   Tensor Core
      NVIDIA GPU 中的专用矩阵运算单元，支持混合精度（FP16/BF16/TF32/FP8）矩阵乘法累加操作。从 Volta 架构引入。

   Matrix Core
      AMD GPU 中与 Tensor Core 对应的矩阵运算单元，支持 FP16/BF16 精度。

   XMX Core
      Intel Xe GPU 中的矩阵乘法加速单元（Xe Matrix eXtension），对标 Tensor Core 和 Matrix Core。

   CUDA Core
      NVIDIA SM 中的标量计算单元，执行 FP32、INT32、FP64 等运算。每个 SM 包含数十到数百个 CUDA Core。

   Subcore
      SM 内部的分区（也称处理块），拥有独立的 warp scheduler、dispatch unit、寄存器文件和 CUDA Core 子集。Ampere 架构每个 SM 包含 4 个 subcore。

   PTX
      并行线程执行（Parallel Thread Execution）。NVIDIA 的中间指令集架构（ISA），位于 CUDA C++ 和原生机器码 SASS 之间，保证代码跨代兼容。

   SASS
      NVIDIA GPU 的原生机器码（Streaming ASSembly）。PTX 经过编译后生成的芯片原生指令，不同代架构不兼容。

   ISA
      指令集架构（Instruction Set Architecture）。处理器执行的机器指令格式，定义了寄存器、寻址模式、指令编码等。

   SPIR-V
      Standard Portable Intermediate Representation - Vulkan。Vulkan 使用的二进制中间表示格式，类似于 PTX 在 CUDA 中的角色。

   HBM
      高带宽内存（High Bandwidth Memory）。通过硅通孔（TSV）技术堆叠的 3D DRAM，GPU 的主流显存方案。当前主流为 HBM2e、HBM3。

   GDDR
      图形双倍数据率内存（Graphics Double Data Rate）。用于消费级 GPU 的显存方案，如 GDDR6、GDDR6X。

   DMA
      直接内存访问（Direct Memory Access）。由硬件控制器在无需 CPU 参与下执行的数据传输。GPU 使用 DMA 引擎进行显存拷贝。

   MMU
      内存管理单元（Memory Management Unit）。GPU 中管理虚拟地址到物理地址转换的硬件单元，通过多级页表完成地址转换。

   TLB
      转换后备缓冲器（Translation Lookaside Buffer）。MMU 中的页表缓存，加速虚拟地址到物理地址的转换。GPU 拥有多级 TLB（微 TLB、L1 TLB、L2 TLB）。

   IOMMU
      输入输出内存管理单元（Input-Output Memory Management Unit）。系统层面将 GPU/PCIe 设备的 DMA 访问映射到系统物理地址。

   PCIe
      PCI Express。GPU 与主机通信的标准高速串行总线。当前主流版本为 PCIe 4.0 (32 GB/s x16) 和 5.0 (64 GB/s x16)。

   NVLink
      NVIDIA 的 GPU 间高速互联协议，提供比 PCIe 高数倍的带宽。当前版本为 NVLink 5.0（100 GB/s 单链路）。

   NVSwitch
      NVLink 的全连接交换机，实现 GPU 间的任意拓扑互联，内置 SHARP 规约引擎。用于 DGX/HGX 系列。

   SHARP
      可扩展层次聚合协议（Scalable Hierarchical Aggregation Protocol）。NVSwitch 内嵌的规约引擎，在交换机内部完成 AllReduce 操作。

   Infinity Fabric
      AMD 的多芯片互联技术，用于 GPU 间通信和 CPU-GPU 统一内存访问。支持分布式路由，兼容 CXL 协议。

   Chiplet
      小芯片设计。将 GPU 拆分为多个较小的芯片（die）并通过高速互联封装在一起。AMD MI300 系列和 NVIDIA Grace Hopper 采用此设计。

   HWS
      硬件调度器（Hardware Scheduler）。GPU 固件中的调度单元，管理 channel 间的时间片轮转、优先级仲裁和抢占。从 Pascal 架构引入。

   Channel
      GPU 固件中的执行上下文通道。每个 CUDA context 对应一个 channel，HWS 在 channel 间以时间片轮转调度。

   TDR
      超时检测恢复（Timeout Detection and Recovery）。GPU 驱动检测到执行超时后的重置流程，防止 GPU 挂起影响系统稳定。

   SR-IOV
      单根输入输出虚拟化（Single Root I/O Virtualization）。允许单个物理 GPU 被多个虚拟机共享的技术。

   VF
      虚拟功能（Virtual Function）。SR-IOV 中的轻量级虚拟设备实例，呈现为独立 GPU 给虚拟机。

   PF
      物理功能（Physical Function）。SR-IOV 中持有完整 GPU 资源的物理设备管理者。

   MIG
      多实例 GPU（Multi-Instance GPU）。NVIDIA Ampere 架构引入的硬件级 GPU 分区技术，最多 7 个独立实例。

   MPS
      多进程服务（Multi-Process Service）。NVIDIA 的 GPU 共享方案，允许多个进程共享单一 CUDA context，实现 kernel 级并发。

   Occupancy
      占用率。SM 中活跃 warp 数量与最大支持 warp 数量的比值。高占用率有助于隐藏内存延迟，但并非唯一性能指标。

   Coalescing
      合并访问。GPU 内存控制器将同一 warp 的多个线程访问合并为少量大粒度内存事务的能力。是 GPU 性能优化最重要的原则之一。

   Sector
      扇区。GPU 内存访问的最小单元（32 字节），硬件按 sector 为单位读写显存和 L2 缓存。

   Bank Conflict
      Bank 冲突。多个线程同时访问共享内存同一 bank 中不同地址导致的串行化现象。共享内存分为 32 个 4 字节 bank。

   FMA
      融合乘加（Fused Multiply-Add）。执行 ``D = A * B + C`` 的算术运算，CUDA Core 和 Tensor Core 的基本运算单元。

   Unified Memory
      统一内存。CUDA 6+ 引入的 CPU-GPU 统一虚拟地址空间，支持自动页迁移，通过 GPU 页错误和驱动页迁移引擎实现。

   UVA
      统一虚拟地址（Unified Virtual Address）。CUDA 4+ 引入的 CPU 和 GPU 共享同一虚拟地址空间的机制，地址高位标识设备归属。

   GPGPU
      通用计算图形处理器（General-Purpose Graphics Processing Unit）。用于非图形计算的 GPU，将 GPU 的并行能力用于科学计算和 AI。

   warpSize
      CUDA 中定义 warp 大小的内置常量，始终为 32。用于编写与硬件无关的代码。

   UMD
      用户态驱动（User-Mode Driver）。运行在用户空间的 GPU 驱动层，负责 API 实现、命令缓冲构建和着色器编译。

   KMD
      内核态驱动（Kernel-Mode Driver）。运行在内核空间的 GPU 驱动层，负责硬件初始化、中断处理、MMU 管理和电源管理。

   ROCm
      Radeon Open Compute。AMD 的开源 GPU 计算平台，核心编程模型为 HIP。

   HIP
      异构接口可移植性（Heterogeneous Interface for Portability）。AMD 的 GPU 编程模型，提供与 CUDA 高度相似的 API。

   oneAPI
      Intel 主导的统一编程模型，基于 SYCL 标准，支持 Intel GPU、CPU 和 FPGA 等多平台。

   SYCL
      基于 C++ 的开放标准异构计算框架，通过 lambda 表达式和设备选择器实现跨厂商计算。

   CXL
      Compute Express Link。开放标准的缓存一致性互联协议，Infinity Fabric 与之兼容。

   GPUDirect
      NVIDIA 的直接数据传输技术族，包括 RDMA（绕过 CPU）、P2P（GPU 间直连）、Storage（GPU 直接读取存储设备）。

   RDMA
      远程直接内存访问（Remote Direct Memory Access）。允许网卡绕过 CPU 直接读写远程内存的技术。InfiniBand 和 RoCE 是其物理实现。

   RoCE
      基于融合以太网的 RDMA（RDMA over Converged Ethernet）。在标准以太网上实现 RDMA 的技术。

   GDS
      GPUDirect Storage。允许 GPU 直接读写 NVMe SSD 的 GPUDirect 子技术，无需 CPU 中转。

   NCCL
      NVIDIA 集合通信库（NVIDIA Collective Communications Library）。实现 AllReduce、AllGather 等集合通信操作。

   RCCL
      ROCm 集合通信库（ROCm Collective Communications Library）。与 NCCL API 兼容的 AMD 集合通信库。

   oneCCL
      Intel oneAPI 集合通信库（oneAPI Collective Communications Library）。Intel 平台上的集合通信实现。

   SHFL
      Shuffle 指令。CUDA warp 级的数据交换指令，允许线程直接访问其他线程的寄存器。

   Fence
      栅栏。GPU 同步的基本原语，CPU 或 GPU 在栅栏上等待操作完成信号。

   Page Fault
      页错误。GPU 访问统一内存时，目标页面不在本地显存时触发的硬件中断，由驱动负责页迁移。

   TLBs shootdown
      TLB 失效操作。页表更新后，所有 SM 的 TLB 缓存条目需要被标记为无效，确保后续页面访问使用最新的页表映射。

   Partition Camping
      分区不平衡。大量线程同时访问同一 L2 缓存分区地址时，造成该分区拥塞而其他分区空闲的现象。