========================
内存层次结构
========================

.. epigraph::

   For a successful technology, reality must take precedence over public relations, for nature cannot be fooled.

   — Richard Feynman

GPU 的内存层次结构是理解和优化 GPU 程序性能的关键。与 CPU 类似，GPU 也采用多级存储层次，但容量和带宽的取舍更加极端。

.. mermaid::

   flowchart LR
       subgraph REG["寄存器 (Register)"]
           R["~256 个/线程<br/>每个线程私有"]
       end
       subgraph SMEM["共享内存 / L1 缓存"]
           S["~48-128 KB 可配置<br/>每个 SM 私有"]
       end
       subgraph L2["L2 缓存"]
           L["~4-40 MB<br/>芯片全局共享"]
       end
       subgraph HBM["全局显存 (HBM / GDDR)"]
           G["~16-80 GB<br/>带宽 ~2 TB/s"]
       end
       subgraph CPU_DRAM["主机内存 (CPU DRAM)"]
           C["~64-512 GB<br/>带宽 ~50 GB/s<br/>PCIe 4.0 x16"]
       end

       REG -->|"容量↑ 速度↓"| SMEM
       SMEM -->|"容量↑ 速度↓"| L2
       L2 -->|"容量↑ 速度↓"| HBM
       HBM -->|"容量↑ 速度↓"| CPU_DRAM

       style REG fill:#f3e5f5,color:#7b1fa2
       style SMEM fill:#e8eaf6,color:#283593
       style L2 fill:#e8f5e9,color:#1b5e20
       style HBM fill:#fff3e0,color:#e65100
       style CPU_DRAM fill:#fce4ec,color:#b71c1c

.. note::

   这提醒我们一个重要的优化原则：距离 ALU 越近的存储器速度越快、容量越小。将频繁使用的数据放在共享内存或寄存器中，可以比全局内存访问快一到两个数量级。

全局显存（Global Memory）
=============================

全局显存是 GPU 上最大的存储空间，也是主机和设备之间数据传输的主要对象。它是理解 GPU 带宽瓶颈的起点。如果把 GPU 芯片比作一座繁忙的城市，全局显存就是城市边缘的大型仓库——容量大但距离远，每次取货需要经历漫长的运输。

- **容量**: 通常 4-80 GB
- **带宽**: HBM2e 可达 ~2 TB/s，GDDR6X 可达 ~1 TB/s
- **延迟**: ~200-800 个时钟周期
- **特点**: 所有线程均可访问，读写颗粒度为 32 字节（sector）

**内存合并访问（Memory Coalescing）**
    当同一 warp 的多个线程访问连续地址时，硬件将这些访问合并为少量大粒度内存事务，从而充分利用显存带宽。这是 GPU 性能优化最重要的原则之一。

L2 缓存
============

- 所有 SM 共享
- 容量: 4 MB (Turing) 到 40 MB (Hopper)
- 缓存行大小: 32 字节（sector）/ 128 字节（cache line）
- 分区设计，不同地址范围映射到不同 L2 分区

L1 缓存 / 共享内存
========================

在 NVIDIA GPU 中，L1 缓存和共享内存共享同一块片上 SRAM，可通过配置分配比例。

.. list-table::
   :header-rows: 1

   * - 架构
     - L1/Shared Memory 总和
     - 可配置比例
   * - Turing
     - 96 KB
     - 32/64 或 48/48
   * - Ampere
     - 128 KB
     - 可动态划分
   * - Hopper
     - 256 KB
     - 支持动态分配

共享内存（Shared Memory）
=============================

共享内存是片上 SRAM，延迟极低（~30 个周期），由同一个线程块（Thread Block）内的所有线程共享。

- **用途**: 线程块内数据复用、规约（reduction）、协作计算
- **Bank 冲突（Bank Conflict）**: 共享内存被分为 32 个 bank（4 字节宽）。当同一 warp 的多个线程访问同一 bank 中不同的地址时，发生 bank 冲突，访问被串行化。

**共享内存使用示例**:

.. code-block:: cuda
   :linenos:

   __global__ void shared_memory_example(const float* input, float* output, int N) {
       __shared__ float tile[256];

       int tid = threadIdx.x;
       int idx = blockIdx.x * blockDim.x + tid;

       if (idx < N) {
           tile[tid] = input[idx];
       }
       __syncthreads();

       // 规约操作
       for (int s = 128; s > 0; s >>= 1) {
           if (tid < s) {
               tile[tid] += tile[tid + s];
           }
           __syncthreads();
       }

       if (tid == 0) {
           output[blockIdx.x] = tile[0];
       }
   }

寄存器文件（Register File）
===============================

- 每个 SM 拥有大量寄存器（如 65536 个，Ampere）
- 寄存器在硬件线程之间静态分配
- 每个线程可用的寄存器数量由编译器决定，受 ``--maxrregcount`` 限制
- 过度使用寄存器导致占用率（occupancy）下降

常量内存（Constant Memory）
===============================

- 只读，驻留在显存，但通过常量缓存访问
- 缓存广播：同一 warp 的线程访问同一地址时，仅一次内存事务
- 容量限制: 64 KB
- 适用: 查找表、系数数组

纹理内存（Texture Memory）
===============================

- 通过纹理缓存访问的只读内存
- 支持硬件插值和边界处理
- 适合 2D 空间局部性访问模式

GPU 虚拟地址与 MMU
===============================

GPU 使用虚拟地址（Virtual Address, VA）访问显存和系统内存，通过硬件 **MMU（Memory Management Unit）** 完成地址转换。虚拟地址机制为统一内存、内存隔离和灵活的内存管理提供了基础。

GPU 虚拟地址空间布局
-------------------------

GPU 的虚拟地址空间分为多个区域，每个区域对应不同的物理目标和访问权限：

.. code-block:: text

   虚拟地址空间 (64-bit, 48-bit 实际使用):
   +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
   |  0x0000_0000_0000                         |
   |  +--------------------------------------+ |
   |  | 用户 VA 空间                          | |  ← cudaMalloc / cudaMallocManaged
   |  | (每个进程独立)                        | |     分配的区域
   |  |                                       | |
   |  |  - 显存映射 (device memory)           | |
   |  |  - 统一内存 (managed memory)          | |
   |  |  - 固定内存映射 (pinned memory)       | |
   |  +--------------------------------------+ |
   |  |                                       | |
   |  |  ... 空洞 / guard pages ...           | |
   |  |                                       | |
   |  +--------------------------------------+ |
   |  | 内核 VA 空间                          | |  ← KMD 使用
   |  | (KMD 管理，用户不可访问)              | |
   |  |  - BAR 映射                           | |
   |  |  - 固件区域                           | |
   |  |  - 寄存器 MMIO 空间                   | |
   |  +--------------------------------------+ |
   |  0xFFFF_FFFF_FFFF                         |
   +~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

**显存映射**: 通过 ``cudaMalloc`` 分配的设备内存，映射到 GPU 虚拟地址空间中的连续区域。每个 ``cudaMalloc`` 调用返回一个虚拟地址，MMU 将其转换为物理显存地址。

**统一内存（Unified Memory, UM）**: 虚拟地址可以动态指向 CPU DRAM 或 GPU 显存，根据访问位置由驱动自动迁移。

**固定内存映射**: 主机端 ``cudaHostRegister`` 锁定的内存，通过 GPU MMU 的 IOMMU 支持可直接被 GPU DMA 访问。

GPU 页表结构
-----------------

GPU 使用多级页表进行地址转换，与 CPU 的 x86-64 页表原理类似：

.. code-block:: text

                               虚拟地址 (48-bit)
     +------------------------------------------------------+
     |  PGD 索引  |  PUD 索引  |  PMD 索引  |  PTE 索引  | 偏移 |
     |  (9-bit)   |  (9-bit)   |  (9-bit)   |  (9-bit)   |(12-bit)|
     +------------------------------------------------------+
           |           |           |           |
           v           v           v           v
     +--------+  +--------+  +--------+  +--------+  +---------+
     | PGD    |→| PUD    |→| PMD    |→| PTE    |→| 物理页   |
     +--------+  +--------+  +--------+  +--------+  +---------+
     每级页表: 512 个条目, 每条目 8 字节 = 4 KB/页

.. list-table::
   :header-rows: 1

   * - 页大小
     - 页表级数
     - 映射范围 / 页
     - 典型用途
   * - 4 KB
     - 4 级 (PGD→PUD→PMD→PTE)
     - 4 KB
     - 小对象、页错误粒度
   * - 64 KB
     - 3 级 (跳过 PTE)
     - 64 KB
     - CUDA 默认页大小
   * - 2 MB
     - 2 级 (跳过 PTE+PMD)
     - 2 MB
     - 大页分配、性能优化
   * - 1 GB
     - 1 级 (跳过 PTE+PMD+PUD)
     - 1 GB
     - BAR 映射、显存窗口

**GPU 页表与 CPU 页表的区别**:

- GPU 页表由 GPU MMU 硬件遍历，不占用 SM 计算资源
- 页表更新由 KMD 通过 MMIO 或专属命令完成
- GPU 支持更大的页（64 KB 起），减少 TLB miss
- 支持 SMMU（System MMU）与 IOMMU 协同

GPU TLB 层次
-----------------

GPU 的 TLB 也采用多级缓存设计，以减少地址转换延迟：

.. code-block:: text

   GPU MMU
   +----------------------------------------------------------+
   |  L2 TLB (全局)                                            |
   |  所有 SM 共享，~2K-8K 条目                                  |
   |  命中延迟: ~30-50 周期                                     |
   +----------------------------------------------------------+
               |                    |                    |
      +--------+--------+  +--------+--------+  +--------+--------+
      | SM 0 L1 TLB      |  | SM 1 L1 TLB      |  | SM 2 L1 TLB      |
      | ~128-256 条目    |  | ~128-256 条目    |  | ~128-256 条目    |
      | 命中延迟: ~5 周期 |  | 命中延迟: ~5 周期 |  | 命中延迟: ~5 周期 |
      +------------------+  +------------------+  +------------------+
               |                       |                       |
      +--------+--------+    +--------+--------+    +--------+--------+
      | SM 0 微 TLB     |    | SM 1 微 TLB     |    | SM 2 微 TLB     |
      | ~32-64 条目     |    | ~32-64 条目     |    | ~32-64 条目     |
      | 命中延迟: ~1 周期|    | 命中延迟: ~1 周期|    | 命中延迟: ~1 周期|
      +------------------+    +------------------+    +------------------+

**TLB 命中延迟对比**:

.. list-table::
   :header-rows: 1

   * - 级别
     - 延迟
     - 命中率（典型工作负载）
   * - 微 TLB (L0)
     - ~1 周期
     - ~60-80%
   * - L1 TLB
     - ~5 周期
     - ~85-95%
   * - L2 TLB (全局)
     - ~30-50 周期
     - ~95-99%
   * - 页表遍历 (page walk)
     - ~100-500 周期
     - 缺失时触发

统一虚拟地址（Unified Virtual Address, UVA）
------------------------------------------------------

UVA 是 CUDA 4+ 引入的重要特性，使 CPU 和 GPU 共享同一虚拟地址空间：

.. code-block:: text

   CPU 视角:                         GPU 视角:
   +---------------------+          +---------------------+
   | CPU VA 空间          |          | GPU VA 空间          |
   |                     |          |                     |
   |  0x7f... (host ptr)  |          |  0x7f... (host ptr)  |
   |  +--------------+   |          |  +--------------+   |
   |  | 固定内存      |   | ← 互通 →  | 固定内存      |   |
   |  +--------------+   |          |  +--------------+   |
   |                     |          |                     |
   |  0x7f... (dev ptr)   |          |  0x7f... (dev ptr)   |
   |  +--------------+   |          |  +--------------+   |
   |  | 显存窗口      |   | ← 互通 →  | 显存          |   |
   |  +--------------+   |          |  +--------------+   |
   +---------------------+          +---------------------+

**UVA 的核心机制**:

1. CPU 和 GPU 使用同一 64 位虚拟地址编码
2. 地址高位（bit 47-63）标识该地址属于哪个设备（CPU 或某 GPU）
3. ``cudaMemcpy`` 不再需要指定方向（HostToDevice / DeviceToHost），库通过地址推断
4. ``cudaHostAlloc`` 分配的固定内存，在 CPU 和 GPU 侧可见相同的虚拟地址

.. code-block:: cuda

   // UVA 简化了数据传输
   float* d_data;
   cudaMalloc(&d_data, N * sizeof(float));

   // 无需指定方向 — UVA 可从地址推断
   cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyDefault);

**地址空间标识**:

.. code-block:: text

   Bit 63-62: 设备标识符
   0b00 = CPU 内存
   0b01 = GPU 0
   0b10 = GPU 1
   0b11 = 系统保留

虚拟地址对编程的影响
-------------------------

.. list-table::
   :header-rows: 1

   * - 特性
     - 依赖的虚拟地址机制
     - 性能影响
   * - 统一内存
     - 页迁移 + 页表更新
     - 缺页延迟 ~1-50 us
   * - MIG 隔离
     - 每个 MIG 实例独立页表
     - 无性能开销（硬件隔离）
   * - 内存池 (mem pool)
     - 子地址空间分配
     - 减少碎片化
   * - 大页 (2 MB)
     - 减少页表级数
     - 降低 TLB miss
   * - Peer access
     - 跨设备 VA 映射
     - NVLink 直连延迟

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA 编程指南中的内存层次结构说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 内存优化最佳实践
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — Ampere 内存子系统的微基准测试分析
- Understanding GPU Page Migration - IISWC 2022 — GPU 页迁移机制的深入研究
- Parallel Thread Execution ISA (https://docs.nvidia.com/cuda/parallel-thread-execution/) — PTX 指令集手册中关于内存访问指令的规范
