========================
GPU 驱动架构
========================

.. epigraph::

   Talk is cheap. Show me the code.

   — Linus Torvalds

现代 GPU 驱动程序采用**用户态驱动（User-Mode Driver, UMD）** 和**内核态驱动（Kernel-Mode Driver, KMD）** 的分层设计。这种分离在保证安全性的同时提供了高性能。

.. code-block:: text

   用户空间 (User Space)
   +-----------------------------------------------+
   |  应用程序                                       |
   |     |                                           |
   |     v                                           |
   |  CUDA Runtime / ROCm Runtime / Vulkan API       |
   |     |                                           |
   |     v                                           |
   |  用户态驱动 (UMD) - libcuda.so / libamdocl64.so |
   |  - API 实现                                     |
   |  - 上下文管理                                   |
   |  - 内存管理                                     |
   |  - 着色器编译                                   |
   +---------------------|-------------------------+
                          | ioctl 调用
   +---------------------|-------------------------+
   | 内核空间 (Kernel Space)                        |
   |     v                                           |
   |  内核态驱动 (KMD) - nvidia.ko / amdgpu.ko      |
   |  - 硬件访问                                     |
   |  - 中断处理                                     |
   |  - 虚拟内存管理 (IOMMU)                         |
   |  - 电源管理                                     |
   |  - MMU/页表管理                                 |
   +---------------------|-------------------------+
                          | PCIe 总线
   +---------------------|-------------------------+
   | 硬件层 (Hardware)                              |
   |  GPU 芯片                                      |
   +-----------------------------------------------+

用户态驱动 (UMD)
=====================

UMD 运行在用户空间，直接链接到应用程序中。它的主要职责：

1. **API 实现**：实现 CUDA/HIP/OpenCL/Vulkan 等 API 层
2. **上下文管理**：维护 GPU 上下文、流（stream）、事件（event）等对象
3. **内存管理**：虚拟地址空间分配、页迁移、统一内存（Unified Memory）
4. **着色器/内核编译**：将 PTX/HIP 代码编译为 GPU 机器码（SASS/ GCN 指令）
5. **命令缓冲区构建**：将 API 调用转换为 GPU 可执行的命令链

**关键数据结构 — 上下文（Context）**:

.. code-block:: c

   // CUDA 驱动的上下文结构（简化）
   struct CUContext {
       cuuint64_t    context_id;
       CUdevice      device;
       CUaddress_mode address_mode;
       // 页表指针
       struct page_table* pagetables;
       // 流列表
       struct CUStream* streams;
       // 模块/内核缓存
       struct hash_table* module_cache;
   };

内核态驱动 (KMD)
=====================

KMD 运行在内核空间，负责直接操作 GPU 硬件。它的主要职责：

1. **硬件初始化与配置**：GPU 枚举、BAR 空间映射、中断设置
2. **命令提交**：通过硬件队列将命令提交到 GPU
3. **虚拟内存管理**：操作 GPU MMU（Memory Management Unit），管理页表
4. **中断处理**：处理 GPU 中断（kernel 完成、页错误、DMA 完成）
5. **电源管理**：动态时钟频率调节、PCIe 链路状态管理
6. **错误恢复**：GPU 挂起检测、重置流程（TDR — Timeout Detection and Recovery）

**ioctl 通信**:

用户态和内核态驱动通过标准 ``ioctl`` 系统调用通信：

.. code-block:: c
   :linenos:

   // 典型 ioctl 调用路径
   // 在 UMD 中:
   int fd = open("/dev/nvidia0", O_RDWR);
   struct nv_ioctl_submit submit_cmd = {
       .cmd_buffer = cmd_buffer_dma_addr,
       .cmd_size   = size,
   };
   ioctl(fd, NV_IOCTL_SUBMIT_GPU_COMMAND, &submit_cmd);

**关键 ioctl 命令**:

.. list-table::
   :header-rows: 1

   * - ioctl 命令
     - 功能
   * - ``NV_IOCTL_SUBMIT``
     - 提交 GPU 命令
   * - ``NV_IOCTL_ALLOC_OBJ``
     - 分配 GPU 资源对象
   * - ``NV_IOCTL_MAP_MEM``
     - 将显存映射到用户地址空间
   * - ``NV_IOCTL_CTX_CREATE``
     - 创建 GPU 上下文

GPU 调度器
==============

GPU 调度器负责将提交的命令分配到 GPU 执行单元，管理硬件时间和空间资源的复用。

**硬件调度器（Hardware Scheduler, HWS）**

NVIDIA 从 Pascal 架构开始引入硬件调度器（HWS），在 GPU 固件中实现调度决策：

.. mermaid::

   flowchart TD
       UMD["用户态驱动<br/>UMD 提交命令缓冲区"] --> KMD["内核态驱动<br/>KMD 验证并写入环形缓冲区"]
       KMD --> DMA["GPU 固件<br/>Channel DMA Engine<br/>解析命令头部"]
       DMA --> HWS["Hardware Scheduler (HWS)"]
       HWS -->|Channel 管理| CH["每个 context 对应一个 channel"]
       HWS -->|时间片轮转| TS["在 active channel 间轮转"]
       HWS -->|优先级仲裁| PR["高优先级 channel 优先"]
       HWS -->|抢占| PM["可抢占正在执行的 kernel"]
       HWS --> SM["SM / CU 执行"]

       style UMD fill:#e3f2fd,color:#1a237e
       style KMD fill:#e3f2fd,color:#1a237e
       style DMA fill:#e8f5e9,color:#1b5e20
       style HWS fill:#fff3e0,color:#e65100
       style SM fill:#f3e5f5,color:#7b1fa2

**Channel 机制**:

每个 CUDA context 对应一个 GPU 侧的 **channel**。HWS 在 channel 间以时间片（~10-100 us）轮转：

.. mermaid::

   gantt
       title Channel 时间片轮转
       dateFormat X
       axisFormat %s
       section Channel 0 (CUDA App)
       Channel 0    : 0, 4
       section Channel 1 (Graphics)
       Channel 1    : 4, 4
       section Channel 2 (Video Decode)
       Channel 2    : 8, 6
       section Channel 3 (CUDA App 2)
       Channel 3    : 14, 4

**抢占级别**:

.. list-table::
   :header-rows: 1

   * - 抢占类型
     - 粒度
     - 开销
     - 实现方式
   * - 指令级抢占
     - 单条指令
     - 极低
     - 在安全指令边界暂停
   * - 线程块级抢占
     - 粗粒度
     - 高
     - 等待当前 block 结束
   * - 计算超时（TDR）
     - 进程级
     - 极高
     - GPU 重置

**TDR（Timeout Detection and Recovery）**:

当 GPU 任务超过指定时间未完成（Windows 默认 2 秒，Linux 无默认），KMD 触发 TDR 流程：

1. 检测 GPU 是否挂起（通过心跳检查）
2. 尝试重置 GPU（DRC — Dynamic Reset Circuit）
3. 若恢复失败，触发系统级 GPU 重置
4. 相关进程收到错误，驱动状态重新初始化

TDR 是驱动正确性的关键保障机制，也是 GPU 计算任务时间限制的来源（长时间运行 kernel 需切分为子任务）。

NVIDIA vs AMD 驱动架构
===========================

.. list-table::
   :header-rows: 1

   * - 特性
     - NVIDIA
     - AMD
   * - 内核驱动名称
     - ``nvidia.ko``
     - ``amdgpu.ko``
   * - 用户态驱动
     - ``libcuda.so``
     - ``libamdocl64.so`` / ``libhsa-runtime64.so``
   * - 封闭/开源
     - 封闭（部分开源：nvidia-open-kmod）
     - 完全开源（AMDGPU）
   * - 调度模型
     - 硬件调度器（可配置）
     - 软件 + 硬件调度
   * - 虚拟化支持
     - GPU 直通 (vGPU / MIG)
     - SR-IOV

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — CUDA Driver API 的完整文档
- Dissecting the Ampere GPU Architecture via Microbenchmarking (https://arxiv.org/abs/2202.00517) — 通过微基准测试分析驱动提交延迟和调度行为
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 驱动上下文管理和 TDR 相关最佳实践
