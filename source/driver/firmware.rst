=========================
GPU 固件（Firmware）
=========================

.. epigraph::

   Any sufficiently advanced technology is indistinguishable from magic.

   — Arthur C. Clarke

在 UMD 和 KMD 之下，GPU 芯片内部运行着一套不为人所见的软件栈——GPU 固件（firmware）。固件运行在 GPU 芯片内置的微控制器上，负责最底层的硬件管理和调度。理解固件有助于把握驱动-硬件边界的真实位置。

GPU 固件的微控制器架构
==============================

GPU 芯片上集成多个专用微控制器，各自负责特定功能域：

.. list-table::
   :header-rows: 1

   * - 微控制器
     - 厂商
     - 职责
     - 时钟频率
   * - PMU（Power Management Unit）
     - NVIDIA
     - 电源管理、时钟频率控制、温度监测
     - ~100-200 MHz
   * - SEC（Security Engine）
     - NVIDIA
     - 固件签名验证、密钥管理、安全启动
     - ~100 MHz
   * - FLS（Falcon — Fast Logic Engine）
     - NVIDIA
     - 命令解析、Channel DMA、上下文切换、调度
     - ~200-400 MHz
   * - SMU（System Management Unit）
     - AMD
     - 电源管理、时钟频率控制
     - ~100-200 MHz
   * - PSP（Platform Security Processor）
     - AMD
     - 安全启动、固件验证
     - ~100 MHz
   * - GSP（GPU System Processor）
     - AMD（RDNA3+）
     - 调度、上下文管理（替代驱动部分职责）
     - ~400 MHz

**Falcon（FLS）微控制器** 是 NVIDIA GPU 固件的核心。Falcon 是一个专门为实时控制设计的小型 RISC 核心，运行自定义的固件代码：

.. code-block:: text

   Falcon 微控制器架构:
   +---------------------------------------------+
   |  RISC 核心                                     |
   |  - 独立指令缓存 (I-cache)                    |
   |  - 独立数据缓存 (D-cache)                    |
   |  - 硬件乘法器                                 |
   |  - 中断控制器                                 |
   |                                               |
   |  专用 SRAM (DMEM ~64-256 KB)                  |
   |  专用 SRAM (IMEM ~32-128 KB)                  |
   |                                               |
   |  DMA 引擎 — 直接读写 GPU 显存和寄存器         |
   |                                               |
   |  消息传递单元 — 与 KMD 通信的邮箱寄存器         |
   +---------------------------------------------+

NVIDIA 从 Maxwell 架构开始将原本由 KMD 负责的调度决策（时间片轮转、优先级仲裁）移入 Falcon 上运行的 HWS 固件。这使得调度不受 CPU 驱动延迟的影响，也让 NVIDIA 可以在不修改内核驱动的情况下更新调度策略。

固件的职责
================

.. mermaid::

   flowchart TB
       subgraph FW["GPU 固件 (Falcon/PMU)"]
           HS["HWS<br/>硬件调度器"]
           PM["PMU<br/>电源管理"]
           SP["SEC/PSP<br/>安全引擎"]
           ER["错误恢复<br/>TDR 处理"]
       end

       subgraph KMD["内核态驱动 (KMD)"]
           IOCTL["ioctl 接口"]
           MEM["MMU/页表管理"]
       end

       subgraph HW["GPU 硬件"]
           SM["SM/CU 核心"]
           MEMC["显存控制器"]
           PCIE["PCIe 接口"]
       end

       KMD -->|"固件加载<br/>启动参数传递"| FW
       FW -->|"调度决策"| SM
       FW -->|"P-state 切换"| MEMC
       FW -->|"链路管理"| PCIE
       FW -->|"中断上报"| KMD

       style FW fill:#fff3e0,color:#e65100
       style KMD fill:#e3f2fd,color:#1565c0
       style HW fill:#f5f5f5,color:#1a1a1a

**1. 硬件调度（HWS）**

从 Pascal 架构开始，时间片轮转和优先级仲裁不再由 KMD 直接管理，而是由 Falcon 上运行的 **HWS 固件** 完成：

.. code-block:: text

   HWS 固件的调度循环:
   while (1) {
       channel = select_next_channel(优先级队列);
       run_channel(channel, timeslice_us);
       channel->consumed += timeslice_us;
       if (channel->consumed > channel->budget) {
           preempt_channel(channel);
       }
   }

   调度决策完全在 GPU 内部完成，与 CPU 驱动异步。
   KMD 仅通过 MMIO 写入参数（时间片长度、优先级等）。

**2. 电源管理（PMU）**

PMU 固件实时监测 GPU 的温度、电流和功耗，动态调整时钟频率：

.. code-block:: text

   PMU 控制循环 (~1 ms 周期):
   1. 读取温度传感器（hotspot、memory、VRM）
   2. 读取电流传感器
   3. 计算功耗预算
   4. 选择 P-state（频率/电压组合）
   5. 应用新频率（通过 Voltage-Frequency 曲线）
   6. 上报遥测数据到 KMD

   P-state 层级:
   P0: 最高性能（训练/推理）
   P2: 中等性能（视频编码/常规计算）
   P8: 低功耗（空闲）
   P10/12: 深度睡眠（几乎零功耗）

**3. 固件加载流程**

GPU 固件的加载在系统启动和驱动加载过程中分阶段完成：

.. mermaid::

   sequenceDiagram
       participant BIOS as 系统 BIOS
       participant VBIOS as GPU VBIOS
       participant KMD as 内核态驱动
       participant FW as GPU 固件
       participant HW as GPU 硬件

       BIOS->>VBIOS: 枚举 PCIe 设备
       VBIOS->>VBIOS: Option ROM 执行
       VBIOS->>HWS: 加载基础固件 (bootloader)
       HWS->>HWS: 自检、初始化时钟
       
       Note over VBIOS,KMD: 内核加载阶段
       KMD->>VBIOS: 读取 VBIOS 固件数据
       KMD->>HWS: 加载主固件镜像
       Note over HWS: 固件签名验证 (SEC)
       HWS->>HWS: 初始化调度器、电源管理
       HWS->>KMD: 就绪通知
       
       KMD->>KMD: 初始化驱动栈
       KMD->>HW: 开始提交命令

**4. 固件升级**

固件可以通过驱动更新，无需更新 VBIOS：

.. code-block:: bash

   # NVIDIA 固件更新
   nvidia-smi -firmware-update ./gpu_firmware.bin

   # AMD 固件更新
   amdgpu-firmware-update /lib/firmware/amdgpu/navi31_smc.bin

   # 查看当前固件版本
   nvidia-smi --query-gpu=vbios_version,firmware_version --format=csv

.. note::

   固件升级的不可逆性：VBIOS 中存储的 bootloader 仅支持已验证签名的固件镜像，
   一种少见的升级中断可能使 GPU 变砖。数据中心 GPU 通常支持双镜像备份以减少风险。

固件驱动的边界演进
========================

近年来，GPU 固件的职责边界在不断扩展：

.. list-table::
   :header-rows: 1

   * - 架构
     - 新固件功能
     - 移除的 KMD 职责
   * - Maxwell (GM200)
     - 基础 HWS
     - 调度决策从 KMD 移入固件
   * - Volta (GV100)
     - SEC 安全固件
     - 安全启动从 KMD 移入固件
   * - Ampere (GA100)
     - MIG 管理固件
     - MIG 分区管理从 KMD 移入固件
   * - Hopper (GH100)
     - Transformer Engine 控制
     - DPX 指令调度由 HWS 管理
   * - RDNA3 (AMD)
     - GSP 系统处理器
     - 调度和上下文管理从 KMD 移入 GSP

这个趋势的本质是：**GPU 朝着更自给自足的方向演进**——固件接管更多实时管理职责，KMD 逐渐退化为控制面的 API 映射和数据传输层。对于开发者而言，这意味着 GPU 的调度行为越来越"黑盒"，但也更加稳定和可预测。

参考与拓展阅读
====================

- NVIDIA GPU Firmware Architecture — 基于 Falcon 微控制器的 HWS 和 PMU 固件设计
- AMD GPU System Processor (GSP) — RDNA3 架构中的 GSP 固件介绍
- VBIOS and GPU Firmware Update Guide — NVIDIA 固件升级说明
- Understanding GPU Timeout Detection and Recovery — TDR 与固件协作的机制分析
