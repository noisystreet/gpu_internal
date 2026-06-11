==========================
GPU 虚拟化技术概览
==========================

.. epigraph::

   Beware of bugs in the above code; I have only proved it correct, not tried it.

   — Donald E. Knuth

GPU 虚拟化允许将一块物理 GPU 分割为多个逻辑 GPU 实例，供多个用户或工作负载独立使用。与 CPU 虚拟化不同，GPU 虚拟化面临一个独特的挑战：GPU 的设计目标是最大化并行吞吐，其硬件假设"独占"资源。因此，不同的虚拟化技术在隔离性、性能、灵活性间做出不同的取舍。

虚拟化层次
===============

GPU 虚拟化可以在多个层次实现。理解这些层次有助于根据业务需求选择合适的方案：

.. mermaid::

   flowchart TB
       subgraph HW["硬件分区"]
           MIG["MIG (NVIDIA)<br/>物理隔离"]
           SRIOV["SR-IOV (AMD/Intel)<br/>物理隔离"]
       end
       subgraph DRV["驱动/运行时"]
           TS["时间切片<br/>逻辑隔离"]
           VGPU["vGPU (NVIDIA)<br/>逻辑隔离"]
       end
       subgraph API["API 转发"]
           RD["remote-desktop<br/>逻辑隔离"]
           RC["rCUDA / gVirt<br/>弱隔离"]
       end

       MIG -->|"隔离最强"| SRIOV
       SRIOV --> TS
       TS --> VGPU
       VGPU --> RD
       RD --> RC

       style HW fill:#e8f5e9,color:#1b5e20
       style DRV fill:#fff3e0,color:#e65100
       style API fill:#fce4ec,color:#b71c1c

技术对比
=============

不同的虚拟化方案在不同维度的表现差异很大。下表从隔离性、性能开销、故障隔离等关键维度进行对比：

.. list-table::
   :header-rows: 1

   * - 特性
     - MIG
     - SR-IOV
     - vGPU (时间切片)
     - API 转发
   * - 隔离级别
     - 物理硬件隔离
     - 物理硬件隔离
     - 逻辑隔离
     - 进程级
   * - 性能开销
     - 接近 0
     - 接近 0
     - < 5%
     - 10-30%
   * - 故障隔离
     - 完全隔离
     - 完全隔离
     - 部分
     - 部分
   * - 驱动要求
     - 硬件 + 驱动支持
     - 硬件 + 驱动支持
     - NVIDIA vGPU 驱动
     - 特殊运行时
   * - 支持厂商
     - NVIDIA
     - AMD, Intel
     - NVIDIA
     - 各厂商
   * - 跨网络
     - 不支持
     - 不支持
     - 不支持
     - 支持

典型应用场景
=================

.. list-table::
   :header-rows: 1

   * - 场景
     - 推荐方案
     - 理由
   * - AI 训练多租户
     - MIG / SR-IOV
     - 严格资源隔离，性能可预测
   * - AI 推理弹性部署
     - MIG / 时间切片
     - 灵活调配算力，GPU 利用率最大化
   * - 云桌面/VDI
     - vGPU
     - 完整的图形和计算能力
   * - HPC 集群共享
     - SR-IOV / 裸金属
     - 低延迟，高带宽
   * - 跨数据中心
     - rCUDA / Remote API
     - 资源池化远程访问

性能基准对比
=================

以下数据展示不同虚拟化方案在典型 AI 训练工作负载（ResNet-50）上的性能开销：

.. list-table::
   :header-rows: 1

   * - 方案
     - 训练吞吐（相对于裸金属）
     - 额外显存开销
     - Kernel 启动延迟
   * - 裸金属（基准）
     - 100%
     - 0
     - 基准
   * - MIG (1g.10gb)
     - ~98-99%
     - ~200 MB（固件预留）
     - 基准
   * - SR-IOV (VF)
     - ~97-99%
     - ~150 MB
     - +1-2 us
   * - vGPU 时间切片 (4 VM)
     - ~93-96%
     - ~500 MB
     - +5-10 us
   * - MPS (4 client)
     - ~95-97%
     - 共享显存池
     - +2-3 us
   * - rCUDA (同一节点)
     - ~85-95%
     - ~100 MB
     - +10-30 us
   * - rCUDA (跨网络)
     - ~60-85%
     - ~100 MB
     - +50-200 us

.. note::

   以上数据为参考值，实际性能取决于具体工作负载、GPU 型号和驱动版本。推理场景下的性能差异通常小于训练场景。

选型决策流程
=================

选择合适的 GPU 虚拟化方案需要综合考虑隔离需求、性能容忍度和管理复杂度：

.. code-block:: text

   是否需要物理隔离（故障隔离 + QoS 保障）？
   ├── 是 ──→ 是否需要多 VM 直接直通？
   │          ├── 是 ──→ SR-IOV (AMD/Intel 环境)
   │          └── 否 ──→ MIG (NVIDIA 环境)
   │                     └── 是否需要动态调整分区？
   │                         ├── 是 ──→ 考虑 SR-IOV 的动态 VF
   │                         └── 否 ──→ MIG 满足需求
   │
   └── 否 ──→ 是否需要跨网络共享？
              ├── 是 ──→ rCUDA / Remote API
              └── 否 ──→ 是否需要图形能力？
                         ├── 是 ──→ vGPU 时间切片
                         └── 否 ──→ 是否需要多进程并发？
                                    ├── 是 ──→ MPS
                                    └── 否 ──→ 裸金属或时间切片

   （参考上述决策树，结合硬件支持情况和业务需求）

典型部署拓扑
=================

**单机多租户（MIG）**:

.. code-block:: text

   +----------------------------------------------------+
   |               物理服务器 (A100 x8)                   |
   |  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      |
   |  │租户A   │ │租户B   │ │租户C   │ │租户D   │      |
   |  │MIG:3g40│ │MIG:1g20│ │MIG:2g20│ │MIG:1g20│      |
   |  └────────┘ └────────┘ └────────┘ └────────┘      |
   |  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      |
   |  │租户E   │ │租户F   │ │租户G   │ │租户H   │      |
   |  │MIG:1g20│ │MIG:2g20│ │MIG:1g10│ │MIG:1g10│      |
   |  └────────┘ └────────┘ └────────┘ └────────┘      |
   +----------------------------------------------------+

**Kubernetes GPU 共享（MPS）**:

.. code-block:: text

   +----------------------------------------------------+
   |                Kubernetes 节点                       |
   |  ┌──────────────┐  ┌──────────────┐                 |
   |  │ Pod A        │  │ Pod B        │                 |
   |  │ 推理服务     │  │ 推理服务     │                 |
   |  └──────┬───────┘  └──────┬───────┘                 |
   |         │                 │                          |
   |         └──────┬──────────┘                          |
   |           CUDA MPS Server                           |
   |                │                                     |
   |           GPU (通过 Kubernetes device plugin)        |
   +----------------------------------------------------+

行业趋势
=================

GPU 虚拟化技术处于快速发展中，几个值得关注的趋势：

.. list-table::
   :header-rows: 1

   * - 趋势
     - 说明
     - 代表技术/厂商
   * - 更细粒度的分区
     - 从整卡到 1/7 再到 1/32 的粒度演进
     - NVIDIA MIG, AMD SR-IOV 增强
   * - 动态重配置
     - 不重启即可调整分区大小
     - SR-IOV, 未来 MIG 演进
   * - 池化 + 编排
     - GPU 资源池化，Kubernetes 原生调度
     - Run:ai, Volcano, NVIDIA AI Enterprise
   * - 远程 + 边缘
     - 跨网络 GPU 共享，边缘推理
     - rCUDA, NVIDIA Fleet Command
   * - 开放标准
     - 厂商中立的标准正在形成
     - CXL (Compute Express Link), 开放加速器基础设施

参考与拓展阅读
====================

- NVIDIA Multi-Instance GPU User Guide (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — NVIDIA MIG 用户指南
- CUDA Multi-Process Service (https://docs.nvidia.com/deploy/mps/) — CUDA Multi-Process Service 文档
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — GPU 虚拟化场景下的性能调优建议
