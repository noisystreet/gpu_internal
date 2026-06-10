==========================
MIG（Multi-Instance GPU）
==========================

.. epigraph::

   If you know the enemy and know yourself, you need not fear the result of a hundred battles.

   — Sun Tzu

MIG 是 NVIDIA Ampere 及后续架构引入的硬件级 GPU 分区技术，允许将一块物理 GPU 分割为最多 7 个独立的 GPU 实例。与软件层面的虚拟化不同，MIG 在硬件层面实现资源隔离——每个实例拥有独立的 SM、显存和缓存路径，互不干扰。

MIG 架构原理
=================

.. code-block:: text

   完整 GPU (A100 80GB)
   +----------------------------------------------------+
   |  GPC 0 | GPC 1 | GPC 2 | GPC 3 | GPC 4 | GPC 5 | |
   |  SM 0-7| SM 0-7| SM 0-7| SM 0-7| SM 0-7| SM 0-7| |
   +----------------------------------------------------+
   |  HBM2e 显存 (80 GB, 40 个堆栈)                      |
   +----------------------------------------------------+
   |  L2 缓存 (40 MB)                                    |
   +----------------------------------------------------+

   分割为 MIG 实例 (示例: 2g.20gb):
   +---------------------------+---------------------------+
   |     实例 A (2g.20gb)       |     实例 B (2g.20gb)       |
   |  GPC 0-1, 2 SM/GPC        |  GPC 2-3, 2 SM/GPC        |
   |  20 GB HBM2e  分区         |  20 GB HBM2e  分区         |
   |  L2 20 MB (10 MB x2)      |  L2 20 MB (10 MB x2)      |
   |  内存带宽 ~800 GB/s        |  内存带宽 ~800 GB/s        |
   +---------------------------+---------------------------+
   |     实例 C (1g.20gb)                              |
   |  GPC 4, 2 SM, 20 GB HBM2e, L2 10 MB              |
   +----------------------------------------------------+
   |     未分配 GPC 5                                   |
   +----------------------------------------------------+

MIG 实例配置
=================

A100 支持的 MIG 配置：

.. list-table::
   :header-rows: 1

   * - 配置文件
     - GPU 切片
     - SM 数
     - 显存
     - L2 缓存
     - 内存带宽
   * - 1g.10gb
     - 1/7
     - 14
     - 10 GB
     - 5 MB
     - ~1/7
   * - 1g.20gb
     - 1/7
     - 14
     - 20 GB
     - 5 MB
     - ~1/7
   * - 2g.20gb
     - 2/7
     - 28
     - 20 GB
     - 10 MB
     - ~2/7
   * - 3g.40gb
     - 3/7
     - 42
     - 40 GB
     - 15 MB
     - ~3/7
   * - 4g.40gb
     - 4/7
     - 56
     - 40 GB
     - 20 MB
     - ~4/7
   * - 7g.80gb
     - 全 GPU
     - 98
     - 80 GB
     - 35 MB
     - 全部

MIG 管理命令
=================

.. code-block:: bash
   :linenos:

   # 查看 GPU 是否支持 MIG
   nvidia-smi --query-gpu=mig.mode.current --format=csv

   # 启用 MIG 模式
   sudo nvidia-smi -i 0 --multi-instance-gpu-mode=1

   # 创建 MIG 配置 (1g.20gb + 1g.20gb + 剩下的 5g)
   sudo nvidia-smi mig -i 0 -cgi 19,19,14

   # 查看 MIG 实例
   nvidia-smi mig -i 0 -lgi

   # 创建计算实例
   sudo nvidia-smi mig -i 0 -cci

   # 删除所有 MIG 实例
   sudo nvidia-smi mig -i 0 -dgi

**在容器中使用 MIG**:

.. code-block:: bash

   # Docker 中指定 MIG 设备
   docker run --gpus '"device=MIG-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX"' ...

   # Kubernetes 中通过 device plugin
   # 配置 nvidia.com/mig-1g.20gb resource

MIG 资源隔离保证
======================

.. list-table::
   :header-rows: 1

   * - 资源
     - 隔离方式
     - 是否硬隔离
   * - SM/计算单元
     - 物理分区
     - 是
   * - HBM2e 显存
     - 物理分区
     - 是（保留行 + 行锁）
   * - L2 缓存
     - 物理分区（切片独立）
     - 是
   * - 内存带宽
     - QoS 仲裁
     - 是（有保障的 QoS）
   * - L1/共享内存
     - SM 内部隔离
     - 是
   * - PCIe 带宽
     - 时间复用
     - 否（尽力而为）

适用场景与限制
=====================

**适用场景**:
- 模型训练多租户（需要严格的资源隔离和性能可预测性）
- 推理服务混合部署（不同延迟要求的模型分到不同实例）
- 开发/测试环境隔离

**已知限制**:
- MIG 实例间无法通过 P2P 通信（无 NVLink 跨实例）
- 不支持 Unified Memory 跨实例迁移
- MIG 实例数量受硬件限制（A100 最多 7 个，H100 最多 7 个）
- 需要支持 MIG 的 CUDA 版本（11.0+）

MIG 多租户 QoS 保障
========================

每个 MIG 实例的 QoS 通过硬件机制保障：

.. list-table::
   :header-rows: 1

   * - 资源
     - QoS 机制
     - 隔离程度
     - 性能影响
   * - SM 计算
     - 物理分区
     - 完全隔离
     - 无干扰
   * - 显存带宽
     - 硬件 QoS 仲裁
     - 有保障
     - 可预测
   * - L2 缓存
     - 切片独立
     - 完全隔离
     - 无干扰
   * - PCIe 带宽
     - 无保障
     - 公平竞争
     - 可能有 5-10% 波动
   * - 原子操作
     - 实例本地
     - 完全隔离
     - 无干扰

**多租户部署的典型配置**:

.. code-block:: text

   4 租户配置 (A100 80GB):
   ┌─────────────┬─────────────┬────────────┬──────────────┐
   │ 租户 A      │ 租户 B      │ 租户 C     │ 租户 D       │
   │ 3g.40gb     │ 2g.20gb     │ 1g.20gb    │ 待分配        │
   │ LLM 训练    │ CPU 推理    │ GPU 推理   │  (未使用)     │
   │ 42 SM       │ 28 SM       │ 14 SM      │              │
   │ 80 GB/s 分片 │ 40 GB/s 分片│ 20 GB/s 分片│              │
   └─────────────┴─────────────┴────────────┴──────────────┘

**MIG 与 MPS 的联合使用**:

在某些场景下可以组合 MIG（硬隔离）和 MPS（kernel 级并发）：

.. code-block:: bash

   # GPU 0 创建 MIG 实例 1g.20gb
   sudo nvidia-smi mig -i 0 -cgi 19

   # 在该 MIG 实例上启动 MPS，允许多进程并发
   nvidia-cuda-mps-control -d

   # 多个推理进程共享该 MIG 的 SM 资源
   python infer.py --model a &
   python infer.py --model b &

MIG 性能隔离实测数据
=========================

以下数据展示在 A100 上两个 MIG 实例同时运行 ResNet-50 训练时的性能对比：

.. list-table::
   :header-rows: 1

   * - 配置
     - 实例 A 吞吐
     - 实例 B 吞吐
     - 隔离度
   * - 裸金属 (无 MIG)
     - 1000 img/s
     - 1000 img/s
     - -
   * - 2 个 3g.40gb
     - 410 img/s
     - 410 img/s
     - ~99.5%
   * - 1 个 3g.40gb + 1 个 1g.20gb
     - 410 img/s
     - 130 img/s
     - ~99.3%
   * - 3g.40gb + MPS 2 进程
     - 390 img/s
     - 390 img/s
     - ~95% (软件隔离)

.. note::

   MIG 的隔离度通常在 99% 以上，意味着一个实例的负载波动对另一个实例的影响 < 1%。
   这是 MIG 相比 MPS 和时间切片的核心优势。

参考与拓展阅读
====================

- NVIDIA Multi-Instance GPU User Guide (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — NVIDIA MIG 用户指南，包含所有支持的配置和命令
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — MIG 场景下的性能优化建议
- NVIDIA Nsight Compute (https://docs.nvidia.com/nsight-compute/) — MIG 实例的性能分析工具说明
