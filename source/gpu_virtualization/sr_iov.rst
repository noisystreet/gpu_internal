==========================
SR-IOV 与 vGPU
==========================

.. epigraph::

   In mathematics, you don't understand things. You just get used to them.

   — John von Neumann

SR-IOV（Single Root I/O Virtualization）是一种 PCIe 标准，允许一个物理设备呈现为多个独立的虚拟设备。

SR-IOV 架构
=================

.. code-block:: text

   物理功能 (Physical Function, PF)
   +----------------------------------------------------+
   |  完整 GPU 硬件                                       |
   |  具备所有硬件资源和完整驱动栈                          |
   |  管理所有 VF 的生命周期                               |
   +---------------------------+------------------------+
                               |
          +--------------------+--------------------+
          |                    |                    |
   +------+------+     +------+------+     +------+------+
   | VF 0       |     | VF 1       |     | VF 2       |
   | (虚拟功能)  |     | (虚拟功能)  |     | (虚拟功能)  |
   | SM 切片    |     | SM 切片    |     | SM 切片    |
   | 显存分区    |     | 显存分区    |     | 显存分区    |
   | QM (队列)  |     | QM (队列)  |     | QM (队列)  |
   +------------+     +------------+     +------------+

**关键组件**:

- **PF（Physical Function）**: 持有完整 GPU 资源的管理者，驱动运行在宿主机
- **VF（Virtual Function）**: 轻量级虚拟设备，呈现为独立 GPU 给虚拟机
- **QM（Queue Manager）**: 硬件队列管理器，为每个 VF 提供独立命令队列

**与 MIG 的关键区别**:

.. list-table::
   :header-rows: 1

   * - 特性
     - MIG
     - SR-IOV
   * - 标准
     - NVIDIA 私有
     - PCIe 标准 (SR-IOV)
   * - 虚拟机直通
     - 需要 vGPU 软件栈
     - 原生支持 VF 直通
   * - 动态分区
     - 需重建配置
     - 可以动态调整
   * - 厂商支持
     - NVIDIA
     - AMD (AMDGPU), Intel (GVT-g)
   * - 功能集
     - 完整 CUDA 功能
     - 部分高级功能受限

AMD GPU SR-IOV 的实现
==============================

AMD 的 GPU SR-IOV 实现基于其 ROCm 驱动栈：

.. code-block:: text

   虚拟机 0              虚拟机 1              虚拟机 N
   +------------+       +------------+       +------------+
   | ROCm+应用  |       | ROCm+应用  |       | ROCm+应用  |
   | amdgpu vf  |       | amdgpu vf  |       | amdgpu vf  |
   +-----+------+       +-----+------+       +-----+------+
         |                     |                     |
   +-----+---------------------+---------------------+---+
   | Hypervisor (KVM)                                   |
   | SR-IOV Passthrough                                 |
   +-----+---------------------+---------------------+---+
         |                     |                     |
   +-----+---------------------+---------------------+---+
   | 宿主机 amdgpu PF 驱动                               |
   | PF 管理: VF 创建/销毁、显存分区、队列分配              |
   +----------------------------------------------------+
   | AMD GPU 硬件 (MI200/MI300)                          |
   +----------------------------------------------------+

**配置示例**:

.. code-block:: bash
   :linenos:

   # 查看 SR-IOV 能力
   cat /sys/class/dev/pci/<address>/sriov_totalvfs

   # 设置 VF 数量
   echo 4 > /sys/class/dev/pci/<address>/sriov_numvfs

   # 查看 VF
   lspci | grep "AMD.*MI"

NVIDIA vGPU（时间切片）
==============================

NVIDIA vGPU（原 GRID vGPU）是另一种虚拟化方案，使用软件方式在时间维度上复用 GPU：

.. code-block:: text

   虚拟机 0          虚拟机 1          虚拟机 2
   +--------+       +--------+       +--------+
   | vGPU   |       | vGPU   |       | vGPU   |
   | 驱动    |       | 驱动    |       | 驱动    |
   +----+---+       +----+---+       +----+---+
        |                |                |
   +----+----+-----+----+----+-----+----+----+
   |   NVIDIA vGPU Manager (宿主机)          |
   |   - 时间切片调度                        |
   |   - 显存超分管理                        |
   |   - 帧缓存管理                          |
   +----+----------------------------------+
        |
   +----+----------------------------------+
   |   NVIDIA GPU 硬件                      |
   +---------------------------------------+

**vGPU 时间切片工作模式**:

.. code-block:: text

   时间 →
   +----------+----------+----------+----------+
   | VM0      | VM1      | VM2      | VM0      |
   | (5 ms)   | (5 ms)   | (5 ms)   | (5 ms)   |
   +----------+----------+----------+----------+

每个时间片内，vGPU Manager 将全部 GPU 资源分配给对应的虚拟机，通过快速上下文切换实现多 VM 共享。

GPU 池化与远程虚拟化
==============================

以上方案都是针对单机内的 GPU 共享，但在数据中心场景中，往往需要在更宏观的层面管理 GPU 资源——多个服务器之间的 GPU 池化、容器化编排、以及跨网络的远程 GPU 访问。

.. list-table::
   :header-rows: 1

   * - 技术
     - 原理
     - 性能开销
     - 适用场景
   * - rCUDA
     - CUDA API 转发 (网络)
     - 10-30%
     - 数据中心 GPU 池化
   * - gVirt (Intel)
     - 半虚拟化 GPU
     - 5-15%
     - 云桌面
   * - GPU 分区 (MPS)
     - CUDA MPS 服务
     - < 5%
     - 多进程共享计算资源
   * - Run:ai / Volcano
     - Kubernetes 调度
     - < 3%
     - 容器化 GPU 编排

其中 **rCUDA** 的实现思路值得关注——它在远程服务器上运行一个 CUDA API 代理，客户端的 CUDA 调用被序列化为网络消息发送到代理执行。这种模式使得没有 GPU 的机器也能运行 CUDA 程序，代价是显著的网络延迟开销（尤其在小数据传输中）。

而 **MPS（Multi-Process Service）** 则是单节点内最轻量的 GPU 共享方案，允许多个进程共享同一 GPU 的计算资源。与虚拟化方案不同，MPS 不提供隔离保障，但在推理服务等对隔离要求不高的场景中表现优异。

MPS 架构
"""""""""""

MPS 通过一个持久化的控制守护进程（``nvidia-cuda-mps-control``）和客户端库实现多进程 CUDA kernel 并发：

.. mermaid::

   flowchart TB
       subgraph Apps["多进程"]
           PA["进程 A<br/>(CUDA)"]
           PB["进程 B<br/>(CUDA)"]
           PC["进程 C<br/>(CUDA)"]
           PD["进程 D<br/>(CUDA)"]
       end
       subgraph MPS_LIB["MPS Client Library"]
           MCL["合并提交"]
       end
       subgraph MPS_SRV["CUDA MPS Server<br/>nvidia-cuda-mps-control"]
           MS["单一 CUDA context<br/>命令合并"]
       end
       subgraph HWS["GPU 硬件调度器"]
           GW["时间片轮转"]
       end
       GPU["GPU"]

       PA --- MCL
       PB --- MCL
       PC --- MCL
       PD --- MCL
       MCL --- MS
       MS --- GW
       GW --- GPU

       style Apps fill:#e8eaf6,color:#283593
       style MPS_LIB fill:#e3f2fd,color:#1565c0
       style MPS_SRV fill:#fff3e0,color:#e65100
       style HWS fill:#f3e5f5,color:#7b1fa2

**MPS 的工作原理**:

1. MPS Server 创建一个**共享的 CUDA context**，所有连接的 client 共享
2. 所有 client 的 kernel 提交到同一个 context，通过 GPU 硬件调度器并发执行
3. MPS 会为 client 端的 kernel 生成唯一标识（GUEST_ID），用于错误隔离
4. 数据通道（client → server → GPU）使用共享内存以减少开销

**配置参数**:

.. code-block:: bash

   # 启动 MPS
   nvidia-cuda-mps-control -d

   # 设置默认活跃线程百分比 (0-100)
   echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control

   # 设置每客户端的最大活跃线程百分比
   echo "set_client_active_thread_percentage 30" | nvidia-cuda-mps-control

   # 查看 MPS 进程
   ps aux | grep mps

   # 停止 MPS
   echo "quit" | nvidia-cuda-mps-control

MPS vs MIG vs 时间切片
"""""""""""""""""""""""""

.. list-table::
   :header-rows: 1

   * - 特性
     - MPS
     - MIG
     - 时间切片
   * - 隔离级别
     - 进程级（无强隔离）
     - 硬件级隔离
     - 逻辑隔离
   * - 错误隔离
     - 差（一个 client 崩溃可能影响所有 client）
     - 完全隔离
     - 较好
   * - 显存隔离
     - 无（共享显存池）
     - 物理分区
     - 无（依赖驱动管理）
   * - 并发粒度
     - kernel 级（多 kernel 同时执行）
     - SM 分区独立运行
     - 时间片轮转
   * - 调度延迟
     - 极低（共享 context）
     - 低（硬件分区）
     - 中（需上下文切换）
   * - 适用场景
     - 多进程推理服务、小批量计算
     - 多租户训练、严格隔离需求
     - 桌面虚拟化、通用 GPU 共享

**MPS 的典型使用模式**:

.. code-block:: bash

   # AI 推理场景：多个 Python 进程共享 GPU
   nvidia-cuda-mps-control -d

   # 启动多个推理进程，kernel 会自动在 GPU 上并发执行
   python inference.py --model a &
   python inference.py --model b &
   python inference.py --model c &

   wait
   echo "quit" | nvidia-cuda-mps-control

**MPS 限制**:

- 不支持 ``cudaSetDevice()`` 在 MPS client 内部切换 GPU
- MPS client 崩溃可能导致 MPS server 终止，所有 client 受影响
- 不支持所有 CUDA 版本的高级 API（如 CUDA Graph 部分功能受限）
- 显存共享意味着一个 client 可能耗尽所有显存

参考与拓展阅读
====================

- CUDA Multi-Process Service (https://docs.nvidia.com/deploy/mps/) — CUDA MPS 完整文档和配置指南
- NVIDIA Multi-Instance GPU User Guide (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — 与 MIG 方案的对比参考
- AMD ROCm Documentation (https://rocm.docs.amd.com/) — AMD GPU SR-IOV 的实现说明
- CUDA C++ Best Practices Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — 多进程 GPU 共享的性能优化
