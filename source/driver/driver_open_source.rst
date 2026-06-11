========================
开源 vs 闭源驱动
========================

.. epigraph::

   Given enough eyeballs, all bugs are shallow.

   — Eric S. Raymond, 《大教堂与市集》

GPU 驱动是操作系统中最复杂的用户态-内核态接口之一。不同厂商在驱动的开源程度上采取了截然不同的策略，这直接影响开发者的调试能力、部署灵活性和系统安全性。

各厂商的开源策略
=======================

**NVIDIA**

NVIDIA 长期以来采用 **完全闭源** 的策略。驱动由 NVIDIA 开发并以二进止形式提供：

.. code-block:: text

   NVIDIA 驱动栈:
   用户态: libcuda.so — 闭源 (如有契约可获源码)
   用户态: libnvidia-ptxjitcompiler.so — 闭源
   内核态: nvidia.ko — 闭源 (自 2022 年起提供 nvidia-open 内核模块)
   固件: 闭源，带数字签名

   License:
   - 内核模块: GPL + NVIDIA 专有例外
   - 用户态库: NVIDIA EULA (专有)
   - 开发者 SDK: CUDA Toolkit 免费但非开源

2022 年转变：NVIDIA 开源了内核模块 nvidia-open（GitHub: NVIDIA/open-gpu-kernel-modules），但：
- 仅支持较新的 GPU（Turing+）
- 用户态库（libcuda.so、CUDA Runtime）仍然闭源
- 这是一个 GPL 兼容的发布，但 NVIDIA 仍控制开发

.. code-block:: bash

   # 使用开源内核模块（推荐用于新 GPU）
   # /etc/modprobe.d/nvidia.conf
   options nvidia-drm modeset=1
   # 安装开源模块
   apt install nvidia-driver-545-open  # Ubuntu
   dnf install nvidia-open             # Fedora

**AMD**

AMD 采用 **完全开源** 的策略，驱动从内核到用户态全部开源：

.. code-block:: text

   AMD 驱动栈:
   用户态: ROCm Runtime — 开源 (MIT)
   用户态: HIP 编译器 — 开源 (LLVM)
   用户态: rocBLAS / MIOpen / RCCL — 开源 (MIT)
   内核态: amdgpu.ko — 开源 (GPL + MIT)
   固件: 闭源 (带数字签名)

   License:
   - 内核模块: GPL
   - 用户态库: MIT / Apache 2.0
   - 固件: AMD 专有 (amdgpu-firmware)

AMD 的 ROCm 栈从编译器到运行时到数学库全部开源，开发者可以自行构建、修改和调试整个 GPU 软件栈。这是 AMD 的主要差异化优势。

.. code-block:: bash

   # 从源码构建 ROCm (可完全定制)
   git clone https://github.com/ROCm/ROCm.git
   cd ROCm
   python3 install.py --build_all

**Intel**

Intel 同样采用 **完全开源** 的策略：

.. code-block:: text

   Intel 驱动栈:
   用户态: Level Zero — 开源 (MIT)
   用户态: oneAPI 库 — 开源 (MIT)
   内核态: i915.ko / xe.ko — 开源 (GPL)
   固件: 闭源 (带数字签名)

   License:
   - 内核模块: GPL
   - 用户态库: MIT
   - 固件: Intel 专有

Intel 的 xe 驱动（eXperimental Engine）是全新的内核 GPU 驱动架构，采用现代 DRM 框架，代码质量被业界认为是三家中最干净的。

开源的利弊
=================

**开源的优势**：

.. list-table::
   :header-rows: 1

   * - 优势
     - 说明
     - 适用场景
   * - 可调试性
     - 开发者可以阅读驱动代码定位问题，无需依赖厂商支持
     - 驱动 bug 排查、性能分析
   * - 可定制性
     - 可以修改驱动行为（调功耗、改调度参数）
     - HPC 集群定制、嵌入式系统
   * - 安全透明
     - 安全专家可以审计代码，减少后门风险
     - 政府、国防、金融
   * - 社区维护
     - 内核社区帮助发现和修复 bug
     - 长期稳定性（驱动不因厂商放弃而失修）
   * - 集成速度
     - 开源驱动可以直接进入主线内核
     - Linux 发行版开箱即用

**闭源的弊端**：

.. list-table::
   :header-rows: 1

   * - 问题
     - 具体表现
   * - 黑盒调试
     - crash 时只能收集有限的日志，无法单步跟踪
   * - 安全依赖厂商
     - CVE 修补等待厂商发布，中间期风险敞口
   * - 内核兼容
     - 每次内核 ABI 变更需要 NVIDIA 适配新版本
   * - 集成延迟
     - 新 Linux 内核发布后可能需要数周才能获得兼容驱动
   * - 认证限制
     - 某些签名/加密模块要求内核锁状态，与安全启动冲突

闭源驱动在 **性能一致性** 上有优势——厂商专注于有限的硬件组合做深度优化，不受社区补丁质量不均的影响。

Nouveau：逆向工程的开源驱动
===================================

Nouveau 是社区通过逆向工程开发的 NVIDIA GPU 开源驱动，目前仍是一个**实验性方案**：

.. code-block:: text

   Nouveau 现状:
   用户态: Mesa + Nouveau Gallium3D — 开源 (MIT)
   内核态: nouveau.ko — 开源 (GPL)
   固件: 无法加载签名固件 → 功能受限
   性能: 普遍为官方驱动的 30-60%

   主要限制:
   - 无法调整 GPU 时钟（没有 PMU 固件的控制接口）
   - 不支持 GPU 重新时钟（reclocking）
   - 不支持 CUDA / GPGPU（没有用户态 API）
   - 不支持电源管理（GPU 以最低频率运行）
   - 不支持 GPU 虚拟化和 MIG

Nouveau 受限于 NVIDIA 不公开 GPU 寄存器文档，且签名固件无法被替换。尽管 NVIDIA 开源了内核模块（nvidia-open），但 Nouveau 仍然无法利用这些新代码，因为用户态接口仍然封闭。

实际部署建议
======================

.. code-block:: text

   场景                         推荐方案
   ───────────────────────────────────────────────────
   深度学习训练/推理             NVIDIA 闭源驱动
   CUDA 开发                     NVIDIA 闭源驱动
   通用桌面图形（NVIDIA）         NVIDIA 闭源驱动 (nvidia-open)
   通用桌面图形（AMD）            amdgpu 开源驱动 + Mesa
   通用桌面图形（Intel）          i915 开源驱动 + Mesa
   HPC 集群                      AMD 开源栈 / NVIDIA 闭源栈
   嵌入式/定制系统               AMD ROCm（可定制）
   嵌入式（NVIDIA）              NVIDIA 闭源驱动（JetPack）
   桌面 Linux（仅显示）          Nouveau（基本可用）

四大厂商驱动对比
=======================

.. list-table::
   :header-rows: 1

   * - 维度
     - NVIDIA
     - AMD
     - Intel
     - Nouveau
   * - 内核模块
     - 闭源 / 部分开源
     - 开源
     - 开源
     - 开源
   * - 用户态库
     - 闭源
     - 开源
     - 开源
     - 开源
   * - 固件
     - 闭源签名
     - 闭源签名
     - 闭源签名
     - N/A (受限)
   * - CUDA/HIP/SYCL
     - CUDA（仅 NVIDIA）
     - HIP / ROCm
     - SYCL / oneAPI
     - 不支持
   * - 独立 GPU 支持
     - 全系列
     - Radeon + Instinct
     - Arc + Max 系列
     - 部分旧 GPU
   * - 性能（3D）
     - 100%（基准）
     - ~95-100%
     - ~90-95%
     - ~30-60%
   * - 性能（计算）
     - 100%
     - ~95-100%
     - ~85-95%
     - 不支持
   * - 内核主线集成
     - 否（需要 DKMS）
     - 是（amdgpu 在内核主线）
     - 是（i915/xe 在内核主线）
     - 是
   * - 调试工具
     - Nsight（闭源）
     - rocprof（开源）
     - Intel VTune（部分开源）
     - GALLIUM_HUD

参考与拓展阅读
====================

- NVIDIA Open GPU Kernel Modules — https://github.com/NVIDIA/open-gpu-kernel-modules
- AMD ROCm Documentation — https://rocm.docs.amd.com/
- Nouveau Wiki — https://nouveau.freedesktop.org/
- Linux Kernel GPU Driver Documentation — https://kernel.org/doc/html/latest/gpu/
- Freedesktop Mesa 3D — https://www.mesa3d.org/
