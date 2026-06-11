========================
WMMA API 与 mma API
========================

NVIDIA 提供了两套编程接口操作 Tensor Core，它们在灵活性和性能上存在显著差异（Sun 等, 2022）：

.. list-table::
   :header-rows: 1

   * - 特性
     - WMMA API (legacy)
     - mma API (PTX)
   * - 抽象层次
     - 高层封装（fragment 抽象）
     - 底层 PTX 指令直接操控
   * - 支持的 tile 形状
     - 仅 16x16x16 (FP16)
     - m16n8k16, m16n8k8, m8n8k4 等多种
   * - 稀疏矩阵支持
     - 不支持
     - Ampere+ 支持 2:4 稀疏 (mma.sp)
   * - 数据加载
     - load_matrix_sync（自动布局）
     - ldmatrix（手动布局，精确控制）
   * - 性能（同条件下）
     - 基准
     - 略优（减少 fragment 布局开销）
   * - 可编程性
     - 更简单
     - 更复杂

.. code-block:: cuda

   // mma API (PTX) 直接调用 Tensor Core 指令
   __global__ void tensor_core_mma(const half* A, const half* B, float* D) {
       // 使用 ldmatrix 指令加载数据到寄存器
       uint32_t a_reg[4], b_reg[4];  // 寄存器碎片
       asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
           : "=r"(a_reg[0]), "=r"(a_reg[1]), "=r"(a_reg[2]), "=r"(a_reg[3])
           : "r"(shared_addr_A));

       uint32_t c_reg[4] = {0};  // 累加器初始化为 0
       uint32_t d_reg[4];

       // mma 指令：16x8x16 FP16 → FP32
       asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
           "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15};\n"
           : "=r"(d_reg[0]), "=r"(d_reg[1]), "=r"(d_reg[2]), "=r"(d_reg[3])
           : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
             "r"(b_reg[0]), "r"(b_reg[1]), "r"(b_reg[2]), "r"(b_reg[3]),
             "r"(c_reg[0]), "r"(c_reg[1]), "r"(c_reg[2]), "r"(c_reg[3]));
   }

指令吞吐与延迟数据
======================

Sun 等人（2022）通过微基准测试测得的 Tensor Core 指令性能：

.. list-table::
   :header-rows: 1

   * - 指令（Ampere A100）
     - 延迟（周期）
     - 吞吐（每 SM/周期）
     - 说明
   * - mma.m16n8k16 (FP16)
     - ~8 周期
     - 4 条
     - 主流矩阵尺寸
   * - mma.m16n8k8 (BF16)
     - ~8 周期
     - 4 条
     - BF16 格式
   * - mma.m16n8k4 (TF32)
     - ~8 周期
     - 4 条
     - 19 位精度
   * - mma.m16n8k16 (INT8)
     - ~8 周期
     - 4 条
     - INT8 量化
   * - mma.sp (2:4 稀疏)
     - ~8 周期
     - 4 条
     - 2x 加速比
   * - ldmatrix (加载)
     - ~12-16 周期
     - 2 条
     - 数据加载到寄存器

.. note::

   上述数据基于 Ampere A100 通过 CUDA PTX 指令级别的微基准测试（Sun 等, 2022; Abdelkhalik 等, 2022）。
   实际应用中的端到端吞吐还受数据加载、共享内存 bank 冲突和寄存器压力等因素影响。

参考与拓展阅读
====================

- 深入理解 :doc:`tensor_core_architecture` — Tensor Core 工作原理
- 深入理解 :doc:`tensor_core_numerical` — 数值行为分析
- Demystifying Nvidia Ampere Architecture (https://arxiv.org/abs/2208.11174) — 指令周期测量
