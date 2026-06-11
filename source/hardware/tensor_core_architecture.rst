========================
Tensor Core 工作原理
========================

Tensor Core 的操作单位是 **warp（32 线程）**。整个 warp 协作完成一个矩阵分块的乘法累加，而不是每个线程独立运算：

.. code-block:: text

   Warp (32 线程) 协作执行一次 MMA 操作：

   A (16x16)       B (16x16)        C (16x16)         D (16x16)
   ┌──────┐        ┌──────┐         ┌──────┐          ┌──────┐
   │      │        │      │         │      │          │      │
   │      │   ×    │      │    +    │      │    =     │      │
   │      │        │      │         │      │          │      │
   └──────┘        └──────┘         └──────┘          └──────┘

   - A 和 C 矩阵存储在每个线程的寄存器中
   - B 矩阵存储在共享内存中（或寄存器，取决于 tile 大小）
   - 一次 MMA 指令：所有 32 线程协作，在 1 个周期内完成
   - 等价于 4096 次 FMA 运算（16x16x16）

指令级操作
==============

Tensor Core 操作在 PTX 指令层面表示为 ``mma.sync``：

.. code-block:: text

   // Ampere 架构的 mma 指令示例
   mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
   { d[0..3] }, { a[0..3] }, { b[0..3] }, { c[0..3] }

   参数含义:
   - m16n8k16: A 矩阵 16x16, B 矩阵 16x8, 内积维度 k=16
   - row.col: A 行主序, B 列主序
   - f16.f16.f16.f16: A/B/C/D 的精度

编程接口
============

**WMMA API（warp matrix multiply-accumulate）** — 用户友好的高层抽象：

.. code-block:: cuda

   #include <mma.h>
   using namespace nvcuda;

   __global__ void tensor_core_example(half* A, half* B, float* C, float* D) {
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

       wmma::load_matrix_sync(a_frag, A, 16);
       wmma::load_matrix_sync(b_frag, B, 16);
       wmma::load_matrix_sync(c_frag, C, 16, wmma::mem_row_major);

       wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

       wmma::store_matrix_sync(D, d_frag, 16, wmma::mem_row_major);
   }

参考与拓展阅读
====================

- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — mma 指令参考
- CUTLASS (https://github.com/NVIDIA/cutlass) — NVIDIA 开源 GEMM 模板库
- cuBLAS / cuDNN — 通过库 API 自动调用 Tensor Core
