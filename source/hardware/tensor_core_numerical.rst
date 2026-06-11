========================
Tensor Core 数值行为
========================

Tensor Core 的浮点运算**不遵循 IEEE 754 标准**，其内部实现采用了多种与标准不同的优化策略（Fasi 等, 2021; Sun 等, 2022）。理解这些数值特性对于科学计算和高精度场景至关重要。

关键数值特性
================

.. list-table::
   :header-rows: 1

   * - 特性
     - 描述
     - 与 IEEE 754 的差异
   * - 乘积精度
     - A×B 的乘法结果以至少单精度（FP32）计算
     - 符合 NVIDIA 文档
   * - 累加精度
     - 中间累加以至少单精度（FP32）执行
     - 符合 NVIDIA 文档
   * - 舍入模式
     - RTZ（Round-to-Zero，截断）
     - IEEE 754 默认为 RNE
   * - 保护位（guard bits）
     - 无保护位，直接截断
     - IEEE 754 RTZ 需要保护位
   * - 中间和归一化
     - 不归一化（intermediate sums not normalized）
     - IEEE 754 要求归一化
   * - 加法器对齐
     - 尾数仅按最大幅度一次对齐
     - IEEE 754 逐对对齐
   * - 非规格化数
     - 清零（flush-to-zero）
     - IEEE 754 支持非规格化数
   * - NaN/Inf 处理
     - 符合 IEEE 754 规范
     - 一致

加法器微架构分析
====================

Fasi 等人（2021）通过精心设计的数值实验，揭示了 Tensor Core 内部的加法器结构：

.. code-block:: text

   d11 = a11·b11 + a12·b21 + a13·b31 + a14·b41 + c11

   在 Tensor Core 内部的加法树中：

   5 个乘积项同时进入多操作数加法器
          │
          ↓
   尾数对齐：基于 5 项中的最大指数一次性对齐
   （而非 IEEE 754 的逐对对齐）
          │
          ↓
   对齐后的尾数直接相加（不归一化）
   使用 3 个进位位处理超范围结果
          │
          ↓
   结果截断（Round-to-Zero）
          │
          ↓
   输出最终累加值

设计的后果
==============

1. **非单调性**：由于中间和不归一化，多操作数加法可能出现非单调性——即增加一项的值可能反而使结果变小。Fasi 等人指出，这在科学计算中需要特别注意。

2. **舍入误差模型**：不同于 IEEE 754 的标准舍入误差分析，Tensor Core 的截断舍入导致误差分布非对称（偏向零），且不满足标准浮点分析中的常见假设。

3. **运算顺序无关**：因为所有乘积项的对齐基于最大指数一次完成，各项的排列顺序不影响最终结果——这与 IEEE 754 的非结合性形成鲜明对比。

精度恢复技术
================

研究人员提出了多种方法来恢复 Tensor Core 的精度损失（Ootomo 和 Yokota, 2022; Markidis 等, 2018）：

.. code-block:: text

   Markidis 方法（2018）：
   1. 使用 Tensor Core 计算低精度结果
   2. 计算残差矩阵 R = D - C - A·B（使用 CUDA Core 以 FP32 计算）
   3. 用 Tensor Core 在残差上修正结果

   Ootomo-Yokota 改进（2022）：
   1. 识别 Tensor Core 的舍入误差来源是 RTZ 而非 IEEE-754
   2. 显式补偿 RTZ 误差项
   3. 在保持 Tensor Core 高吞吐的同时恢复单精度精度
   4. 效果：接近 FP32 精度，超过 FP32 CUDA Core 的吞吐

参考与拓展阅读
====================

- Numerical Behavior of NVIDIA Tensor Cores (https://peerj.com/articles/cs-330/) — Fasi 等, 2021
- Recovering Single Precision Accuracy from Tensor Cores — Ootomo & Yokota, 2022
- NVIDIA Tensor Core Programmability, Performance & Precision — Markidis 等, 2018
