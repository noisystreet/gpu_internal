========================
Vulkan Compute
========================

.. epigraph::

   Be water, my friend. Empty your mind. Be formless, shapeless — like water.

   — Bruce Lee

Vulkan 是由 Khronos Group 定义的跨平台 GPU API，提供对 GPU 硬件的底层控制。虽然 Vulkan 以图形渲染闻名，但其计算着色器（Compute Shader）也是 GPGPU 编程的重要接口。与 CUDA 不同，Vulkan 追求"明确即高效"——要求开发者精确控制所有资源的管理和同步。

Vulkan 软件栈
================

.. code-block:: text

   应用程序
       |
   Vulkan API (volk / libvulkan.so)
       |
   验证层 (Validation Layers, 调试可选)
       |
   用户态驱动 (Vulkan ICD — Installable Client Driver)
       |
   内核态驱动
       |
   GPU 硬件

SPIR-V 与着色器编译
=========================

Vulkan 使用 SPIR-V 作为中间语言（类似 CUDA 的 PTX），着色器需要预先编译为 SPIR-V 后再加载：

.. code-block:: bash

   # GLSL → SPIR-V (使用 glslangValidator 或 glslc)
   glslangValidator -V compute_shader.glsl -o compute_shader.spv
   glslc -c compute_shader.glsl -o compute_shader.spv

   # HLSL → SPIR-V
   glslangValidator -V compute_shader.hlsl -o compute_shader.spv

   # 查看 SPIR-V 指令
   spirv-dis compute_shader.spv

在应用加载 SPIR-V：

.. code-block:: cpp

   // 加载预编译的 SPIR-V 到 Vulkan
   VkShaderModuleCreateInfo createInfo{};
   createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   createInfo.codeSize = sizeof(spirv_code);
   createInfo.pCode = spirv_code;

   VkShaderModule shaderModule;
   vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);

   // 创建计算管线
   VkComputePipelineCreateInfo pipelineInfo{};
   pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
   pipelineInfo.stage.module = shaderModule;
   pipelineInfo.stage.pName = "main";  // 入口函数名

   VkPipeline computePipeline;
   vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                            &pipelineInfo, nullptr, &computePipeline);

内存屏障体系
====================

Vulkan 的内存屏障体系比 CUDA 复杂得多，需要明确指定：

.. list-table::
   :header-rows: 1

   * - 屏障类型
     - 作用范围
     - CUDA 对应
   * - ``VkMemoryBarrier``
     - 所有资源（全局）
     - ``__threadfence()``
   * - ``VkBufferMemoryBarrier``
     - 指定缓冲区
     - N/A
   * - ``VkImageMemoryBarrier``
     - 指定图像/纹理
     - N/A

**管线阶段屏障示例**：

.. code-block:: cpp

   // 阶段 1: 计算着色器写入缓冲 A
   // 阶段 2: 计算着色器读取缓冲 A
   // 需要屏障确保写完成后再读

   VkBufferMemoryBarrier barrier{};
   barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
   barrier.buffer = buffer;
   barrier.size = VK_WHOLE_SIZE;
   barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

   vkCmdPipelineBarrier(
       cmd,
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 源阶段
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // 目标阶段
       0,
       0, nullptr,       // 全局屏障
       1, &barrier,      // 缓冲屏障
       0, nullptr        // 图像屏障
   );

.. note::

   CUDA 中一条 ``__syncthreads()`` 或 ``__threadfence()`` 完成的工作，
   在 Vulkan 中需要显式指定源阶段、目标阶段、访问掩码三个维度的信息。
   这也是 Vulkan 学习曲线陡峭的核心原因之一。

Push Constants
=====================

Vulkan 提供 Push Constants 机制，用于向着色器传递小量数据（通常 ≤ 128 字节），效率远高于描述符（Descriptor）：

.. code-block:: cpp

   // 管线布局中声明 push constant 范围
   VkPushConstantRange pushRange{};
   pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
   pushRange.offset = 0;
   pushRange.size = sizeof(float) * 4;  // 4 个 float

   // 提交时直接写入
   float constants[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
   vkCmdPushConstants(cmd, pipelineLayout,
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(constants), constants);

.. code-block:: glsl

   // GLSL 接收
   layout(push_constant) uniform PushConstants {
       float a;
       float b;
       float c;
       float d;
   } pc;

   void main() {
       result[gl_GlobalInvocationID.x] = data[0] * pc.a;
   }

多命令缓冲与并发提交
============================

Vulkan 支持多命令缓冲的并发构建和提交，这对于 GPU 流水线重叠至关重要：

.. code-block:: cpp

   VkCommandBuffer cmdBuffers[4];
   // 每帧创建 4 个命令缓冲
   for (int i = 0; i < 4; i++) {
       vkBeginCommandBuffer(cmdBuffers[i], &beginInfo);
       // ... 记录计算、屏障、拷贝操作 ...
       vkEndCommandBuffer(cmdBuffers[i]);
   }

   // 并发提交到同一个队列
   VkSubmitInfo submits[4];
   for (int i = 0; i < 4; i++) {
       submits[i].commandBufferCount = 1;
       submits[i].pCommandBuffers = &cmdBuffers[i];
       submits[i].signalSemaphoreCount = 1;
       submits[i].pSignalSemaphores = &semaphores[i];
   }

   // 批量提交（一次内核调用提交 4 个命令缓冲）
   vkQueueSubmit(queue, 4, submits, VK_NULL_HANDLE);

.. list-table::
   :header-rows: 1

   * - 模式
     - CUDA
     - Vulkan Compute
   * - 多操作提交
     - 多 `<<<>>>` 启动
     - 多命令缓冲批量提交
   * - 同步
     - Stream / Event
     - Semaphore / Fence / Barrier
   * - 命令复用
     - CUDA Graph（驱动缓存）
     - Secondary Command Buffer（一级缓存）
   * - 提交开销
     - ~20-50 us (单次)
     - ~1-5 us (单次)

Vulkan 与 CUDA 的性能对比
===============================

.. list-table::
   :header-rows: 1

   * - 指标
     - CUDA
     - Vulkan Compute
   * - Kernel 启动延迟
     - ~20-50 us
     - ~1-5 us
   * - 设备初始化
     - ~50 ms
     - ~100-500 ms
   * - 带宽 (H2D, 4KB)
     - ~12 GB/s
     - ~12 GB/s (无差异)
   * - 带宽 (GEMM, 大矩阵)
     - 接近硬件极限
     - 接近硬件极限（同等优化）
   * - 内存分配开销
     - ~1-5 us (cudaMalloc)
     - ~5-20 us (vkAllocateMemory)
   * - 同步开销 (时间点)
     - ~3-5 us (cudaEvent)
     - ~1-2 us (vkGetQueryPoolResults)

**主要性能优势方向**:

- Vulkan 在短生命周期、频繁启动的 Compute Kernel 场景有显著优势（低至 1/10 的启动延迟）
- CUDA 在批量数学运算（GEMM、卷积）上与 Vulkan 持平，因为实际计算时间远大于启动开销
- CUDA 开发效率远高于 Vulkan，对于计算为主的应用，CUDA 通常是更务实的选择

选择 Vulkan Compute 的考量
===============================

**与 CUDA 的设计哲学对比**：

.. list-table::
   :header-rows: 1

   * - 特性
     - CUDA
     - Vulkan Compute
   * - 厂商绑定
     - 仅 NVIDIA
     - 跨厂商（NVIDIA, AMD, Intel, Apple）
   * - 抽象层次
     - 较高（运行时管理）
     - 极低（显式控制所有资源）
   * - 着色器语言
     - CUDA C++
     - GLSL / HLSL / SPIR-V
   * - 内存管理
     - 自动（cudaMalloc）
     - 完全手动（VkDeviceMemory）
   * - 同步
     - 隐式（cudaDeviceSynchronize）
     - 显式（Fence, Semaphore, Barrier）
   * - 初始化开销
     - 低（几行代码）
     - 高（数百行样板代码）

计算着色器示例
===================

**Step 1: 着色器代码 (GLSL)**

.. code-block:: glsl
   :linenos:

   #version 450
   layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

   layout(binding = 0) buffer Input  { float data[]; };
   layout(binding = 1) buffer Output { float result[]; };

   void main() {
       uint idx = gl_GlobalInvocationID.x;
       result[idx] = data[idx] * 2.0f;
   }

**Step 2: Vulkan 应用关键步骤**

.. code-block:: cpp
   :linenos:

   // 1. 创建实例 (VkInstance)
   // 2. 选择物理设备 (VkPhysicalDevice)
   // 3. 创建逻辑设备 (VkDevice) 并获取计算队列
   // 4. 创建缓冲区 (VkBuffer)
   // 5. 分配并绑定设备内存 (VkDeviceMemory)
   // 6. 创建描述符集布局 (VkDescriptorSetLayout)
   // 7. 创建计算管线 (VkPipeline) — 编译 SPIR-V

   // 8. 创建命令缓冲区
   VkCommandBuffer cmd;
   vkBeginCommandBuffer(cmd, &begin_info);

   // 绑定描述符集
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   // 分派计算任务
   vkCmdDispatch(cmd, groupCountX, 1, 1);

   vkEndCommandBuffer(cmd);

   // 9. 提交到队列
   VkSubmitInfo submit = {};
   submit.commandBufferCount = 1;
   submit.pCommandBuffers = &cmd;
   vkQueueSubmit(queue, 1, &submit, fence);

   // 10. 等待完成
   vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

Vulkan 中的内存屏障
=========================

Vulkan 要求开发者显式指定内存访问顺序：

.. code-block:: cpp
   :linenos:

   VkMemoryBarrier barrier{};
   barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
   barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

   vkCmdPipelineBarrier(
       cmd,
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // srcStageMask
       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dstStageMask
       0,
       1, &barrier,
       0, nullptr,
       0, nullptr
   );

工作组（Work Group）与 CUDA 对应关系
===========================================

.. list-table::
   :header-rows: 1

   * - CUDA 概念
     - Vulkan Compute 对应
   * - Grid
     - 分派调用 (vkCmdDispatch)
   * - Thread Block
     - Local Work Group
   * - Thread
     - Local Invocation
   * - Warp
     - Subgroup (大小取决于 GPU 厂商)
   * - Shared Memory
     - 工作组共享内存 (``shared`` GLSL 关键字)
   * - __syncthreads()
     - ``barrier()`` (GLSL)

**Vulkan Subgroup**:
    Vulkan 提供 subgroup 操作，类似 CUDA warp 级原语：

    .. code-block:: glsl

       #extension GL_KHR_shader_subgroup_arithmetic : enable

       void main() {
           float val = data[gl_GlobalInvocationID.x];
           // subgroup 内求和
           float sum = subgroupAdd(val);
           if (gl_SubgroupInvocationID == 0) {
               result[gl_WorkGroupID.x] = sum;
           }
       }

优势
=======
- 跨厂商支持
- 极低的 API 开销（适合实时应用）
- 与图形渲染紧密集成
- 精确的资源控制

**劣势**:
- 开发效率低（大量样板代码）
- 没有 CUDA 级别的数学库支持
- 生态不如 CUDA 成熟
- 调试工具不如 NVIDIA Nsight 完善

参考与拓展阅读
====================

- Vulkan 1.3 Specification (https://registry.khronos.org/vulkan/specs/1.3/html/) — Vulkan 1.3 规范中计算着色器的完整章节
- CUDA C++ Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — 与 CUDA 计算模型的对比参考
- Programming Massively Parallel Processors - Kirk & Hwu (4th ed.) — 第 14 章介绍 OpenCL/Vulkan 计算
