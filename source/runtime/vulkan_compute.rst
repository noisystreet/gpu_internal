========================
Vulkan Compute
========================

.. epigraph::

   Be water, my friend. Empty your mind. Be formless, shapeless — like water.

   — Bruce Lee, 武术家、演员

Vulkan 是由 Khronos Group 定义的跨平台 GPU API，提供对 GPU 硬件的底层控制。虽然 Vulkan 以图形渲染闻名，但其计算着色器（Compute Shader）也是 GPGPU 编程的重要接口。

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

与 CUDA 的设计哲学差异
==============================

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

选择 Vulkan Compute 的考量
===============================

**优势**:
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
