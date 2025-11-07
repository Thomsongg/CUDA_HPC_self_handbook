# CUDA_HPC_self_handbook
CUDA C++高性能计算学习笔记，包含理论知识与源码

## 1 硬件原理
### 1.1 GPU架构知识：

1. 了解SM（Streaming Multiprocessor）的基本结构：包括CUDA Core、Tensor Core（如果涉及）、内存层次（寄存器、共享内存、L1/L2缓存、常量内存、纹理内存等）。

2. 了解线程调度的基本单位：Warp（32个线程）的概念，以及Warp调度器如何工作。

3. 了解不同GPU架构的特性（如Fermi、Kepler、Maxwell、Pascal、Volta、Turing、Ampere等），特别是你目标公司可能使用的架构。

### 1.2 GPU内存层次与缓存：

1. 理解全局内存、共享内存、常量内存、纹理内存、本地内存等的特性和使用场景。

2. 理解缓存层次结构（L1、L2），以及如何利用缓存提高性能。

3. 理解合并内存访问（Coalesced Memory Access）的概念，以及如何通过调整内存访问模式来减少缓存miss。

### 1.3 内存访问优化：

1. 了解Bank Conflict（共享内存的bank冲突）及其避免方法。

2. 了解如何通过数据布局优化（例如Structure of Arrays vs Array of Structures）来提高内存访问效率。

3. 了解预取（Prefetching）技术，以及如何利用它来隐藏内存延迟。

### 1.4 并行性与资源限制：

1. 了解Occupancy（占用率）的概念，以及如何通过调整线程块大小和资源使用来提高Occupancy。

2. 了解寄存器压力对性能的影响，以及如何减少寄存器使用（例如使用启动边界（launch bounds）或调整编译器优化标志）。

### 1.4 指令级优化：

1. 了解指令吞吐量（例如，单精度浮点、双精度浮点、整数运算的吞吐量）。

2. 了解控制流分歧（Branch Divergence）对性能的影响，以及如何避免。

3. 了解使用内建函数（intrinsics）和汇编代码进行极优化（这部分可能不是必须，但了解有助于理解性能瓶颈）。

### 1.5 性能分析工具：

1. 熟练使用Nsight Systems（宏观性能分析）和Nsight Compute（微观内核分析）等工具。

2. 能够通过工具识别性能瓶颈，例如内存带宽瓶颈、计算瓶颈、指令吞吐瓶颈等。

### 1.6 缓存miss的排查与优化：

1. 使用Nsight Compute可以检测到缓存miss（例如L1/L2缓存未命中率）。

2. 了解如何通过调整内存访问模式、数据块大小、数据布局等来减少缓存miss。

3. 理解时间局部性和空间局部性，并据此优化代码。

### 1.7 数据结构与算法选择：

根据访问模式选择数据结构。例如，连续访问的数组（O(1)随机访问）可能比链表（O(n)访问）在GPU上更高效，因为数组更容易实现合并访问和缓存友好。

现实案例：例如，在GPU上实现哈希表时，可能会因为冲突和随机访问导致缓存miss较高，这时可以考虑使用开放寻址法而不是链地址法，因为开放寻址法更连续。

### 1.8 跨架构兼容性：

了解不同GPU架构的差异，并编写能够适应多种架构的代码（例如使用CUDA的PTX和编译选项）。

### 1.9 其他硬件特性：

了解GPU的PCIe总线传输、NVLink（多GPU通信）、统一内存（Unified Memory）等。

## 2 CUDA软件编程
### 2.1 CUDA核心概念
1. CUDA核心架构：线程-线程束Warp-线程块-网格、SIMT单指令多线程、CUDA流与并发执行模式

### 2.2 经典案例
1. 【入门】向量加法：了解CUDA C++基本编程要素，包含kernel核函数、__global__关键字、kernel网格大小与线程块大小定义、GPU内存申请(CudaMalloc)、CPU-GPU内存转移(CudaMemcpy)、GPU内存释放(CudaFree)等基本用法。
2. 【入门】激活函数：ReLU、Sigmoid、Tanh
3. 【进阶】矩阵转置Transpose：朴素转置(容易引起bank conflict)、使用共享内存的矩阵转置，bank conflict的概念、原理、如何检测(ncu指令)、规避方式(padding、调整共享内存访问模式、float4减少访问事务)
4. 【进阶】归约运算Reduction：朴素归约、折半归约、Warp Shuffle归约法(2种)、使用float4优化。**重要，必须掌握！！！**
5. 【进阶】矩阵乘法GEMM：朴素矩阵乘、利用k分割思想的共享内存矩阵乘(block_tile)、使用寄存器的矩阵乘(thread_tile)。**重要，必须掌握！！！**
6. 【进阶】Softmax算子：使用归约运算的算子实现
7. 【高级】其他大模型算子：卷积(Convolution)、池化(Pooling)、归一(Normalization)
8. 【必修】性能评估：使用Nsight Compute评估程序性能，针对性优化；对于经典问题，如bank confilict、warp divergence、partional wave等，如何鉴别、如何规避与优化。
9. 【选修】优秀开源项目：使用CUDA C++优化其他算子
