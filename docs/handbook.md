# CUDA C++学习笔记

本文档主要以个人的学习笔记、心得体会为主，聚焦以下几个方面：

* GPU硬件架构，SM和SP，与CPU架构的不同
* CUDA抽象硬件层，线程、Warp、线程块、网格之间的关系，SIMT机制
* CUDA编程实战，经典案例与算子，Nsight Compute性能分析
* AI Infra基础知识，分布式存储
* AI大模型基础

## 1 引言

## 2 GPU硬件架构

## 3 CUDA基础知识

### 3.1 CUDA基本框架

网格 - 线程块 - 线程

### 3.2 CUDA流 & 并发模式

**特点：**

- 顺序性：同一流中的操作按顺序执行
- 并发性：不同流中的操作可以并发执行
- 独立性：不同流之间默认相互独立

**使用场景：**

- 重叠主机代码与设备kernel，异步执行
- 并发执行多个独立的kernel
- 流水线处理，同一流按顺序执行，不同流并发执行
- 多GPU编程

**关键步骤：**

1. 创建流 cudaStreamCreate(&stream[i])
2. 等待步骤完成后启动流 cudaStreamWaitEvent(stream[i], process_done, 0)
3. 同步流 cudaStreamSynchronize(stream[i])

#### 3.2.1 任务依赖与事件同步【基础】

**任务依赖管理：**
默认情况下，同一流中的操作按顺序执行，不同流中的操作则可能并发执行。但如果没有正确管理依赖关系，可能会出现数据竞争。我们可以使用事件（cudaEvent_t）来同步不同流中的操作。

**事件同步机制：**
事件是CUDA中用于同步的重要工具。我们可以记录事件到流中，然后等待事件发生。事件可以用于测量时间，也可以用于流之间的同步。

#### 3.2.2 异步执行与多流流水线【进阶】

重要概念：固定内存、内存传输重叠、流式回调函数、资源竞争

**核心思想：** 内存分若干块，每一块异步执行kernel

通过将计算和内存传输重叠，可以隐藏内存传输延迟。典型的方法是将数据分块，使用多个流，在每个流中依次执行：主机到设备的内存传输、内核执行、设备到主机的内存传输。这样，当其中一个流在执行内核时，另一个流可以进行内存传输。

**步骤：**

* 主机分配固定内存: cudaMallocHost
* 内存分块(chunk)，每块由一个cuda流控制：
  * 创建流
  * 内存异步传输 cudaMemcpyAsync
  * 流式执行核函数
  * 【可选】添加回调函数 cudaStreamAddCallback, 继续执行其他核函数
  * 等待流同步 cudaStreamSynchronize
* 清理固定内存 cudaFreeHost 和GPU内存

##### 固定内存

固定内存是主机内存的一种，它不会被操作系统分页并交换到磁盘上，因此设备可以直接通过DMA（直接内存访问）访问固定内存，而不需要CPU的参与。

特点：

* 分配和释放成本较高
* 传输速度比可分页内存快
* 允许与设备执行并行传输

固定内存的分配 cudaMallocHost：

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));
```

#### 3.2.3 优先级流管理【进阶】

可以创建具有不同优先级的流。高优先级流中的操作可以优先得到调度。使用cudaStreamCreateWithPriority创建优先级流。

## 4 CUDA编程实战

### 4.1 矩阵转置Transpose

#### 4.1.1 朴素解法 naive

naive解法简单直接：将矩阵索引直接转置

```cpp
__global void transpose_naive(float *input, float *output, int N)
{
    int idx = thread.Idx.x + blockDim.x * blockIdx.x;
    int idy = thread.Idx.y + blockDim.y * blockIdx.y;
    if (idy < M && idx < N)
    {
        output[idx * M + idy] = input[idy * N + idx];
    }
}
```

经过Nsight Compute性能分析，这种解法在(1)GPU吞吐量Throughput和(2)内存占用率Occupancy上，效率很低。

##### 核心问题：非连续的内存访问

朴素解法的问题根源在于：对一个Warp内的所有线程，内存的读取和写入操作并不都是连续的。
我们看如下例子。假设在block0的Warp0内，数据维度(M x N) 1024 * 1024，按threadIdx.x(即列向量)的维度，观察相邻的线程：

```cpp
读取&写入事务:
thread0: 读取 input[(0 * 1024) + 0] = input[0] -> 写入 output[(0 * 1024) + 0] = output[0]
thread1: 读取 input[(0 * 1024) + 1] = input[1] -> 写入 output[(1 * 1024) + 0] = output[1024]  // 跨步 1024
thread2: 读取 input[(0 * 1024) + 2] = input[2] -> 写入 output[(2 * 1024) + 0] = output[2048]  // 跨步 1024
// 读取是连续内存，但写入时产生1024的跨步
```

相邻线程的非连续访问，会严重降低并行计算效率！
优化思路：使用共享内存，每次进行读取操作(从input转移数据)时，先对数据进行转置，再写入output。这样保证了数据的连续性，**能够同时合并读取和写入操作**！

#### 4.1.2 基于共享内存的GPU解法

采用共享内存，作为数据中转。

1. 线程块开辟出一个共享内存 smem[BLOCK_SIZE][BLOCK_SIZE] 作为数据中转站 -> 可加1单位的padding，有效缓解bank conflict
2. 块内每个线程转移一个数据 -> 可通过float4向量化方法，一次读取4个数据，减少内存处理事务

```cpp
__global void transpose_shared(float *input, float *output, int N)
{
    __shared__ float smeme[BLOCK_SIZE][BLOCK_SIZE + 1] // 1字节的padding
    int idx = thread.Idx.x + BLOCK_SIZE * blockIdx.x;
    int idy = thread.Idx.y + BLOCK_SIZE * blockIdx.y;
    if (idy < M && idx < N)
    {
        smem[threadIdx.y][threadIdx.x] = input[idy * N + idx];
    }

    // 将转置后的共享内存数据，按新的索引存放到结果中
    // 注意：矩阵B中的线程索引，按线程块转置后处理，块内索引不变
    // 转移后矩阵B 每一行的元素为 M
    int new_idx = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int new_idy = threadIdx.y + BLOCK_SIZE * blockIdx.x;
    if (new_idx < M && new_idy < N)
    {
        output[new_idy * M + new_idx] = smem[threadIdx.x][threadIdx.y];
    }
}
```

核心：**共享内存充当了数据重排的临时缓冲区**，在取出时隐式地进行了转置(通过索引倒置)，同时合并了读取和写入操作，有效解决了内存访问不连续的问题：

```cpp
读取&写入事务:
thread0: 读取 input[(0 * 1024) + 0] = input[0] -> 存入smem[0][0] -> 取出smem[0][0] -> 写入 output[(0 * BLOCK_SIZE) + 0] = output[0]
thread1: 读取 input[(0 * 1024) + 1] = input[1] -> 存入smem[0][1] -> 取出smem[1][0] -> 写入 output[(0 * BLOCK_SIZE) + 1] = output[1]  // 未产生跨步
thread2: 读取 input[(0 * 1024) + 2] = input[2] -> 存入smem[0][2] -> 取出smem[1][0] -> 写入 output[(0 * BLOCK_SIZE) + 2] = output[2]  // 未产生跨步
// 读取和写入，均为连续操作！！！
```

#### 4.1.3 Swizzling 技术

注意下面的代码：

`output[x * height + y] = input[y * width + x]`

每次都是 (1)按行读取input (2)按列写入output

`thread[0][0], thread[0][1], thread[0][2]: 读取input[0], input[1], input[2]，是连续内存`

`thread[0][0], thread[0][1], thread[0][2]: 访问output[0], input[height], input[2 * height]，间隔height`

这样导致每个块内的相邻线程，访问的是不连续的内存，可能产生以下问题：

1. 无法一次性读取全部数据，需要额外的内存处理事务，降低内存吞吐量
2. 共享内存可能产生bank conflict，严重影响了内存处理效率

基于此，推荐使用Swizzling技术，将矩阵内每个元素通过数学方法映射到新的索引，保证访问的内存是连续的。

##### 4.1.3.1 XOR Swizzling 异或映射

原理：通过threadIdx.x ^ threadIdx.y 异或操作，**将连续的线程打散，避免多个线程同时访问一个bank**，可有效避免bank conflict

```cpp
template<int BLOCK_SIZE>
__global__ void transpose_xor_swizzling(float *input, float *output, int M, int N)
{
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE];    // 不需要padding
    int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int idy = threadIdx.y + BLOCK_SIZE * blockIdx.y;
    if (idx < N && idy < M)
    {
        // 对共享内存的列 执行异或
        smem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[idy * N + idx];
    }

    __syncthreads();

    int new_idx = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int new_idy = threadIdx.y + BLOCK_SIZE * blockIdx.x;
    if (new_idx < M && new_idy < N)
    {
        output[new_idy * M + new_idx] = smem[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}
```

##### 4.1.3.2 XOR Swizzling 异或映射

原理：选择一个与BLOCK_SIZE互质的因子 SWIZZLE_FACTOR, a * FACTOR + b 会形成**完全映射**，即与 BLOCK_SIZE 交错。

方法：使用互质因子，使得

`存入时 smem[ty][tx] -> 输出时 smem[ty'][tx'], 其中 ty' = (tx * SWIZZLE_FACTOR + ty), tx' = (tx + ty)`

```cpp
存入时: smem[ty][tx] = input[idy * N + idx];
取出时: output[idx * M + idy] = smem[ty'][tx']
```

##### 4.1.3.3 其他Swizzling 映射

## 5 AI Infra基础
