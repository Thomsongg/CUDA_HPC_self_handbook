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

## 4 CUDA编程实战

### 4.1 矩阵转置Transpose

#### 4.1.1 朴素解法 naive

naive解法简单粗暴：将矩阵索引直接转置

```cpp
__global void transpose_naive(float *input, float *output, int N)
{
    int idx = thread.Idx.x + blockDim.x * blockIdx.x;
    int idy = thread.Idx.y + blockDim.y * blockIdx.y;
    output[idx * M + idy] = input[idy * N + idx];
}
```

问题也显而易见：全程使用全局变量，数据读取和处理效率极差。

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


#### 4.1.3 Swizzling 技术

上述代码，无论是全局变量还是共享内存法，都存在一个共同的问题：内存访问不连续。

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
