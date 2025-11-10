# GPU硬件基础

本章节聚焦GPU硬件与CUDA架构基础知识。建议与CUDA编程共同学习，在底层加深对GPU编程的理解，明确CUDA高性能计算和优化的目的、方向。

## 开场寄语

随着各种大模型的横空出世，AI不再停留在基本的应用，而是成为了新的科技生态，从底层到上层共同发展。越来越多的企业尝试构建自己的AI生态、训练专属的模型以应对复杂的业务、打破其他厂家的技术壁垒，这时AI的底层构建与部署就极为必要，催生出AI Infra的迅猛发展。

而基于底层硬件进行高性能计算，尽可能发挥硬件的价值、减少性能浪费而产生的额外成本，在AI生态中的重要性越来越明显，其应用在智能驾驶、工业自动化、量化等诸多领域。

基于计算硬件的高性能优化软件开发（如CUDA、昇腾CANN等），需要开发者具备扎实的硬件基础。这也是硬件、嵌入式开发工程师进入AI领域的好机会。

有一句话想送给大家：

> 如果有一列呈指数级加速的火车，你唯一要做的，就是**跳上去**。 一旦上车，**所有问题都可以在途中解决**。因此，试图预测一列每秒都在加速的火车会驶向何方，然后妄图在某个路口拦截它，这是徒劳的。
> ————黄仁勋最新专访：关于投资OpenAI、AI泡沫、ASIC的竞争

## 1 GPU硬件基础知识

首先让我们来看一个问题：GOU和CPU有什么区别，我们为什么要选用GPU作为高性能并行计算的硬件？

CPU和GPU的硬件架构不同，导致二者的拥有不同的性能和擅长的使用场景。

1. CPU的层级架构，包含计算单元、指令处理、L1缓存、L2缓存和内存；GPU则包含大量的计算单元，剩下的空间大部分留给了显存，其他单元很少。
2. 硬件架构的不同导致了不同的性能: CPU可用处理更复杂的工作流程，GPU适合处理简单且高强度、大规模的计算任务；CPU的算数逻辑单元和浮点单元少，但是CPU Core中的ALU和FPU能力更强，且CPU的缓存内存更多
3. 二者适合处理不同类型的工作负载。CPU通常适用于多任务和快速串行处理，GPU适用于大规模并行架构、高计算吞吐量的场景，浮点计算能力强大。

基于此，我们接下来详细介绍GPU的硬件结构。

### 1.1 GPU硬件架构

GPU架构演进：Fermi -> Kepler -> Volta -> Ampere -> Ada Lovelace(40系显卡)
GPU硬件架构（以Volta架构为例）：
GPU的核心：SM(多处理器)，负责线程调度与内存分配的核心，包含以下核心组件：

1. 计算单元：Cuda Cores(SP), 执行基本算术运算；Tensor Cores, 执行矩阵运算；FP64 Cores, 执行双精度浮点运算
2. 存储资源：Register File, 用于存储线程上下文；Shared Memory, Block内线程共享；L1 Cache/Texture Cache, 缓存子系统
3. 控制单元：Warp Scheduler, 管理warp调度；Dispatch Unit, 指令分发；其他特殊功能单元，执行特殊运算。

### 1.2 GPU内存层次与缓存
1. SM内：寄存器 -> 共享内存 -> L1缓存
2. 整个GPU：L2缓存 -> 全局内存、常量内存及其他

#### 1.2.1 寄存器 (Register File)
GPU内存的最核心区域，由SM分配，每个线程独享一定空间的寄存器。
**特点：** 高带宽，数据传输和内存访问效率最高的部分。寄存器空间被分为32个bank，与一个Warp的32个线程相对应。

#### 1.2.2 L1缓存 (L1 Cache)
L1缓存是SM片上的缓存区，提供SM片内到主机内存(RAM)的加速和数据缓冲功能。
**特点：** 寄存器**溢出区**。如果寄存器内活动的数据量过大，超过了寄存器总体承载量，溢出的部分则会由L1缓存承载，并被组装为bank。由于L1缓存带宽低于寄存器，故应尽量避免寄存器过度溢出。

#### 1.2.3 共享内存 (Shared Memory)
SM片上的内存。由SM分配给若干个线程块，同一线程块内的线程共享，实现线程间通信。
**特点：** 所有数据可被块内任一线程访问。内存划分与寄存器类同，都由32个bank组成。
用法：需使用关键字 __shared__ 显式声明指定内存空间。
由于块内所有线程均可读取和写入，为防止数据被异常篡改，可使用 __syncthreads() 函数 对块内所有线程进行同步。

#### 【重要】Bank Conflict 问题
##### (1) 共享内存&寄存器的 32-bank 工作原理
- 内存空间被划分为32个bank，每个bank在列排布上都储存一个4byte的数
- 每个Warp在若干行上操作。理想状况下，Warp内的每个线程都连续的不同列的数据
- 单个bank在每个时钟周期内，只能执行一次内存访问事务。

对于32位数据，其地址与bank_id有如下映射关系：
`bank_id = (byte_addr / 4) % 32`

进而可得：每隔32个数据，就会在同一个bank的不同行内存放。

##### (2) Bank Conflict的产生机制
Bank Conflict的定义：当一个Warp的不同线程，访问了同一个bank，即此bank的不同行方向上的数据，就会产生bank conflict。
正常情况(无conflict)：
```cpp
__shared__ float smem[128];

// 同一个Warp内读取
float val = smem[threadIdx.x];  // 线程0-31 访问smem[0]-smem[31], 即bank0 - bank31，不存在同一bank
```
我们来看有bank conflict的例子：
###### 1. 多对一映射
```cpp
__shared__ float smem[32][32]; // 寄存器同理，  float reg_mem[32][32];

// 同一个Warp内读取
float val = smem[threadIdx.x][0];  // 线程0-31 访问smem[1][0] - smem[31][0], 即所有32个线程均访问 bank0, 产生32way bank conflict
```
多个线程访问同一个bank。n个线程访问一个bank的n行，也称为 n-way bank conflict。需要n个时钟周期，完成全部的内存访问，效率极差！

###### 2. 跨步长访问
```cpp
__shared__ float smem[128];

// 同一个Warp内读取
float val = smem[threadIdx.x * 2];  // 步长为2；线程0和16 访问bank0, 线程1和17 访问bank1... 产生2way bank conflict
```
同一维度的跨步长访问时，每隔32个数据，会访问同一个bank。

##### (3) Bank Conflict的监测

##### (4) Bank Conflict的规避与优化
###### 1. 数据布局优化
**方法： 填充Padding 或 转置数据布局**
<1> 【基础】填充Padding：
行访问易出现bank conflict时，对列进行填充：
```cpp
__shared__ float smem[32][33];  // 列方向+1的padding

// 此时按行方向读取，不产生bank conflict
float val = smem[threadIdx.x][0];

// 转置数据布局
__shared__ float smem[33][32]; // 行方向加padding

// 按列方向访问，不产生bank conflict
float val = smem[0][threadIdx.x];
```
按行方向访问，就在列方向上加Padding, 使数据访问地址产生错位。

<2> 【进阶】数据布局优化：调整数据结构，不同结构体实例访问(AOS) 转变为 同一实例的对象访问连续的内存(SOA)
优化前，使用不同结构体实例访问，每次间隔一定的步长(为结构体对齐的空间大小)
```cpp
struct Particles
{
    float x,y,z;
    float vx,vy,vz;
    float mass;
}
// 相邻线程访问不连续的内存
// thread0 访问 particles[0].x
// thread1 访问 particles[1].x
// 相邻线程间隔sizeof(Particles)，即 40个字节
particles[tid].x += particles[tid].vx * dt;
```

优化后，调整结构体，使用同一结构体实例，合并内存访问：
```cpp
struct Particles
{
    float *x, *y, *z;
    float *vx, *vy, *vz;
    float mass;
}
// 使用单个结构体对象，其成员按连续的内存空间存储
// 相邻线程访问连续的内存
// thread0 访问 particles.x[0]
// thread1 访问 particles.x[1]
particles.x[tid] += particles.vx[tid] * dt;
```

###### 2. 访问模式优化
经典案例：矩阵转置
方法：加Padding；使用映射Swizzling。
```cpp
// 加载到共享内存 - 无冲突
tile[threadIdx.y][threadIdx.x] = input[y * width + x];
__syncthreads();
    
// 转置写入 - 有bank conflict!
output[x * width + y] = tile[threadIdx.x][threadIdx.y];
```

###### 3. 广播机制
当有不同线程需要访问相同的数据时，可以使用广播机制，减少内存访问事务次数。
```cpp
__shared__ float smem[128];

// 不同线程访问相同数据
if (threadIdx.x == 0)
{
    float common_val = smem[0];
}
__syncthreads();
```

###### 4. bank友好的归约操作
```cpp
__global__ void reduction_kernel(float *input, float *output) {
    __shared__ float sdata[256];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    
    // bank友好的归约
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 最后warp内的归约 - 无bank conflict
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

【重要】Bank Conflict优化的要点：
1. 确保全局内存访问是合并的。如果是非连续内存访问，要使用共享内存/寄存器进行数据处理。
2. 优先调整数据布局与访问模式

## 2 CUDA架构层详解

## 3 基于硬件的高性能优化

### 3.1 内存访问优化

### 3.2 并行性与资源限制

### 3.3 指令级优化

### 3.4 常用优化技术举例

#### 3.4.1 Roofline模型

#### 3.4.2 合并内存访问(Coalesced Memory Access)

#### 3.4.3 使用共享内存(Shared Memory)

#### 3.4.4 延迟隐藏(Latency Hiding)

#### 3.4.5 Warp级优化

#### 3.4.6 循环并行化(Loop Parallelization)
