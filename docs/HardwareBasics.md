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

### 特殊内存模型
#### (1) 固定内存
**原理：** 也称页锁定内存，是一种特殊的CPU内存，位于CPU主板DRAM中。区别于其他的CPU内存，固定内存是一段被锁定的区域，无法由操作系统分页并换出到磁盘(Not Page Out)。

**传输过程：** 
1. 普通CPU内存：在实现主机到设备的数据传输时，需要由CUDA驱动将数据拷贝到一个特定的内存，然后从此段区域传输到GPU/CPU。产生额外的内存开销
2. 固定内存：可通过DMA引擎，经由PCIe总线传输到GPU/CPU。避免了额外的拷贝开销，提升了带宽。

**特点：**
1. 零拷贝开销。
2. 实现异步传输：是使用Cuda流实现异步传输的必要条件
3. 传输速率较慢：使用PCIe总线传输，而非GPU内存总线(如GDDR、HBM)

**使用：**
1. 申请与释放：cudaMallocHost() 与 cudaFreeHost()
2. 内存传输(普通与异步)：cudaMemcpy() 与 cudaMemcpyAsync()

#### (2) 统一内存
**原理：** 是一段单一的、系统范围的内存地址空间。可由CPU和GPU共同访问，并根据需要实现数据迁移。
1. 按需分页迁移：CPU和GPU共用内存指针；CUDA驱动会根据CPU或GPU的访问需求，将数据迁移到对应位置(按需迁移)
2. 透明化管理：无需显式调用cudaMemcpy，自动且隐式地迁移。基于MMU(内存管理单一)和页面故障(Page Fault)机制实现

**特点：**
1. 统一管理内存：主机与设备共用内存指针；谁来用，就迁移到哪里
2. 自动且隐式地数据迁移：简化编程复杂度；迁移时会产生页面故障开销
3. 性能：传输性能与数据规模、数据结构强相关
    1. 不规则或复杂的内存访问模式：性能强于普通显存
    2. 大规模连续访问：性能较差
4. 初始位置：可能位于CPU的DRAM(固定内存)或GPU内存中

**使用：**
1. 申请：cudaMallocManaged(&data, size), 其中data为CPU和GPU共用数据段
2. 传输：无需单独编程，其中一方使用数据段时自动实现
3. 释放：cudaFree(data)

### 附：DMA机制(直接内存访问)
**概念：** 是一种允许外设(GPU、网卡、硬盘等)绕过CPU，直接访问CPU特定内存并完成数据传输的机制。

#### (1) DMA的原理
**原理：** DMA相比传统数据传输，无需经由CPU执行拷贝指令和额外的加载动作(将源地址数据加载到寄存器，再拷贝到目标地址)。而是由DMA硬件(DMA引擎或控制器)介入，只需CPU初始设置后，有PCIe总线直接进行数据传输。

**GPU的DMA硬件：** DMA引擎在GPU硬件被集成为拷贝引擎(Copy Engine)。与GPU计算核心独立，可用于计算任务并行处理，是异步数据传输的核心。

CPU初始设置：将内存源地址、模板地址和数据量，提供给DMA硬件。

**优势：** 节省了CPU的计算资源，CPU可以在DMA传输时执行其他计算任务。

**缺点：** 使用PCIe总线传输数据，带宽低于GPU内部传输，适合单次大批量的数据传输。

#### (2) DMA的实现
CUDA驱动实现DMA的关键：**内存地址转换**(CPU/GPU 虚拟/物流地址)、**页锁定**

1. DMA的前提：页锁定，即固定内存
    - 寻址方式差异：CPU使用虚拟内存寻址，而GPU和DMA引起需要物理地址才能传输数据；且传输过程中源地址和目标地址必须固定不变
    - 页锁定：CUDA驱动实现。在调用cudaMallocHost申请固定内存时，CUDA驱动将CPU虚拟地址对应的物理地址内存段锁定，使这段内存无法被修改，且不能由OS换出到磁盘，保证了这段内存固定不变。
    
2. 地址转换和IOMMU：使能GPU访问固定内存
    - 内存隔离：GPU架构(内存与总线)与CPU独立，无法直接访问CPU的物理地址
    - IOMMU：能够将CPU虚拟地址转换为GPU可用的地址。CUDA驱动会获取固定内存的稳定物理地址，然后在IOMMU的页表中为GPU映射到固定内存。
    
3. 数据传输流程：通过固定内存、DMA硬件、IOMMU单元、GPU拷贝引擎，串联全流程
   
    3.1 启动传输：主机线程调用内存传输命令(cudaMemcpy或异步)时，启动DMA传输
   
    3.2 设置DMA：主机上的CUDA驱动，将传输请求写入**GPU任务队列**。传输请求包含源地址、目标地址、数据量
   
    3.3 拷贝引擎接管：GPU拷贝引擎读取到内存传输请求
   
    3.4 DMA传输：拷贝引擎通过PCIe总线执行DMA传输；过程包含CPU/GPU的内存地址转换
    - 拷贝引擎发送 IO虚拟地址请求
    - IOMMU拦截到请求，将IO虚拟地址转换为 固定内存的物理地址
    - 拷贝引擎访问指定内存地址(CPU DRAM内)，通过PCIe总线传输数据，并存入GPU内存，绕开CPU。
    
    3.5 传输完成通知：传输完成后，拷贝引擎向CPU发出中断，停止DMA传输；通常通过流完成时间通知CPU线程。

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
