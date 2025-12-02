# CUDA C++学习笔记

本文档主要以个人的学习笔记、心得体会为主，聚焦以下几个方面：

* CUDA抽象硬件层，线程、Warp、线程块、网格之间的关系，SIMT机制
* CUDA编程实战，经典案例与算子，Nsight Compute性能分析
* AI Infra基础知识，分布式存储
* *AI大模型基础*

## 0 引言

## 1 CUDA基础知识

### 1.1 CUDA基本框架

网格 - 线程块 - 线程

## 2 CUDA流 & 并发模式

### 2.1 了解 CUDA Stream
**CUDA Stream(流)：** CUDA Stream本质是一个在GPU执行的**FIFO的任务队列**。在同一个Stream添加的操作（如Kernel启动、内存拷贝），会按之前添加的的顺序执行。

**并发机制：** 不同 Stream 之间的操作，可能并发执行，也可能乱序执行；取决于GPU硬件资源和依赖关系。

**CUDA Stream 的类型：**
- **默认流：** 未显式创建的 Stream，在不指定显式流执行操作时会在默认流中执行（如调用 Kernel<<<>>>() 或 cudaMemcpy 时，不指定 Stream 参数）
- **并发流：** 显式创建的 Stream

### 2.2 并发的第一步：异步传输与页锁定内存
**异步传输的目的：** 默认的内存传输操作 cudaMemcpy()是同步的，会阻塞CPU线程执行，直到完成GPU内存数据拷贝。为了不影响GPU和CPU的运行，我们采用异步传输机制。

**方法：** 申请页锁定内存 + 异步传输，使用 cudaMemcpyAsync() 实现异步传输。

**示例：** 

```cpp
// 申请页锁定内存和GPU内存
cudaMallocHost(&h_pinned, N * sizeof(float));
cudaMalloc(&d_data, N * sizeof(float));

// 数据初始化
do_initialization(h_pinned);

// 申请 CUDA Stream
cudaStream_t streaml
cudaStreamCreate(&stream);

// 异步传输
cudaMemcpyAsync(d_data, h_pinned, N * sizeof(float), cudaMemcpyHostToDevice, stream);

// 启动 Kernel, 其中 sharedMem 可设置为 0
// 调用后立即返回，不影响 CPU 执行，GPU 在未来某个时刻执行
myKernel<<<gridSize, blockSize, sharedMem, stream>>>(...);
```

**异步传输的核心：** 页锁定内存(Pinned Memory)
- 为什么使用页锁定内存：GPU 的 DMA 引擎，需要一个稳定的、不会分页且交换到磁盘中的内存，由 CUDA 驱动先从可分页内存中**同步地**拷贝到一个临时的、驱动程序管理的页锁定内存中后，才能发起异步 DMA 传输。
- 如何使用： cudaHostAlloc() 或 cudaMallocHost() 分配页锁定内存， cudaFreeHost() 来释放

### 2.3 同步机制：如何“等待”
颗粒度从粗到细：Device 同步 > Stream 同步 > Event 同步

1. 设备同步 `cudaDeviceSynchronize():` 阻塞 CPU 线程，直到 GPU 中所有流的所有任务执行完毕。一般用于程序末尾，所有操作完成后使用（不推荐）。
2. 流同步 `cudaStreamSynchronize():` 阻塞 CPU 线程，直到这个 Stream 的所有任务执行完毕；推荐使用。
3. 事件同步(Events)

#### 事件机制 cudaEvent
事件(Events)是实现复杂并发（流间依赖、性能分析）的最重要工具。

**事件：** 是一个标记，用来插入到 Stream 的任务队列中。当GPU执行到这里时，事件的状态更新为“已完成”。

**用法/API：** 
声明一个 Event: `cudaEvent_T event;`
- `cudaEventCreate(&event)`: 创建一个事件
- `cudaEventDestroy(event)`: 销毁一个事件
- `cudaEventRecord(event, stream)`: 将事件放入 stream 队尾（与 CPU 异步）；在 stream 队列之前的任务都完成后，再执行这个事件
- `cudaEventSynchronize(event)`: 事件同步。阻塞 CPU 线程，直到这个 event 被 GPU 触发
- `cudaStreamWaitEvent(stream2, event)`: 流间依赖（GPU 之间等待，不阻塞 CPU）。让 stream2 等待，直到 stream 中的 event 被触发时，继续执行 stream2 的后续任务。
- `cudaEventElapsedTime(&milleSeconds(float*), start, stop)`: 计算两个已完成事件 start 和 stop 之间的时间；常用于性能分析。


### 2.4 场景示例
##### 2.4.1 默认流
代码忽略，在默认流中实现。
`cudaMalloc -> myKernel<<<gridSize, blockSize>>>(...) -> cudaFree`

**问题：** 
- GPU 在 数据 HtoD 和 DtoH 都是空闲的
- CPU 在 HtoD, myKernel 调用 和 DtoH 都是阻塞的

##### 2.4.2 简单的流并发
创建多个流，每次执行 HtoD -> Kernel -> DtoH 并提交给 Stream

**优点：**
- 两个任务的流程，会在 GPU 上并发执行，抢占拷贝引擎 (DMA 资源) 和计算资源 (SM)
- 吞吐量 (Throughput) 大幅提升

代码示例：

```cpp
// Data 是预先定义的数据结构
void process_concurrent_tasks(Data* data1, Data* data2) {
    // 创建两个 Stream
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 假设已预先分配固定内存 data1->h_in, data1->h_out

    // Task1: 执行 Kernel 并提交到 Stream1
    cudaMemcpyAsync(data1->d_in, data1->h_in, N * sizeof(float), cudaMemcpyHostToDevice, stream1);
    myKernel<<<gridSize, blockSize, 0, stream1>>>(...);
    cudaMemcpyAsync(data1->d_out, data1->h_out, N * sizeof(float), cudaMemcpyDeviceToHost, stream1);
}

    // Task2: 执行 Kernel 并提交到 Stream2
    cudaMemcpyAsync(data2->d_in, data2->h_in, N * sizeof(float), cudaMemcpyHostToDevice, stream2);
    myKernel<<<gridSize, blockSize, 0, stream2>>>(...);
    cudaMemcpyAsync(data2->d_out, data2->h_out, N * sizeof(float), cudaMemcpyDeviceToHost, stream2);

    // 等待两个流执行完成
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 释放资源
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}
```

##### 2.4.3 基于事件的流间依赖
假设有两个 Kernel, 后一个依赖前一个的结果: kernel_A 和 kernel_B, B 要等待 A 完成后再执行；两个 Kernel 分属于不同的 Stream

**基本流程：**
1. kernel_A完成，输出结果 d_dataA，将事件 eventA_done 放入 streamA 队列
2. streamB 等待 eventA_done 完成后开始，传入 d_dataA 执行 kernel_b，输出结果 d_dataB 并回传到主机
3. 等待最后一个流 streamB 完成
4. 释放资源

代码示例：

```cpp
void process_with_dependency(Data* data) {
    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    // 创建 kernel_A 结束事件
    cudaEvent_T kernelA_done;
    cudaEventCreate(&kernelA_done);

    // StreamA: 执行 kernel_A, 结果输出到 d_dataA
    cudaMemcpyAsync(d_dataA, h_dataA, ..., cudaMemcpyHostToDevice, streamA);
    kernel_A<<<gridSize, blockSize, ..., streamA>>>(d_dataA, ...);

    // 【重要】记录 event 并压入 streamA
    cudaEventRecord(kernelA_done, streamA);

    // StreamB: 等待 event 完成后开始
    cudaStreamWaitEvent(streamB, kernelA_done);

    // 执行 kernel_B
    kernel_A<<<gridSize, blockSize, ..., streamA>>>(d_dataB, d_dataA, ...);
    cudaMemcpyAsync(h_dataB, d_dataB, ..., cudaMemcpyHostToDevice, streamA);

    // 流同步：只需同步最后一个流
    cudaStreamSynchronize(streamB);

    // 释放资源
    cudaEventDestroy(kernelA_done);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
}
```

##### 2.4.4 重要应用：实现“重叠(Overlap)”
待补充

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
```

```cpp
取出时: output[idx * M + idy] = smem[ty'][tx']
```


## 5 Thrust并行库
Thrust是CUDA自带的开源库，可以简化地实现大数据并行算法。适合处理大数据中的每个元素都执行相同的操作（即大数据并行）。
**特点：**
1. 高层次抽象：无需了解GPU底层原理，只需调用对应的接口(如thrust::reduce)，即实现并行算法；会自动且隐式地完成线程分配、内存申请与释放、kernel调用等底层操作。
2. 高性能：Thrust库的算法函数在后端完成了优化，会自动根据GPU架构和数据类型，智能选择对应的算法
3. 高操作性：可以自由嵌入Kernel函数和cuBLAs等库函数。

### 5.1 容器
常用容器：
1. 主机vector
2. 设备vector
3. 固定内存vector

```cpp
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/system/cuda/vector.h>

thrust::host_vector<int> h_vec = {1, 2, 3};

// 自动实现 主机->设备的传输
thrust::device_vector<int> d_vec = h_vec;

// 分配固定内存vector
thrust::cuda::vector<int> pinned_vec(100);
```

#### 容器的操作
1. 初始化容器 `thrust::device_vector<int> d_vec(100)`
2. 清空 `d_vec.clear()`
3. 获取大小 `d_vec.size()`
4. 判断是否为空 `bool isEmpty = d_vec.empty()`
5. 访问指定元素 `int ele = d_vec[3]`
6. 返回首元素 `d_vec.front()`
7. 返回尾元素 `d_vec.back()`
8. 预分配内存 `d_vec.reserve(1000)`
9. 获取容量 `size_t cap = d_vec.capacity()`

### 5.2 迭代器
生成一组值的序列
Thrust迭代器的种类：
1. 常值迭代器 constant_iterator
2. 计数迭代器 counting_iterator
3. 变换迭代器 transform_iterator
4. zip迭代器 zip_iterator


```cpp
#include<thrust/iterator/constant_iterator.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/transform_iterator.h>
#include<thrust/iterator/transform_iterator.h>

// 常值迭代器：生成常数序列(42, 42, 42...)
auto constant = thrust::make_constant_iterator(42);

// 计数迭代器：生成递增序列(0, 1, 2...)
auto counting = thrust:make_counting_iterator(0);

// 变换迭代器：对元素应用特定函数
auto transform_iter = thrust::make_transform_iterator(vec.begin(), [] __device__ (int x) { return x * x });

// zip迭代器：组合多个迭代器序列
auto zipped = thrust::make_zip_iterator(thrust::make_tuple(vec1.begin(), vec2.begin()));
```

### 5.3 应用场景 & 基本算法
#### 5.3.1 数据预处理
1. 排序(Sorting)：快速排序, thrust::sort
2. 过滤(Filtering)：筛选符合特定条件的元素, thrust::copy_if
3. 去重(Unique)：移除重复元素, 有 thrust::unique 和 thrust::unique_copy 两种常用操作。

###### thrust::unique
不改变原数据结构的大小。而是直接将唯一的元素挪到数组前端，并返回唯一数组的末尾。
如：

```cpp
int[] h_vec = {1, 1, 2, 3, 3, 5, 5, 8, 8, 9};
thrust::device_vector<int> d_vec = h_vec;
```
应用 thrust::unqiue后：

```cpp
// 返回去重子串的末尾指针
// 即 {1, 2, 3, 5, 8, 9(末尾指针), ?, ?, ?, ?}
auto unique_end = thrust::unique(d_vec.begin(), d_vec.end())
```

#### 5.3.2 数据分析与计算
1. 归约：数据归约 `thrust::reduce(vec.begin(), vec.end(), 0, operator), 其中 operator可使用lambda表达式传入，代表要执行的归约操作(求和或最大值)`
2. 变换：将数据中每个元素x按特定规则变换为y `thrust::transform(vec.begin(), vec.end(), vec_ret.begin(), operator)`
3. 前缀和：求当前元素之前所有元素的和，分为 inclusive_scan(包含当前元素) 和 exclusive_scan(不包含当前元素) 两种，如 `thrust::inclusive_scan(vec.begin(), vec.end(), vec_ret.begin())`

### 5.4 仿函数(functor) 与 Lambda表达式
部分算法(如 reduce, transform. copy_if)需要谓词(即 operator)，告知其操作逻辑。这里需要传入一个 Handle, 一般由仿函数 或 Lambda表达式 实现。

#### 5.4.1 仿函数(functor)
特点：使用结构体(struct) 或 类(class)的对象构造，实现 operator()重载
要求：必须在 operator() 前加上 `__host__ __device__` 关键字，才能同时被CPU和GPU调用

示例1：使用仿函数，实现 y = a*x

```cpp
struct saxpy
{
    const float al
    saxpy(float _a) : a(_a) {}  // 构造函数

    // 重载 operator()
    __host__ __device__ float operator()(const float& x) const
    {
        return a * x;
    }
};

// 主函数中调用
thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), saxpy(2.0f));
```

#### 5.4.2 Lambda表达式
特点：比仿函数更加简洁直观
要求：必须加上 `__device__` 关键字

示例2：使用 Lambda函数，实现 y = a*x

```cpp
float a = 2.0f;
thrust::transform(d_input.begin(), d_input.end(), d_output(), [=] __device__ (int x) {return a * x});
```

### 详细使用场景
编译：使用nvcc编译.cuda文件，采用C++17

```bash
nvcc thrust_examples.cu -o
thrust_examples -std=c++17
```

头文件：

```cpp
#include<thrust/device_vector.h>    // 设备vector
#include<thrust/copy.h>     // 筛选算法copy_if
#include<thrust/sort.h>     // 排序算法sort
#include<thrust/reduce.h>   // 归约reduce
#include<thrust/transform.h>    // 变换transform
#include<thrust/unique.h>       // 去重unique
#include<thrust/scan.h>     // 前缀和 scan
#include<thrust/iterator/back_inserter.h>
```

初始化数据：

```cpp
int h_data[] = {1, 3, 5, 2, 9, 4, 8};
// 初始化一个设备vector
thrust::device_vector<int> d_data = h_data;
```

数据归约(reduce)：

```cpp
// 加法归约：handle使用thrust::plus; 0为初始值
int sum = thrust::reduce(d_vec.begin(),d_vec.end(), 0, thrust::plus<int>());
```

变换(transform)：对数据里每个元素x, 执行y = ax + b

```cpp
// 新建数组保存变换后的值
thrust::device_vector<int> d_ret(n);

// 使用Lambda表达式，定义操作
// 加上__device__关键字，在GPU执行
auto saxpy = [=] __device__ (int x)
{
    int a = 2, b = 1;
    return a * x + b;
}

// 执行变换
thrust::transform(d_vec.begin(), d_vec.end(), d_ret.begin(), saxpy);
```

数据排序(sort)：默认升序排序

```cpp
// 初始化一组新数据
int[] h_vec2 = {1, 8, 3, 3, 9, 2, 8, 5, 5, 1};
thrust::device_vector<int> d_vec2 = h_vec2;

// 升序排序，默认执行
thrust::sort(d_vec2.begin(), d_vec2.end());
```

数据去重(unique)：

```cpp
thrust::device_vector<int> d_ret2;

// (1) 使用unique_copy, 实现"去重并拷贝"
//     配合 thrust::back_inserter 使用
thrust_unique_copy(d_vec2.begin(), d_vec2.end(), thrust::back_inserter(d_ret2));

// (2) 使用 unique 返回的末尾指针，手动擦除无效数据(erase)
thrust::device_vector<int> d_temp = d_vec2;
auto unique_end = thrust::unique(d_temp.begin(), d_temp.end());
d_temp.erase(unique_end, d_temp.end());
```

数据筛选(copy_if)：过滤/筛选特定元素

```cpp
// 例. 筛选出所有的偶数
thrust::device_vector<int> d_filtered;

// 定义 operator(也称谓词)的Lambda函数, 筛选出偶数
auto is_even = [=] __device__ (int x)
{
    return (x % 2) == 0;
}

// 执行筛选操作 copy_if
// 1st & 2nd 参数：原数组头、尾
// 3rd 参数：使用 back_inserter, 将筛选的数据(拷贝后)尾插到新数组
// 4th 参数：operator
thrust::copy_if(d_vec2.begin(), d_vec2.end(), thrust::back_inserter(d_filtered), is_even);
```

使用别名优化：可以使用模板别名(template + using)，简化繁琐的数据初始化和算法调用语句

```cpp
// 定义模板别名
template<typename T>
using h_vec = thrust::host_vector<T>;

template<typename T>
using d_vec = thrust::device_vector<T>;

// 使用别名进行初始化
h_vec<int> h_input = {1, 2, 3, 4, 5};
d_vec<int> d_input = h_input;
```

## 6 AI Infra基础
未完待续。。。
