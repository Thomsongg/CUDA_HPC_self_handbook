# CUDA C++学习笔记

本文档主要以个人的学习笔记、心得体会为主，聚焦以下几个方面：

* GPU硬件架构，SM和SP，与CPU架构的不同
* CUDA抽象硬件层，线程、Warp、线程块、网格之间的关系，SIMT机制
* CUDA编程实战，经典案例与算子，Nsight Compute性能分析
* AI Infra基础知识，分布式存储
* *AI大模型基础*

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

###### 固定内存 (pinned memory)

固定内存是一种特殊的CPU内存。它无法被分页并交换到磁盘上，故可以直接由GPU使能DMA直接访问，实现与CPU异步执行数据传输，是实现异步操作的关键。

**特点：**

- 内存分配和释放的成本较高
- 传输速度高于可分页内存
- 允许执行并行数据传输

**用法**：

- 分配: cudaMallocHost(&h_pinned, SIZE);
- 释放: cudaFreeHost(h_pinned);

###### 异步传输 (cudaMemcpyAsync)

异步传输，是指从固定内存传输数据到GPU内存时，与主机操作异步执行，不会阻塞CPU的操作。依赖固定内存，需指定Cuda流。

**特点：**

- 无阻塞式传输：内存数据传输与CPU操作异步执行
- 硬件支持：异步传输指令，在GPU的DMA引擎排队，等待执行
- 设备内部并行：异步传输指令与GPU其他操作(传输、执行kernel等)并行执行
- 绑定固定内存与流：必须通过固定内存实现传输，且需指定Cuda流

**关键步骤：**

1. 分配固定内存和GPU内存
2. 创建Cuda流: cudaStreamCreate(&stream)
3. 通过指定流执行异步内存传输: cudaMemcpyAsync(d_data, h_pinned, SIZE, cudaMemcpyHostToDevice, stream)
4. GPU执行kernel，CPU执行单独操作(异步执行，非阻塞)
5. 异步传输，从GPU传回固定内存

###### 回调函数 (cudaStreamAddCallback)

Cuda流的回调函数，用于在流进行到特定位置时执行某些特殊操作，如结果和流信息显示、资源管理和性能监控等。写在主机代码中并由主机执行。

**特点：**

- 插入流的特定位置，由主机调用
- 在流执行所有前置操作后，自动调用
- 在主机线程中执行，访问主机资源

##### 异步传输的优点

###### 延迟隐藏 (Latency Hiding)

异步传输，可隐藏GPU产生的延迟：

1. 数据传输延迟，被计算任务隐藏
2. kernel启动延迟，被其他操作隐藏

###### 提高资源利用率 (Occupancy)

无阻塞式工作模式：让GPU计算单元、数据传输单元(DMA引擎)、CPU时刻保持忙碌状态。

###### 更好的响应性 (Response)

CPU工作无阻塞：不影响CPU处理其他任务(尤其是计算和用户输入)，适合交互式处理。

###### 支持复杂工作流 (Work Streams)

通过Cuda流，可搭配任务依赖、事件同步等，实现多流高并发式处理；还可通过回调函数，实现多流运行监控。

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
```
## 5 Thrust并行库
```cpp
取出时: output[idx * M + idy] = smem[ty'][tx']
```

##### 4.1.3.3 其他Swizzling 映射

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
