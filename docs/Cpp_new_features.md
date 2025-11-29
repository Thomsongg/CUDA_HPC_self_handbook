# C++11及以上 新特性
本文关注C++11、17、20中高频用于CUDA高性能计算的新特性

## 资源
推荐《Effective Modern C++》和C++官方文档。
1. C++11新特性:  [cpp面经](https://github.com/guaguaupup/cpp_interview/blob/main) [C++教程](https://subingwen.cn/cplusplus/)
2. C++17: https://github.com/MeouSker77/Cpp17/tree/master
3. C++20: https://github.com/Mq-b/Cpp20-STL-Cookbook-src

## C++新特性与CUDA高性能计算编程的结合
- 使用现代C++重写CUDA高性能计算代码
- 实现高性能并发数据结构

```C++
// 用现代C++重写CUDA项目，展示技术深度
class ModernCUDALauncher {
    template<typename Kernel, typename... Args>
    void launch_optimized(dim3 grid, dim3 block, Kernel&& kernel, Args&&... args) {
        // 使用完美转发保持参数类型
        kernel<<<grid, block>>>(std::forward<Args>(args)...);
        
        // 现代错误处理
        if (auto error = cudaGetLastError()) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }
};

// 实现高性能并发数据结构
template<typename T>
class GPUReadyQueue {
    moodycamel::ConcurrentQueue<T> cpu_side_;
    std::atomic<bool> gpu_processing_{false};
    
public:
    void enqueue_batch(gsl::span<const T> items) {
        // 零拷贝或批量传输优化
    }
};
```

## 1 重要特性
- 内存模型和原子操作（用于无锁编程）

- 移动语义和完美转发（避免不必要的复制）

- 智能指针（资源管理）

- 并行算法（C++17的并行STL）

- 协程（C++20，用于异步编程）

- 概念（C++20，模板约束）

## 2 :watermelon: C++11新特性
- auto & decltype
- 【重要】C++智能指针
- 【重要】原子操作
- lambda表达式(匿名函数)

### 2.1 C++11关键字
C++11有很多新增的关键字，这里我们只关注高性能计算会用到的几个
#### auto 与 decltype

#### noexcept

#### override

### 2.2 C++智能指针
C++目前还在使用的智能指针有几类：unique_ptr、shared_ptr、weak_ptr、auto_ptr(已废弃)

#### 使用智能指针的原因
1. 内存泄漏，即new与delete不匹配
2. 多线程下对象析构问题，造成这个问题本质的原因是类对象自己销毁(析构)的时候无法对自己加锁,所以要独立出来,采用这个中间层(shared_ptr).

#### 2.2.1 unique_ptr
##### 特点
- 独占所有权，不能拷贝只能移动
- 零开销，性能接近裸指针
- 自动管理资源生命周期

##### 主要方法
- get() - 获取原始指针
- reset() - 重置指针
- release() - 释放所有权
- swap() - 交换指针
- operator-> 和 operator* - 访问对象

#### 2.2.2 shared_pty
##### 特点
- 共享所有权，使用引用计数
- 支持拷贝和赋值
- 当最后一个shared_ptr被销毁时释放资源

##### 主要方法
- use_count() - 获取引用计数
- reset() - 重置指针
- get() - 获取原始指针
- swap() - 交换指针

#### 2.2.3 weak_ptr
##### 特点
- 不增加引用计数
- 解决shared_ptr循环引用问题
- 需要从shared_ptr创建

##### 主要方法
- lock() - 尝试获取shared_ptr
- expired() - 检查对象是否已被销毁
- use_count() - 获取原始shared_ptr的引用计数
- reset() - 重置weak_ptr

## 3 现代C++的应用(11/17/20)
**核心：** 解决异构计算的三大难题
1. 资源安全与管理：资源管理不当导致内存泄漏
2. 代码复用与泛型：通过泛型编程，替代一整个类型(如int, float)，无需重复代码
3. 性能与抽象的平衡：既保证代码的可读性、拓展性，又能保持良好的性能

### 附 C++资源管理“五原则”(五步法)
1. 禁止拷贝构造：防止多个对象管理同一块内存  
```cpp
MyClass(const MyClass&) = delete
```

2. 禁止拷贝赋值：  
```cpp
MyClass& operator=(const MyClass& other) = delete
```

3. 采用移动构造：从临时对象转移所有权
```cpp
MyClass(MyClass&& other) noexcept : d_ptr(other.d_ptr), nums(other.nums)
{
    // 将原对象置NULL
    other.d_ptr = nullptr;
    other.nums = 0;
}
```

4. 采用移动赋值：
```cpp
MyClass&& operator=(MyClass&& other) noexcept
{
    // 防止赋值给自己
    if (this != &other)
    {
        // 释放资源
        if (d_ptr != nullptr)
        {
            cudaFree(d_ptr);
        }

        // 转移所有权
        d_ptr = other.d_ptr;
        nums = other.nums;

        // 将原对象置NULL
        other.d_ptr = nullptr;
        other.nums = 0;
    }
}
```

5. 析构时释放：构造失败时自动释放资源

```cpp
~MyClass()
{
    // 原始指针，需手动释放
    if (d_ptr != nullptr)
    {
        cudaFree(d_ptr);
        d_ptr = nullptr;
    }
}
```

### 3.1 RAII & 智能指针【必须】
**痛点**：传统C++极易出现的资源安全问题。
1. CPU/GPU内存申请后，忘记释放
2. 异常安全：类对象构造时，申请资源后抛出异常，无法自动析构而导致内存泄漏。

**HPC的应用：** 采用RAII类进行包装，实现智能GPU内存管理。
采用原始指针(raw_ptr)时，应严格遵守“五原则”。
若使用智能指针(unique_ptr)，可自动释放内存，无需单独编写构造函数，极大简化对象构造。

RAII的场景：
1. 智能指针(std::unique_ptr)：将智能指针作为RAII类的成员，实现资源自动释放；类可以专注于自身业务逻辑
2. 自定义删除器(Deleter)：一个轻量级结构体，在智能指针成员析构时自动调用，自动释放内存。

代码示例：

```cpp
#include<stdexcept> // 引入异常处理
#include<memory>    // 引入智能指针 unique_ptr
#include<utility>   // 引入移动语义 move

// Deleter结构体
struct DeviceBufferDelelter
{
    void operator()(void* ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
}

// RAII类容器，负责GPU内存管理
template<typename T>
class DeviceBuffer {
private:
    // 关键优化：使用智能指针，实现自动析构与释放
    // 类型为模板定义，析构时自动调用Deleter
    std::unique_ptr<T, DeviceBufferDelelter> d_ptr_ = nullptr;
    size_t num_elements_ = 0;

public:
    // 参数构造，供外部申请内存空间
    explicit DeviceBuffer(size_t num_elements) : num_elements_(num_elements)
    {
        if (num_elements == 0)
        {
            return;
        }

        // 分配原始指针，申请内存空间
        T* raw_ptr = nullptr;
        sizeError_t error;
        error = cudaMalloc(reinterpret_cast<void**>(&raw_ptr), num_elements * sizeof(T));
        if (error != cudaSuccess)
        {
            // 必要的异常处理逻辑(如日志打印)
            reportCudaError();
        }

        // 把原始指针所有权 交还给智能指针
        d_ptr_.reset(raw_ptr);
    }

    // 极简化对象构造(五原则)
    // (1) 禁止拷贝
    DeviceBuffer(const DeviceBuffer& other) = delete;
    DeviceBuffer& operator=(const DeviceBuffer& other) = delete;

    // (2) 默认移动构造
    DeviceBuffer(DeviceBuffer&& other) noexcept = default；
    DeviceBuffer&& operator=(DeviceBuffer&& other) noexcept = default;

    // (3) 默认析构
    ~DeviceBuffer() = default;
}
```

### 3.2 Lambda表达式【必须】
**痛点：** 使用外部并行库(如Thrust)时，需要经常定义大量“一次性”的简单函数对象，过程繁琐、代码可读性和可维护性差。

**Lambda表达式：** 一个匿名函数对象，需要时可直接定义。
**语法：** `[capture](params) -> return_type { body }`
- 捕获列表(capture)：
	- \[=]: 以值的方式，捕获所有外部变量
    - \[&]: 以引用的方式，捕获所有外部变量
    - \[a, &b]: 以值捕获a，以引用捕获b
    - \[this]: 捕获当前类的this指针
- 参数列表(params): 入参列表，可以为空
- 返回类型(return_type): 可省略（由编译器自动推导）
- 函数体(body): 函数执行代码，可为空(必须包含'{}')

**特点：**
- 闭包性：可以捕获所在作用域的变量，并实现变量的访问和操作
- 内联性：由编译器内联，减少函数调用开销
- 匿名类型：提供一个匿名函数对象，可通过std::function对象保存
- 代码简洁
- 捕获上下文

**示例1：** 保存到可调用对象 std::function
```cpp
typedef std::function<int(int, int)> comfun;
// 普通函数 与 lambda表达式
int add(int a, int b) { return a + b; }
auto mod = [](int a, int b){ return a % b; };

int main()
{
    // 将lambda对象保存到可调用对象 std::function
    comfun a = add;
    comfun b = mod;
    std::cout << a(3, 5) << " and " << b(3, 5);
}
```

**示例2：** 简化代码
```cpp
// 使用匿名函数，设置sort谓词
std:sort(nums.begin(), nums.end(),
         [](int a, int b) { return a > b; });

```

**示例3：** 捕获上下文

```cpp
int threshold = 100;
//Lambda函数，用捕获的上下文执行特定操作
auto filter = [threshold](int value) {
    return value > threshold;
};

```

**CUDA HPC的应用：** 并行库Thrust
示例：将数组中的每个元素x 变换为 y = 2*x + 1

```cpp
#include<thrust/device_vector.h>
#include<thrust/transform.h>

template<typename T>
using h_vec = thrust::host_vector<T>

template<typename T>
using d_vec = thrust::device_vector<T>

h_vec<float> h_input = {1, 2, 3, 4, 5};
d_vec<float> d_intput;
d_intput = h_input

d_vec<float> d_output;

// 使用Lambda函数，实现 y = a*x + b, a与b通过拷贝传入
float a = 2.0f, b = 0.2f;
thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), [=] __device__ (int x) {return a * x + b;});
```

### 3.3 编译期条件(constexpr & if constexpr)【必须】
在编译时决定哪部分代码会通过编译，实现性能优化。
**特点：** 编译器条件语句，用 **`if constexpr`** 修饰。
**原理：**
- 如果条件判断为 **true**, 则会编译 **if** 内的语句，**彻底丢弃** else 内的代码块。
- 如果条件判断为 **false**，则恰好相反。

“彻底丢弃”：编译器不会检查剩余代码的语义/语法正确性，放弃编译。

**优势**：
1. 简化模板元编程
2. 提高可拓展性：可以直接在同一个函数模板中添加不同条件下的处理逻辑（如各种硬件架构、不同数据类型），无需重载
3. 零运行开销：最终生成的机器码只包含特定路径，运行时不会因路径判断导致性能损失

**在CUDA HPC的应用：**
1. 消除 Warp Divergence
    - 问题：同一个 Warp 的不同线程，在遇到 if 分支时，会根据各自的条件判断规则，串行执行，浪费大量的执行周期。
    - 编译期条件的优化：如果分支条件是基于一个**模板参数**（在编译时是恒定的），可以使用 `if constexpr`

2. 高优化的内核(Uber-kernel)：创建高度集成化的Kernel, 将微小的功能合并，实现特定条件执行特定逻辑，无需单独创建冗余的Kernel
3. 降低寄存器负载
4. 适配不同的GPU硬件架构

**优势：**
1. 减少GPU周期的浪费
2. 提升GPU Occupancy
3. 提升计算效率

#### 使用场景详解
###### (1) 消除 Warp Divergence
**场景：** 根据操作Handle，选择 向量加法 或 求最大值
**方法：** 使用模板类定义 操作Handle，通过编译期条件 选择特定核函数运行；完美规避运行期的 if分支 导致的 Warp Divergence

代码如下：

```cpp
// 枚举类 定义操作类型
enum class OpType
{
    Sum,
    Max
};

// 使用 编译期条件 的Device函数
template<Optype op, typename T>
__device__ T perform_op(T a, T b)
{
    // 根据编译时实际的操作类型，选择特定分支进行实例化
    if constexpr ()
    {
        return a + b;
    }
    else
    {
        return fmaxf(a, b);
    }
}

// 应用 device函数 的Kernel
template<Optype op, typename T>
__global__ void process_data(const T* A, const T* B, T* C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // 这里编译时会触发device函数的编译器条件判断
        // 最终运行时只会保留一个分支代码
        C[idx] = perform_op<op, T>(A[idx], B[idx]);
    }
}
```

###### (2) “超级内核”(Uber-kernel)
**场景：** 结合模板和编译期条件，控制以下三个维度
1. 数据类型泛化(int / float / double)
2. 是否应用缩放(ApplyScaling)
3. 是否使用原子操作(UseAtomics)

前提-使用<type_traits>库中的 `std::is_same_v`，判断两个数据类型是否相同

代码示例如下：

```cpp
// Kernel使用模板类控制
template<typename T, bool ApplyScaling, bool UseAtomics>
__global__ void uber_kernel(const T* input, T* output, T scale, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }

    // 寄存器内执行高精度计算
    T val = input[idx];

    // (1) 特性1: 特定数据类型执行单独的计算
    //     double类型执行高精度计算, float类型使用快速近似
    if constexpr(std::is_same_v(T, double))
    {
        val = (a * a + a) / 1.23456789;
    }
    else
    {
        // 快速倒数
        val = __frcp_rn(val);
    }

    // (2) 特性2: 可选项，是否对数据进行特定大小的缩放
    if constexpr (ApplyScaling)
    {
        val *= scale;
    }

    // (3) 特性3: 是否使用原子操作
    if constexpr (UseAtomics)
    {
        // 注意：原子操作只有float类型可用
        //      其他类型需要自定义
        if constexpr (std::is_same_v(T, float))
        {
            atomicAdd(&output[0], val);
        }
        else
        {
            // 自定义原子函数，在此忽略...
        }
    }
    else
    {
        output[idx] = val;
    }
}

```

几种编译期条件不同的kernel调用示例：

```cpp
// 传入参数 scale, 但不进行缩放(ApplyScaling=false)
// 结果：参数正常传入，但不会被使用
double scale_d = 2.0;
uber_kernel<double, false, false><<< gridSize, blockSize >>>(d_input, d_output, scale_d, N);

```

###### (3) 降低寄存器压力
**场景：** 部分高精度计算场景，要求GPU分配大量的寄存器以实现更快的数据传输。但低精度(float及以下)时，如果使用 运行期if, 编译器会预先分配额外的寄存器，即使没有用到，也会极大增加寄存器压力，影响 Occupancy等性能指标。

**方法：** 结合模板和编译期条件，区分高/低精度模式，在编译时执行保留并执行特定的逻辑。

代码示例如下：

```cpp
#include<stdlib.h>
#include<type_traits>

template<typename T>
__global__ void complex_cal(const T* input, T* output, int N)
{
    int idx = ...;
    if (idx >= N)
    {
        return;
    }

    // 只有高精度才需要额外分配一个 16bytes 寄存器
    if constexpr (std::is_same_v(T, double))
    {
        // 模拟高精度计算
        double temp_reg[16];
        double val = (double)input[idx];
        temp_reg[0] = val;
        for (int i = 1; i < 16; i++)
        {
            temp_reg[i] = (temp_reg[i - 1] * 1.01 + 0.01) * val;
        }
        output[idx] = temp_reg[15];
    }
    // 低精度无需额外分配寄存器，只执行简单计算
    else
    {
        T val = input[idx];
        val = val * 0.5f + 1.2f;
        output[idx] = val;
    }
}

```

###### (4) 适配硬件架构
**场景：** 归约算法的 Warp Shuffle，需要使用Warp内线程同步操作。需要根据GPU架构匹配对应的处理逻辑。
1. 较新的GPU架构支持 `__shfl__sync()`指令
2. 较老架构只支持 `__sync()`指令
3. 更老的架构均不支持

**方法：** 使用`__CUDA__ARCH__`宏，选择合适的指令实现 Warp Shuffle 的Device函数

**说明：** 
1. `__CUDA__ARCH__`是一个编译期常量，其值等于架构号(如 CC 7.0) * 100
2. 如果没有为设备编译，则`__CUDA__ARCH__`未定义或其值为0

代码如下：

```cpp
__device__ float warp_reduce_sum(float val)
{
    // (1) 新架构(Volta及以上， CC 7.0+)
    #if defined (__CUDA__ARCH__) && __CUDA__ARCH__ >= 700
    {
        if constexpr (__CUDA__ARCH__ >= 700)
        {
            // 支持全线程shuffle同步，采用全掩码
            unsigned mask = 0XFFFFFFFFU;
            for (int offset = 16; offset > 0; offset /= 2;)
            {
                val += __shfl_down_sync(mask, val, offset);
            }
        }
    }

    // (2) 较老架构(CC 3.0-7.0)
    #elif defined (__CUDA__ARCH__) && __CUDA__ARCH__ >= 300
    {
        if constexpr (__CUDA__ARCH__ < 700) && (__CUDA__ARCH__ > 300)
        {
            // 使用较老的shuffle指令，无掩码操作
            // 注意：需要告知编译器，忽略指令过期提示
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            for (int offset = 16; offset > 0; offset /= 2;)
            {
                val += __shfl_down(val, offset);
            }
            #pragma GCC diagnostic pop
        }
    }
    #endif

    return val;
}

```

### 3.4 移动语义 & 完美转发【重要】
移动语义：本质是一种**类型转换**，将参数转换为**右值引用(&&)**，转移资源所有权给新的对象。

**在CUDA HPC中的意义：** 
- 拷贝代价高昂：需要重新调用cudaMalloc分配一块GPU内存，很慢且产生内存占用；还要调用cudaMemcpy额外实现内存传输，占用PCIe/MVlink带宽。
- 期望的效果：如果只想把一个GPU缓冲区转移到其他函数中，或存入一个容器(如thrust::device_vector)时，直接转移而非拷贝。

**应用场景：**
1. 工厂函数：专门创建和初始化GPU缓冲区
2. 在容器中管理多个缓冲区
3. 转移所有权给另一个对象

### 3.5 Host端并发库(std::thread, std::future, std::async)【重要】


### 3.6 std::span


