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

**CUDA HPC的应用：**
- 并行库Thrust

### 3.3 编译期条件(constexpr & if constexpr)【必须】

### 3.4 移动语义 & 完美转发【重要】


### 3.3 Host端并发库(std::thread, std::future, std::async)【重要】


### 3.6 std::span


