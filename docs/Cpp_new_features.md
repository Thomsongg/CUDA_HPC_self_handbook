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

#### 2.2.1 shared_ptr

#### 2.2.2 weak_ptr

#### 2.2.3 unique_ptr

