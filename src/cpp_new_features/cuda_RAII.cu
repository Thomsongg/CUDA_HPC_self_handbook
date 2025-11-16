#include <cuda_runtime.h>
#include <stdexcept>
#include <memory>
#include <iostream>

// 自定义Cuda删除器，释放GPU内存
struct CudaDeleter
{
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
}

// 使用RAII 用于GPU内存管理
// 原理：创建类作为RAII 包装器，将cudaMalloc和cudaFree写入构造、析构函数，实现内存创建与释放的同步
// CudaMemoryRaw: 原生指针
class CudaMemoryRaw
{
    private:
        void* ptr_;
        size_t size_;

    public:
        // 参数构造函数
        CudaMemoryRaw(size_t size) : size_(size), ptr_(nullptr)
        {
            // 构造时分配GPU内存
            cudaError_t error = cudaMalloc(&ptr_, size_t);
            if (error != cudaSuccess)
            {
                throw std::runtime_error("[Error]Allocate device mem failed.");
            }
        }

        // 析构函数，在调用构造函数异常时自动执行
        ~CudaMemoryRaw()
        {
            // 不使用智能指针时：析构函数手动释放内存空间
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
            }
        }

        // 禁止拷贝，保护GPU内存
        CudaMemoryRaw(const CudaMemoryRaw&) = delete;
        CudaMemoryRaw& operator=(const CudaMemoryRaw&) = delete;

        // 允许移动构造
        CudaMemoryRaw(const CudaMemoryRaw&& other) noexcept : size_(other.size_), ptr_(other.ptr_)
        {
            // 原生指针无法使用 std::move, 则需要将原指针置为NULL
            other.ptr = nullptr;
            other.size_ = 0;
        }

        // 移动构造重载运算符'='
        CudaMemoryRaw& operator=(const CudaMemoryRaw&& other) noexcept : size_(other.size_), ptr_(other.ptr_)
        {
            if (this != &other)
            {
                // 先释放当前资源
                if (ptr_)
                {
                    cudaFree(ptr_);
                }
                // 转移所有权，模拟std::move
                ptr_ = other.ptr;
                size_ = other.size_;
                // 将原对象置为NULL
                other.ptr = nullptr;
                other.size_ = 0;
            }

            return *this.
        }
}

// CudaRAIIContainer: 使用智能指针的完整RAII容器(推荐！)
// 特点:
// (1) 使用智能指针管理成员的创建和销毁
// (2) 通过模板元编程，支持动态数据类型传入(int, float, double...)
// (3) 包含基本的成员访问与操作方法
template<typename T>
class CudaRAIIContainer
{
    private:
        std::unique_ptr<T, CudaDeleter> data_;
        size_t size_;

    public:
        // 参数构造
        CudaRAIIContainer(size_t size) : size_(size)
        {
            if (size > 0)
            {
                // 使用原始指针分配GPU内存
                T* ptr_raw = nullptr;
                cudaError_t err = cudaMalloc(&ptr_raw, size * sizeof(T));
                if (err != cudaSuccess)
                {
                    throw std::runtime_error("[Error]Allocate device mem failed!");
                }
                // 销毁原始指针
                data_.reset(ptr_raw);
            }
        }

        // 禁止拷贝构造
        CudaRAIIContainer(const CudaRAIIContainer& other) = delete;
        CudaRAIIContainer& operator=(const CudaRAIIContainer& other) = delete;

        // 移动构造
        CudaRAIIContainer(CudaRAIIContainer&& other) noexcept :
                data_(std::move(other.data_)),
                size_(other.size_)
        {
            other.size_ = 0;
        }

        // 移动赋值运算符
        CudaRAIIContainer& operator=(CudaRAIIContainer&& other) noexcept
        {
            if (this != &other)
            {
                data_ = std::move(other.data_);
                size_ = other.size_;
                other.size_ = 0;
            }

            return *this;
        }

        // Getter: 返回原始指针
        T* get() const
        {
            return data_.get();
        }

        // 类独有方法...
        // 内存拷贝 主机->GPU
        void copyToDevice(const T* host_data, size_t count = 0)
        {
            size_t copy_size = count > 0 ? count : size_;
            // 将cudaMemcpy封装
            // 使用原生指针 data_.get()
            cudaMemcpy(data_.get(), host_data, copy_size * sizeof(T), cudaMemcpyHostToDevice);
        }

        // 内存拷贝 GPU->主机
        void copyToHost(T* host_data, size_t count = 0) const
        {
            size_t copy_size = count > 0 ? count : size_;
            cudaMemcpy(host_data, data_.get(), copy_size * sizeof(T), cudaMemcpyDeviceToHost);
        }
}

// 使用RAII容器的 矩阵乘法GEMM类
// 输入：矩阵A, B, C(均为GPU内存)、矩阵尺寸M, N, K
template<const int BLOCK_SIZE>
class CudaGEMMWithRAII
{
    // 使用RAII容器 创建成员变量，每个变量包含 数据成员data_ 和 尺寸size_
    // 本案例使用 float 型矩阵
    private:
        CudaRAIIContainer<float> A_, B_, C_;
        int M_, N_, K_;
    
    public:
        // 参数构造函数：对容器类成员的初始化，会执行cudaMalloc, 详见上面容器类的代码
        CudaGEMMWithRAII(int M, int N, int K) : M_(M), N_(N), K_(K),
            A_(M * K), B_(K * N), C_(M * N) {}
        
        // 移动构造函数：使用移动语义转移所有权，并将原对象置空
        CudaGEMMWithRAII(CudaGEMMWithRAII&& other) noexcept : M_(other.M_), N_(other.N_), K_(other.K_),
            A_(std::move(other.A_)), B_(std::move(other.B_)), C_(std::move(other.C_))
        {
            other.M_ = other.N_ = other.K_ = 0;
        }

        // 移动赋值运算符
        CudaGEMMWithRAII& operator=(CudaGEMMWithRAII&& other) noexcept
        {
            if (this != &other)
            {
                A_ = std::move(other.A_);
                B_ = std::move(other.B_);
                C_ = std::move(other.C_);
                M_ = other.M_;
                N_ = other.N_;
                K_ = other.K_;
                other.M_ = other.N_ = other.K_ = 0;
            }

            return *this;
        }

        // 分配线程并调用kernel函数
        // 使用host内存传入，GPU内存已使用RAII分配
        void compute(float* h_A, float* h_B, float* h_C)
        {
            // 使用容器类方法，拷贝CPU内存到GPU
            A_.copyToDevice(h_A);
            B_.copyToDevice(h_B);

            // 分配网格
            dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
            dim3 gridSize((N_ + BLOCK_SIZE - 1) / BLOCK_SIZE, (M_ + BLOCK_SIZE - 1));

            // 调用核函数
            gemm_shared<BLOCK_SIZE><<<gridSize, blockSize>>>(A_.get(), B_.get(), C_.get(), M_, N_, K_);
            
            // 等待GPU执行完成
            cudaDeviceSynchronize();

            // 从GPU内存取回计算结果
            C_.copyToHost(h_C);
        }
}
