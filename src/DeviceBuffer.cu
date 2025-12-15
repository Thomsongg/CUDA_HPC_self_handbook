#include<iostream>
#include<stdexcept>
#include<memory>
#include<utility>
#include<cuda_runtime.h>

// 检查CUDA API调用错误的宏
#define CHECK_CUDA(call) do {
    cudaError_t err = call;
    if (err != cudaSuccess)
    {
        std:cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - "
                 << cudaGetErrorString(err) << std::endl;
                 throw std::runtime_error("CUDA Error");
    }
} while (0)

/*
 * 使用RAII类容器 和 智能指针，管理GPU内存
*/

// Deleter结构体 - unique_ptr必备，用来析构和释放资源
struct DeviceBufferDeleter
{
    void operator()(void* ptr) const
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
}

// 使用别名，用unique_ptr 管理GOU内存
template<typename T>
using CudaUniquePtr = std::unique_ptr<T, DeviceBufferDeleter>;

/*
 * 使用RAII类容器，负责GPU内存管理
 * 用模板类做泛型数据类型
*/
template<typename T>
class DeviceBuffer
{
  // 私有成员，含内存空间指针(smartPtr)和空间大小(size_t)
  private:
    CudaUniquePtr data_;
    size_t size_ = 0;
  
  // 构造器与公共方法
  public:
    // 允许外部参数构造，申请特定内存空间
    explicit DeviceBuffer(size_t num_elements) : size_(num_elements)
    {
        if (num_elements > 0)
        {
            T* raw_ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&raw_ptr, size_ * sizeof(T)));
            // 将原指针交由智能指针管理
            CudaUniquePtr data_.reset(raw_ptr);
        }
    }

    // “五步法”
    // (1) 禁止拷贝构造
    DeviceBuffer(DeviceBuffer& other) = delete;
    DeviceBuffer& operator = (DeviceBuffer& other) = delete;

    // (2) 移动构造：使用移动语义，转移所有权
    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(std::move(other.data_)), size_(other.size_)
    {
        // 使用move 后，只需调整 size_， 自动实现 other.data_ = nullptr
        other.size_ = 0;
    }

    DeviceBuffer& operator = (DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            data_ = std::move(other.data_);
            size_ = other.size_;
            other.size_ = 0;
        }

        return *this;
    }

    // (3) 不需要析构函数(由智能指针的Deleter实现)

    // 传输数据到GPU内存
    void copyToDevice(const T* host_data)
    {
        CHECK_CUDA(cudaMemcpy(data_.get(), host_data, size_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    // 传输数据到CPU内存
    void copyToHost(T* host_data) const
    {
        if (size_ != 0)
        {
            CHECK_CUDA(cudaMemcpy(host_data, data_.get(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    // 其他方法
    // 返回成员(原始指针) 和 内存大小
    T* get()
    {
        return data_.get();
    }

    size_t getSize()
    {
        return size_;
    }
}
