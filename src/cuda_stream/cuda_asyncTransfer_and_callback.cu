#include <cuda_runtime.h>
#include <iostream>

// 演示使用CUDA流 进行异步传输 & 回调函数
// CUDA异步执行：设备执行数据传输和核函数时，不影响主机执行
template<typename DataType>
class CudaAsyncExecution
{
    private:

        /* CUDA异步内存传输示例
            主要步骤：
              (1) 分配内存空间，主机分配固定内存
              (2) 创建Cuda流
              (3) 执行异步传输，主机执行其他任务
              (4) 等待流完成
        */
        void displayAsyncTransfer()
        {
            const int DATA_SIZE = 1024 * 1024;
            DataType *h_pinned, *d_data;

            // 分配CPU固定内存
            cudaMallocHost(&h_pinned, DATA_SIZE * sizeof(DataType));
            cudaMalloc(&d_data, DATA_SIZE * sizeof(DataType));

            // 初始化数据，强转为所需数据类型
            for (int i = 0; i < DATA_SIZE; i++)
            {
                h_pinned[i] = static_cast<DataType>(i);
            }

            // 创建Cuda流
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            // 异步数据传输(GPU通过DMA引擎，获取固定内存数据)
            cudaMemcpyAsync(d_data, h_pinned, DATA_SIZE * sizeof(DataType), cudaMemcpyHostToDevice, stream);
            std::cout << "正在异步传输数据到GPU内存，主机可以执行其他操作" << std::endl;

            // 主机执行其他操作
            for (int i = 0; i < 100000; i++)
            {
                int j = i * i;
            }

            // 等待流执行完毕
            cudaStreamSynchronize(stream);
            std::cout << "GPU异步执行完成" << std::endl;

            // 清理内存，销毁Cuda流
            cudaFreeHost(h_pinned);
            cudaFree(d_data);
            cudaStreamDestroy(stream);
        }

        // 添加回调函数
        /*
            特点：插入主机代码特定位置，在流的前置任务完成后自动调用
                  从而实现 (1) 主机信息获取 (2) 信息提示 等功能
            回调函数的定义：需包含关键字 CUDART_CB
        */
        static void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void* userData)
        {
            int* data = static_cast<int>(userData);
            std::cout << "流Id" << (long long) stream << ", 状态为" << status << ", 用户数据为" << data;
        }

        // 示例：GPU调用多个kernel；第一次调用后执行回调函数，由主机输出流信息
        void displayAsyncCallback()
        {
            const int DATA_SIZE = 1024 * 1024;
            DataType *h_pinned, *d_data;

            cudaMallocHost(&h_pinned, DATA_SIZE * sizeof(DataType));
            cudaMalloc(&d_data, DATA_SIZE * sizeof(DataType));

            for (int i = 0; i < DATA_SIZE; i++)
            {
                h_pinned[i] = static_cast<DataType>(i);
            }

            // 创建Cuda流
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            // 异步传输内存数据
            cudaMemcpyAsync(d_data, h_pinned, DATA_SIZE * sizeof(DataType), cudaMemcpyHostToDevice);

            // GPU第一次执行kernel
            int blockSize = 256;
            kernelFunc<<<CEIL(DATA_SIZE, blockSize, 0, stream)>>>(d_data, DATA_SIZE, 50);

            // 执行回调函数，查询特定信息
            int callback_data = 125;
            cudaStreamAddCallback(stream, myCallback, &callback_data, 0);

            // 第二次执行kernel
            kernelFunc<<<CEIL(DATA_SIZE, blockSize, 0, stream)>>>(d_data, DATA_SIZE, 50);

            // 等待流结束
            cudaStreamSynchronize(stream);

            // 清理
            cudaFreeHost(h_pinned);
            cudaFree(d_data);
            cudaStreamDestroy(stream);
        }
}
