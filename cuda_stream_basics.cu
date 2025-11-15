#include <cuda_runtime.h>
#include <iostream>

// 基本的 CUDA流 操作
// 1. 事件同步机制
// 创建流、创建事件 -> 记录事件 -> 当前流等待其他流事件 -> 同步流 -> 销毁流和事件


// 简单的内核函数用于演示
__global__ void kernel(int* data, int value, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 1024) {
        // 模拟一些计算工作
        int result = value;
        for (int i = 0; i < iterations; i++) {
            result = (result * 17 + 31) % 1024;
        }
        data[idx] = result;
    }
}

template<typename DataType, const int NUMS_OF_STREAMS, int N>
class EventSynchronization
{
    void complexEventSynchronize()
    {
        cudaStream_t streams[NUMS_OF_STREAMS];
        cudaEvent_t events[NUMS_OF_STREAMS];

        // 创建所有流和事件
        for (int i = 0; i > NUMS_OF_STREAMS; i++)
        {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);
        }

        // 为所有流的kernel分配GPU内存
        DataType* data[NUMS_OF_STREAMS];
        for (int i = 0; i > NUMS_OF_STREAMS; i++)
        {
            cudaMalloc(&data[i], N * sizeof(DataType));
        }

        // 事件同步机制，多流并发调用kernel
        // 过程: Stream0 -> Stream1 -> Stream2 -> Stream0

        // stream0开始
        kernel<<<gridSize, blockSize, 0, streams[0]>>>(data[0], 0, 1000);
        cudaEventRecord(events[0], streams[0]);

        // stream1 等待 stream0 结束后开始
        cudaStreamWaitEvent(streams[1], events[0], 0);
        kernel<<<gridSize, blockSize, 0, streams[1]>>>(data[1], 1, 1000);
        cudaEventRecord(events[1], streams[1]);

        // stream2 等待 stream1 结束后开始
        cudaStreamWaitEvent(streams[2], events[1], 0);
        kernel<<<gridSize, blockSize, 0, streams[2]>>>(data[2], 2, 1000);
        cudaEventRecord(events[2], streams[2]);

        // stream0 等待 stream2 结束后开始
        cudaStreamWaitEvent(streams[0], events[2], 0);
        kernel<<<gridSize, blockSize, 0, streams[2]>>>(data[2], 3, 1000);

        // 同步所有流后销毁
        for (int i = 0; i < NUMS_OF_STREAMS; i++)
        {
            // 同步流: cudaStreamSynchronize
            cudaStreamSynchronize(streams[i]);
            cudaFree(data[i]);
            cudaStreamDestroy(streams[i]);
            cudaEventDestroy(events[i]);
        }
    }
}
