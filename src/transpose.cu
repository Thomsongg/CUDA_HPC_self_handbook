#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a, b) ((a + b - 1) / b)

// 共享内存 + padding 优化
template<int BLOCK_SIZE>
__global__ void matrixTransposeShared(float *input, float *output, int M, int N)
{
    // 加 1 的padding
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1];
    int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    int idy = threadIdx.y + BLOCK_SIZE * blockIdx.y;

    // 每个线程将一个block_size * (block_size + padding)的转移矩阵A到共享内存
    if (idy < M && idx < N)
    {
        smem[threadIdx.y][threadIdx.x] = input[idy * N + idx];
    }

    __syncthreads();

    // 将共享内存转置后存放入矩阵B，新的位置也进行转置
    int new_idx = threadIdx.x + BLOCK_SIZE * blockIdx.y;
    int new_idy = threadIdx.y + BLOCK_SIZE * blockIdx.x;
    if (new_idx < M && new_idy < N)
    {
        output[new_idy * M + new_idx] = smem[threadIdx.x][threadIdx.y];
    }
}

struct KernelConfig {
    dim3 gridDim;
    dim3 blockDim;
    int threadsPerBlock;
    int totalThreads;
};

KernelConfig configureSharedTranspose(int M, int N, int BLOCK_SIZE) {
    KernelConfig config;
    
    // 验证参数有效性
    if (BLOCK_SIZE * BLOCK_SIZE > 1024) {
        std::cerr << "Error: BLOCK_SIZE is not suggested for over 32" << std::endl;
        exit(1);
    }
    
    // 配置线程块
    config.blockDim = dim3(BLOCK_SIZE, BLOCK_SIZE);
    config.threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    
    // 配置网格
    config.gridDim = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    config.totalThreads = config.gridDim.x * config.gridDim.y * config.threadsPerBlock;
    
    return config;
}

// 使用示例
template<int BLOCK_SIZE>
void launchSharedTranspose(float* input, float* output, int M, int N)
{
    KernelConfig config = configureSharedTranspose(M, N, BLOCK_SIZE);
    
    std::cout << "Grid: " << config.gridDim.x << " x " << config.gridDim.y << std::endl;
    std::cout << "Block: " << config.blockDim.x << " x " << config.blockDim.y << std::endl;
    std::cout << "Threads per block: " << config.threadsPerBlock << std::endl;
    std::cout << "Total threads: " << config.totalThreads << std::endl;
    
    // 调用核函数
    matrixTransposeShared<BLOCK_SIZE>
        <<<config.gridDim, config.blockDim>>>(input, output, M, N);
    
    // 检查核函数调用是否失败
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

int main()
{
    const int BLOCK_SIZE = 32;
    const int M = 512;
    const int N = 256;

    float *h_input, *h_output;
    float *d_input, *d_output;

    h_input = new float[M * N];
    h_output = new float[M * N];

    for (int i = 0; i < N; i++)
    {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemset(d_output, 0, M * N * sizeof(float));

    cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice);

    launchSharedTranspose<BLOCK_SIZE>(d_input, d_output, M, N);

    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    bool ret = true;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (h_input[i * N + j] != h_output[j * M + i])
            {
                ret = false;
                std::cout << "Error occurs, input: " << h_input[i * N + j] << ", output: " << h_output[j * M + i] << std::endl;
            }
        }
    }

    if (ret)
    {
        std::cout << "Result is true, calculation finished." << std::endl;
    }

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}