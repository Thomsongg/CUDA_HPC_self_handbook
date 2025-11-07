#include <cuda_runtime.h>
#include <iostream>

#define CEIL(a, b) ((a + b - 1) / b)

template<int BLOCK_SIZE, int SUB_TILE_SIZE>
__global__ void matrixMulRegister(float *A, float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + 1];

    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    if (row + ty >= M || col + tx >= N)
    {
        return;
    }

    float c_reg[SUB_TILE_SIZE][SUB_TILE_SIZE] = {0.0f};

    // 外循环：与共享内存法相同，每个线程块分行/列块，再进行k分割(大块)
    #pragma unroll
    for (int t = 0; t < CEIL(K, BLOCK_SIZE); t++)
    {
        // 对大块再进行k分割，为小块
        #pragma unroll
        for (int i = 0; i < SUB_TILE_SIZE; i++)
        {
            int A_row = ty * SUB_TILE_SIZE + i;
            int A_col = tx;
            if (row + i < M && t * BLOCK_SIZE + tx < K)
            {
                As[A_row][A_col] = A[(row + i) * K + (t * BLOCK_SIZE + tx)];
            }
        }
        
        #pragma unroll
        for (int i = 0; i < SUB_TILE_SIZE; i++)
        {
            int B_row = ty;
            int B_col = tx * SUB_TILE_SIZE + i;
            if (t * BLOCK_SIZE + ty < K && col + i < N)
            {
                Bs[B_row][B_col] = B[(t * BLOCK_SIZE + ty) * N + (col + i)];
            }
        }

        __syncthreads();

        // 按寄存器的维度计算点积
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            #pragma unroll
            for (int i = 0; i < SUB_TILE_SIZE; i++)
            {
                #pragma unroll
                for (int j = 0; j < SUB_TILE_SIZE; j++)
                {
                    c_reg[i][j] += As[ty * SUB_TILE_SIZE + i][k] + Bs[k][tx * SUB_TILE_SIZE + j];
                }
            }
        }

        __syncthreads();
    }

    // 输出计算结果到矩阵C
    #pragma unroll
    for (int i = 0; i < SUB_TILE_SIZE; i++)
    {
        #pragma unroll
        for (int j = 0; j < SUB_TILE_SIZE; j++)
        {
            if (row + i < M && col + j < N)
            {
                C[(row + i) * N + (col + j)] = c_reg[i][j];
            }
        }
    }
}

struct KernelConfig {
    dim3 gridDim;
    dim3 blockDim;
    int threadsPerBlock;
    int totalThreads;
};

KernelConfig configureRegisterOptimizedGEMM(int M, int N, int BLOCK_SIZE, int SUB_TILE_SIZE) {
    KernelConfig config;
    
    // 验证参数有效性
    if (BLOCK_SIZE % SUB_TILE_SIZE != 0) {
        std::cerr << "Error: BLOCK_SIZE must be divisible by SUB_TILE_SIZE" << std::endl;
        exit(1);
    }
    
    const int THREADS_PER_DIM = BLOCK_SIZE / SUB_TILE_SIZE;
    
    // 配置线程块
    config.blockDim = dim3(THREADS_PER_DIM, THREADS_PER_DIM);
    config.threadsPerBlock = THREADS_PER_DIM * THREADS_PER_DIM;
    
    // 配置网格
    config.gridDim = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    config.totalThreads = config.gridDim.x * config.gridDim.y * config.threadsPerBlock;
    
    return config;
}

// 使用示例
template<int BLOCK_SIZE, int SUB_TILE_SIZE>
void launchRegisterOptimizedGEMM(float* A, float* B, float* C, int M, int N, int K)
{
    KernelConfig config = configureRegisterOptimizedGEMM(M, N, BLOCK_SIZE, SUB_TILE_SIZE);
    
    std::cout << "Grid: " << config.gridDim.x << " x " << config.gridDim.y << std::endl;
    std::cout << "Block: " << config.blockDim.x << " x " << config.blockDim.y << std::endl;
    std::cout << "Threads per block: " << config.threadsPerBlock << std::endl;
    std::cout << "Total threads: " << config.totalThreads << std::endl;
    
    // 调用核函数
    matrixMulRegister<BLOCK_SIZE, SUB_TILE_SIZE>
        <<<config.gridDim, config.blockDim>>>(A, B, C, M, N, K);
}

int main()
{
    const int BLOCK_SIZE = 32;
    const int SUB_TILE_SIZE = 4;
    const int M = 1024;
    const int K = 256;
    const int N = 512;

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // 初始化
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];

    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    

    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < M * N; i++)
    {
        h_C[i] = 0.0f;
    }

    // 分配内存空间
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemset(d_C, 0, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    launchRegisterOptimizedGEMM<BLOCK_SIZE, SUB_TILE_SIZE>(d_A, d_B, d_C, M, N, K);

    // 从GPU转移数据到CPU
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 输出结果
    // for (int i = 0; i < M; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         if (h_C[i * N + j] == 0)
    //         {
    //             continue;
    //         }
    //         std::cout << "row: " << i << ", col: " << j << ", value: " << h_C[i * N + j] << std::endl;
    //     }
    // }

    // 释放内存空间
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}