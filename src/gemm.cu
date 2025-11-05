#include <stdio.h>
#include <time.h>
#include <iostream>

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define CEIL(a, b) ((a + b - 1) / b)
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])

template<int M, int K, int N, int BLOCK_SIZE>
__global__ void gemm_shared(float *A, float *B, float *C)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // 当前线程块控制的矩阵A的行、矩阵B的列位置
    int A_row = blockIdx.y * BLOCK_SIZE + ty;
    int B_col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;

    if (A_row >= M || B_col >= N)
    {
        return;
    }

    // 基本思想：k分割
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
    {
        // 将矩阵A分割为小份，转移到共享内存As
        int A_col = t * BLOCK_SIZE + tx;
        if (A_col <= K)
        {
            As[ty][tx] = A[A_row * K + A_col];
        }

        // 转移矩阵B到Bs
        int B_row = t * BLOCK_SIZE + ty;
        if (B_row <= K)
        {
            Bs[ty][tx] = B[B_row * K + B_col];
        }

        __syncthreads();

        // 计算Cs一个元素的点积
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 输出当前线程计算结果
    C[A_row * M + B_col] = sum;
}

int main()
{
    const int BLOCK_SIZE = 32;
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

    // 调用核函数
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(M * K, BLOCK_SIZE), CEIL(K * N, BLOCK_SIZE));
    gemm_shared<M, K, N, BLOCK_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C);

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