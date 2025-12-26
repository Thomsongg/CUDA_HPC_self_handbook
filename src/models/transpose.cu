template<int BLOCK_SIZE>
__global__ void transpose_shared(float* input, float* output, int M, int N)
{
    int block_idx = threadIdx.x * BLOCK_SIZE;
    int block_idy = threadIdx.y * BLOCK_SIZE;
    int former_idx = block_idx + threadIdx.x;
    int former_idy = block_idy + threadIdx.y;
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE + 1] = {0.0f};

    if (former_idy < M || former_idx < N)
    {
        smem[threadIdx.y][threadIdx.x] = input[former_idy * N + former_idx];
    }
    __syncthreads();

    // 计算新的 idx, idy
    // 此时新的矩阵变为 output[N][M]
    int new_idx = block_idy + threadIdx.x;
    int new_idy = block_idx + threadIdx.y;
    output[new_idy * M + new_idx] = 0.0f;
    if (new_idy < M || new_idx < N)
    {
        output[new_idy * M + new_idx] = smem[threadIdx.x][threadIdx.y];
    }
}

template<int BLOCK_SIZE>
__global__ void transpose_swizzling(float* input, float* output, int M, int N)
{
    int block_idx = blockIdx.x * BLOCK_SIZE;
    int block_idy = blockIdx.y * BLOCK_SIZE;
    int former_idx = block_idx + threadIdx.x;
    int former_idy = block_idy + threadIdx.y;
    __shared__ float smem[BLOCK_SIZE][BLOCK_SIZE] = {0.0f};

    if (former_idy < M && former_idx < N)
    {
        // 异或 Swizzling
        smem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[former_idy * N + former_idx];
    }
    __syncthreads();

    // 转置后也做异或处理
    int new_idx = block_idy + threadIdx.x;
    int new_idy = block_idx + threadIdx.x;
    if (new_idy < N && new_idx < M)
    {
        output[new_idy * M + new_idx] = smem[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}

const int BLOCK_SIZE = 32;
dim3 blockSize(32, 32);
dim3 gridSize(CEIL(N ,32), CEIL(M, 32));
transpose_shared<BLOCK_SIZE><<<gridSize, blockSize>>>(d_input, d_output, M, N);
