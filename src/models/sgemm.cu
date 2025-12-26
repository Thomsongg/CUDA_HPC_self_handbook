#define CEIL(a, b) ((a + b - 1) / b)

__global__ void sgemm_naive(float *mA, float *mB, float *mC, int M, int K, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    float accum = 0.0f;
    if (idy < M || idx < N)
    {
        // 直接在全局内存中计算
        // 用循环计算每行的点积和
        for (int i = 0; i < K; i++)
        {
            // mA 定行变列； mB 定列变行
            accum += mA[idy * N + i] * mB[i * N + idx];
        }
    }

    // 最后输出到 output 的对应行
    mC[idy * N + idx] = accum;
}

// Block_tile 方法
template<const int BLOCK_SIZE>
__global__ void sgemm_shared(float *mA, float *mB, float *mC, int M, int K, int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    __shared__ float smem_A[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float smem_B[BLOCK_SIZE][BLOCK_SIZE + 1];

    if (idy >= M || idx >= N)
    {
        return;
    }

    float accum = 0.0f;

    // mA: 定行变列; mB: 定列变行
    int A_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int B_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 对全局内存分块, 存入共享内存
    for (int t = 0; t < CEIL(K, BLOCK_SIZE))
    {
        // mA 分块后迁移至 smem_A
        int A_col = t * BLOCK_SIZE + threadIdx.x;
        smem_A[threadIdx.y][threadIdx.x] = (A_row < M && A_col < K) ? mA[A_row * K + A_col] ? 0.0f;

        int B_row = t * BLOCK_SIZE + threadIdx.y;
        smem_B[threadIdx.y][threadIdx.x] = (B_row < K && B_col < N) ? mb[B_row * N + B_col] ? 0.0f;

        __syncthreads();

        // 计算点积和
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            accum += smem_A[threadIdx.y][k] * smem_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    // 输出结果到 mC
    mC[idy * N + idx] = (idy < M && idy < N) ? accum : 0.0f;
}

template<int BLOCK_SIZE, int SUB_TILE_SIZE>
__global__ void sgemm_register(float *mA, float *mB, float *mC, int M, int K, int N)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty * SUB_TILE_SIZE;
    int col = blockIdx.x * BLOCK_SIZE + tx * SUB_TILE_SIZE;
    __shared__ float smem_A[BLOCK_SIZE][BLOCK_SIZE] = {0.0f};
    __shared__ float smem_B[BLOCK_SIZE][BLOCK_SIZE] = {0.0f};

    // 分配寄存器
    float reg[SUB_TILE_SIZE][SUB_TILE_SIZE] = {0.0f};

    // 大循环：Block_tile
    for (int t = 0; t < CEIL(K, BLOCK_SIZE); t++)
    {
        // 小循环：Thread_tile
        // 将 mA, mB 继续分割成一个个小块，每个 thread 处理一个小块，迁移到 shared_memory
        // 再根据每个 shared_memory 的横行，分割为一纵列，每个 thread 对应列方向的一小行块
        #pragma unroll
        for (int i = 0; i < SUB_TILE_SIZE; i++)
        {
            if (row + i < M && t * BLOCK_SIZE + tx < K)
            {
                smem_A[ty * SUB_TILE_SIZE + i][tx] = mA[(row + i) * K + (t * BLOCK_SIZE + tx)];
            }
        }

        // 对 mB 再进行处理
        #pragma unroll
        for (int j = 0; j < SUB_TILE_SIZE; j++)
        {
            if (t * BLOCK_SIZE + ty < K && col + j < N)
            {
                smem_B[ty][tx * SUB_TILE_SIZE + j] = mB[(t * BLOCK_SIZE + ty) * N + (col + j)];
            }
        }

        __syncthreads();

        // 将 shared_memory 的结果迁移至 register 进行计算
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            #pragma unroll
            for (int i = 0; i < SUB_TILE_SIZE; i++)
            {
                #pragma unroll
                for (int j = 0; j < SUB_TILE_SIZE; j++)
                {
                    reg[i][j] += smem_A[ty * SUB_TILE_SIZE + i][k] * smem_B[k][tx * SUB_TILE_SIZE + j];
                }
            }
        }
        __syncthreads();
    }

    // 输出 reg 计算结果到 mC
    #pragma unroll
    for (int i = 0; i < SUB_TILE_SIZE; i++)
    {
        #pragma unroll
        for (int j = 0; j < SUB_TILE_SIZE; j++)
        {
            mC[(row + i) * N + (col + j)] = reg[i][j];
        }
    }
}

const int BLOCK_SIZE = 32;
dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

// sgemm_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
sgemm_shared<BLOCK_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
