#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// 添加半精度所需的头文件
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/tensor_view_io.h"

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

// --- CUTLASS GEMM 配置 ---
// 使用 cutlass::half_t, 比 float 获得更快的执行速度
using ElementA = cutlass::half_t;         
using ElementB = cutlass::half_t;         
using ElementC = cutlass::half_t;         
using ElementAccumulator = cutlass::half_t; 

using LayoutA = cutlass::layout::RowMajor; 
using LayoutB = cutlass::layout::RowMajor; 
using LayoutC = cutlass::layout::RowMajor; 

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,    // Tensor Core 加速
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, // 加大了 K 维分块以适配 Tensor Core
    cutlass::gemm::GemmShape<64, 64, 32>,   
    cutlass::gemm::GemmShape<16, 8, 16>,    // Tensor Core 硬件指令形状
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / 16, ElementAccumulator, ElementAccumulator>, 
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3                                  // 增加 Stages 以适配 Tensor Core 模式
>;

int main() {
    // 1. 定义矩阵维度
    int M = 512;
    int N = 512;
    int K = 512;
    float alpha_f = 1.0f;
    float beta_f = 0.0f;

    // CUTLASS 参数需要对应的数据类型
    ElementAccumulator alpha = ElementAccumulator(alpha_f);
    ElementAccumulator beta = ElementAccumulator(beta_f);

    std::cout << "Tensor Core GEMM (FP16) - M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 2. 准备主机内存
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    // 注意：half_t 在主机端需要转换
    std::vector<ElementA> h_A(size_A);
    std::vector<ElementB> h_B(size_B);
    std::vector<ElementC> h_C(size_C);

    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h_A) x = ElementA(dist(gen));
    for (auto& x : h_B) x = ElementB(dist(gen));

    // 3. 准备设备内存
    ElementA *d_A;
    ElementB *d_B;
    ElementC *d_C;

    checkCuda(cudaMalloc(&d_A, size_A * sizeof(ElementA)));
    checkCuda(cudaMalloc(&d_B, size_B * sizeof(ElementB)));
    checkCuda(cudaMalloc(&d_C, size_C * sizeof(ElementC)));

    checkCuda(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(ElementA), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(ElementB), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_C, 0, size_C * sizeof(ElementC)));

    // 4. 构建参数
    typename Gemm::Arguments args(
        {M, N, K},           
        {d_A, K},            
        {d_B, N},            
        {d_C, N},            
        {d_C, N},            
        {alpha, beta}        
    );

    // 5. 运行
    Gemm gemm_op;
    cutlass::Status status = gemm_op(args);
    checkCuda(cudaDeviceSynchronize());

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM execution failed." << std::endl;
        return -1;
    }

    // 6. 拷贝结果
    checkCuda(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(ElementC), cudaMemcpyDeviceToHost));
    std::cout << "CUTLASS Tensor Core GEMM completed." << std::endl;

    // 7. 打印前 5 个结果 (转换为 float 打印)
    std::cout << "First 5 elements: ";
    for(int i=0; i<5; ++i) std::cout << float(h_C[i]) << " ";
    std::cout << std::endl;

    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return 0;
}
