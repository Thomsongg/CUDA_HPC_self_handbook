#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

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
using ElementA = float;         
using ElementB = float;         
using ElementC = float;         
using ElementAccumulator = float; 

using LayoutA = cutlass::layout::RowMajor; 
using LayoutB = cutlass::layout::RowMajor; 
using LayoutC = cutlass::layout::RowMajor; 

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,      // SIMT (CUDA Cores)
    cutlass::arch::Sm80,                // 兼容 Ampere/Ada 架构     Ada 架构建议使用 Sm89
    cutlass::gemm::GemmShape<128, 128, 8>, 
    cutlass::gemm::GemmShape<32, 64, 8>,   
    cutlass::gemm::GemmShape<1, 1, 1>,     
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 1, ElementAccumulator, ElementAccumulator>, 
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    2                                  
>;

int main() {
    // 1. 定义矩阵维度 (必须为 128 的倍数)
    int M = 512;
    int N = 512;
    int K = 512;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::cout << "Matrix Dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // 2. 分配主机内存
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<ElementA> h_A(size_A);
    std::vector<ElementB> h_B(size_B);
    std::vector<ElementC> h_C(size_C);
    std::vector<ElementC> h_C_ref(size_C, 0.0f);

    // 随机初始化数据
    std::default_random_engine gen;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : h_A) x = dist(gen);
    for (auto& x : h_B) x = dist(gen);

    // 3. 分配显存
    ElementA *d_A;
    ElementB *d_B;
    ElementC *d_C;

    checkCuda(cudaMalloc(&d_A, size_A * sizeof(ElementA)));
    checkCuda(cudaMalloc(&d_B, size_B * sizeof(ElementB)));
    checkCuda(cudaMalloc(&d_C, size_C * sizeof(ElementC)));

    checkCuda(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(ElementA), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(ElementB), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_C, 0, size_C * sizeof(ElementC)));

    // 4. CUTLASS 构造参数
    typename Gemm::Arguments args(
        {M, N, K},              // {维度}
        {d_A, K},               // {指针, 步长}
        {d_B, N},            
        {d_C, N},               // {C 输入指针, 步长}
        {d_C, N},               // {C 输出指针, 步长}
        {alpha, beta}        
    );

    // 5. 初始化并运行 CUTLASS Op
    Gemm gemm_op;
    
    // 检查硬件适配性
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS cannot implement this kernel configuration." << std::endl;
        return -1;
    }

    // 执行 CUTLASS 算子
    status = gemm_op(args);
    checkCuda(cudaDeviceSynchronize());

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM execution failed." << std::endl;
        return -1;
    }

    checkCuda(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(ElementC), cudaMemcpyDeviceToHost));

    std::cout << "CUTLASS GEMM completed successfully." << std::endl;

    // 验证前几个元素
    std::cout << "First 5 elements of Result: ";
    for(int i=0; i<5; ++i) std::cout << h_C[i] << " ";
    std::cout << std::endl;

    // 释放资源
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_B));
    checkCuda(cudaFree(d_C));

    return 0;
}
