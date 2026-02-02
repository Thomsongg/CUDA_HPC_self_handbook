cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(exercises LANGUAGES CXX CUDA)

# 设置 C++ 标准为 17 (CUTLASS 必备)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 显卡架构 Ada 89
set(CMAKE_CUDA_ARCHITECTURES 89)

set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/cutlass")

add_executable(gemm_cutlass_tensor_core gemm_cutlass_tensor_core.cu)

target_include_directories(gemm_cutlass_tensor_core PRIVATE 
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)

# 针对 MSVC 优化选项
if(MSVC)
    target_compile_options(gemm_cutlass_tensor_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=/utf-8
    >)
else()
    target_compile_options(gemm_cutlass_tensor_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-Wno-unused-function
    >)
endif()
