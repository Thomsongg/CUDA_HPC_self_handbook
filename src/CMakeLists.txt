cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

PROJECT(exercises LANGUAGES CXX CUDA)

add_executable(gemm_shared gemm_shared.cu)