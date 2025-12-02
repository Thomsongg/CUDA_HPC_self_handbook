# CUDA_HPC_self_handbook
CUDA C++高性能计算学习笔记，包含知识文档、Cuda c++程序源码、性能分析结果

## 📁 项目目录结构

├── docs/ # 项目文档 </br>
├── experiments/ # 实验与性能分析 </br>
├── results/ # 测试与实验结果 </br>
└── src/ # 源代码

## 📚 docs - 知识文档
存放有关 CUDA 高性能计算 及 AI Infra 全栈知识，软硬件兼具。

- **CUDA C++开发指南**：介绍 CUDA C++ 高性能计算开发相关知识，包含 CUDA 架构、经典案例和 Thrust 并行库等部分，并嵌入各重要功能点示例代码，存入 handbook.md。
- **硬件基础**：详细介绍 GPU 和 CUDA 硬件基础知识，存入 HardwareBasics.md
- **CUDA与现代C++**：介绍 C++新特性（11/17）在 CUDA 高性能计算的应用，如 RAII、智能指针、Lambda 表达式、if constexpr 等重要特性。
- **其他文档**：包含学习目标、面试准备等文档。

## 🔬 experiments - 实验与分析
使用 Nsight Compute (UI) 对 CUDA C++ 程序进行性能测试产生的文件，用于分析程序性能瓶颈，并针对性优化。

- **吞吐量 (Throughput)**：GPU 各器件吞吐量、计算与内存吞吐量、不同精度的 Roofline 模型
- **计算强度 (Compute Workload)**：
- **内存负载 (Compute Workload)**：
- **启动数据 (Launch Statistics)**：
- **占用率 (Occupancy)**：

## 📊 results - 运行结果
保存 CUDA 原程序及优化后的测试结果。


## 💻 src - 源代码
项目的主要源代码目录，包含大型且重要的项目代码。

- **矩阵乘法 (GEMM)**
- **规约 (Reduction)**
- **转置 (Transpose)**
- **CUDA Stream 应用**
- **CUDA C++ 新特性**

