# Parallel Mapping Kernel (PMK)

This project implements and evaluates various versions of a Parallel Mapping Kernel (PMK) designed to map keys to partition IDs. The project explores performance across different architectures and optimization levels, including Baseline (scalar), Auto-vectorized, Manual AVX2 Intrinsics, and CUDA (GPU) implementations.

## Features

- **CPU Implementations**: Baseline, Auto-vectorized, and manual AVX2.
- **GPU Implementations**: CUDA versions (including shared memory optimizations).
- **Correctness Verification**: FNV-1a checksum comparison across all implementations.
- **Profiling**: Automated collection of mean execution time and standard deviation.

---

## Getting Started

The project uses a `Makefile` to manage compilation, execution, and reporting.

### Prerequisites

- A C++ compiler (e.g., `g++`) with support for C++17 and AVX2.
- NVIDIA CUDA Toolkit (for GPU versions).
- Access to a compute node (e.g., node09) for accurate performance measurements.

### Build Commands

You can manage the project using the following commands:

#### 1. Compile the Project

To compile all the source files and generate the executables:

```bash
make all
```

#### 2. View Help Guide

To see a detailed list of all available commands and their descriptions:

```bash
make help
```

##### 3. Full Execution and Execution Time Generation

To compile the project and immediately run all performance tests (mean/std deviation analysis and N/P sweeps) on the current node:

```bash
make exec_all
```

*This command automates the entire workflow: it builds the binaries, executes them, and saves the output data for further analysis*.

---

## Optimization and Verification

#### Compiler Flags

The project utilizes specific flags to control and monitor performance:

- **Baseline**: Compiled with `-fno-tree-vectorize` to ensure scalar execution.
- **Auto-vectorization**: Enabled via `-O3 -march=native -mavx2 -ffast-math`.
- **Reporting**: The `-fopt-info-vec-all` flag is used to document optimization successes in the logs.

#### Correctness

Every execution is verified using a checksum.
Executing the single binary will generate the checksum for both the **keys** array and the **part_id** array.
If you execute the `make exec_all` or any other command that starts with `exec_` than the outputs is saved inside the logs folder.
For a standard run with **SEED 634330**, the expected values are:

- **Keys Checksum**: 703482161565084902
- **Part_id Checksum**: 10053871536273375668

---

## Performance Analysis

The project outputs detailed metrics including:

- **H2D/D2H Transfer Times**: Memory latency between Host and Device.
- **Kernel Execution Time**: Pure computation time on the GPU/CPU.
- **Throughput**: Measured in elements processed per second.
