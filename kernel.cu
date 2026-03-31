#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define TIMERSTART(label)                                                      \
  auto start_##label = std::chrono::steady_clock::now();                       \
  auto end_##label = start_##label;                                            \
  double elapsed_##label = 0.0;

#define TIMERSTOP(label)                                                       \
  end_##label = std::chrono::steady_clock::now();                              \
  elapsed_##label =                                                            \
      std::chrono::duration<double>(end_##label - start_##label).count();

// Kernel GPU
__global__ void map_keys_kernel(const uint64_t *__restrict__ keys,
                                uint64_t *__restrict__ part_id, uint64_t P,
                                size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Boundary check
  if (idx < N) {
    const uint64_t prime_number = 0x9E3779B97F4A7C15ULL;
    const uint64_t hash = (keys[idx] * prime_number) >> 32;
    part_id[idx] = hash & (P - 1);
  }
}

extern "C" void map_keys_cuda_timing(const uint64_t *host_keys,
                                     uint64_t *host_part_id, uint64_t P,
                                     size_t N) {
  uint64_t *d_keys = nullptr;
  uint64_t *d_part_id = nullptr;
  size_t bytes = N * sizeof(uint64_t);

  cudaMalloc(&d_keys, bytes);
  cudaMalloc(&d_part_id, bytes);

  // H2D Transfer
  TIMERSTART(h2d)
  cudaMemcpy(d_keys, host_keys, bytes, cudaMemcpyHostToDevice);
  TIMERSTOP(h2d)

  // Setup and Exec Kernel
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  TIMERSTART(kernel_map)
  map_keys_kernel<<<numBlocks, blockSize>>>(d_keys, d_part_id, P, N);
  cudaDeviceSynchronize(); // Attende la fine del kernel [cite: 356]
  TIMERSTOP(kernel_map)

  // D2H Transfer
  TIMERSTART(d2h)
  cudaMemcpy(host_part_id, d_part_id, bytes, cudaMemcpyDeviceToHost);
  TIMERSTOP(d2h)

  cudaFree(d_keys);
  cudaFree(d_part_id);

  std::cout << "Transfer H2D: " << elapsed_h2d << " secs" << std::endl;
  std::cout << "Execution Kernel Map: " << elapsed_kernel_map << " secs"
            << std::endl;
  std::cout << "Transfer D2H: " << elapsed_d2h << " secs" << std::endl;
  std::cout << "Total time: "
            << (elapsed_h2d + elapsed_kernel_map + elapsed_d2h) << " secs"
            << std::endl;
}
