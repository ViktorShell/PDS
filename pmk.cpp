#include <chrono>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifndef SEED
#define SEED 634330 // Cool, my unipi id as a seed :)
#endif

// Sorry I didn't wanted to include the hpc_helper :/
#define TIMERSTART(label)                                                      \
  auto start_##label = std::chrono::steady_clock::now();                       \
  auto end_##label = start_##label;                                            \
  double elapsed_##label = 0.0;

#define TIMERSTOP(label)                                                       \
  end_##label = std::chrono::steady_clock::now();                              \
  elapsed_##label =                                                            \
      std::chrono::duration<double>(end_##label - start_##label).count();

std::vector<uint64_t> generate_keys(size_t N) {
  std::vector<uint64_t> keys(N);
  std::mt19937_64 gen(SEED);
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

  for (size_t i = 0; i < N; ++i) {
    keys[i] = dist(gen);
  }

  return keys;
}

uint64_t checksum(const uint64_t *vec, size_t size) {
  const uint64_t FNV_offset_basis = 0xcbf29ce484222325ULL; // random
  const uint64_t FNV_prime = 11972354906962701971ULL;      // large prime number

  // conversion to ascii
  uint64_t hash = FNV_offset_basis;
  const unsigned char *bytes = reinterpret_cast<const unsigned char *>(vec);
  size_t num_bytes = size * sizeof(uint64_t);

  for (size_t i = 0; i < num_bytes; ++i) {
    hash ^= bytes[i];
    hash *= FNV_prime;
  }

  return hash;
}

#ifdef CUDA
extern "C" void map_keys_cuda_timing(const uint64_t *host_keys,
                                     uint64_t *host_part_id, uint64_t P,
                                     size_t N);
#else

// It seems that always_inline have close to the same execution time as inline
__attribute__((always_inline)) inline void
map_keys(const uint64_t *__restrict__ keys, uint64_t *__restrict__ part_id,
         uint64_t P, size_t N) {
  const uint64_t prime_number = 0x9E3779B97F4A7C15ULL;

#ifdef AVX

  __m256i _prime = _mm256_set1_epi64x(prime_number);
  __m256i _mask = _mm256_set1_epi64x(P - 1);

  // precompute the upper part of the prime number for the emulation
  __m256i _prime_h = _mm256_srli_epi64(_prime, 32);

  size_t i = 0;
  for (; i + 4 <= N; i += 4) {
    __m256i _keys = _mm256_loadu_si256((const __m256i *)&keys[i]);

    // Emulation of the mul 64-bit (K * P)
    // 1. K_L * P_L
    __m256i mul_ll = _mm256_mul_epu32(_keys, _prime);

    // 2. K_H * P_L
    __m256i keys_h = _mm256_srli_epi64(_keys, 32);
    __m256i mul_hl = _mm256_mul_epu32(keys_h, _prime);

    // 3. K_L * P_H
    __m256i mul_lh = _mm256_mul_epu32(_keys, _prime_h);

    // 4. Cross sum ((K_H * P_L) + (K_L * P_H)) << 32
    __m256i cross_sum = _mm256_add_epi64(mul_hl, mul_lh);
    __m256i cross_shifted = _mm256_slli_epi64(cross_sum, 32);

    // 5. Result of the mul as 64bits
    __m256i _hash = _mm256_add_epi64(mul_ll, cross_shifted);

    _hash = _mm256_srli_epi64(_hash, 32);
    __m256i _and_result = _mm256_and_si256(_hash, _mask);

    // Save the result
    _mm256_storeu_si256((__m256i *)&part_id[i], _and_result);
  }

  // THe remaining part not multiple of 4
  for (; i < N; i++) {
    uint64_t key = keys[i];
    uint64_t hash = (key * prime_number) >> 32;
    part_id[i] = hash & (P - 1);
  }

#else

  // Baseline / autovectorization
  for (size_t i = 0; i < N; i++) {
    uint64_t key = keys[i];
    const uint64_t hash = (key * prime_number) >> 32;
    part_id[i] = hash & (P - 1);
  }

#endif
}

// Deriving mean and std deviation
void mean_and_std_deviation(const uint64_t *__restrict__ keys,
                            uint64_t *__restrict__ part_id, size_t times,
                            size_t N, uint64_t P) {
  // Mean
  std::vector<double> times_vec(N);
  double acc = 0.0;
  for (size_t i = 0; i < N; ++i) {
    // Mapping
    TIMERSTART(exec_time);
    map_keys(keys, part_id, P, N);
    TIMERSTOP(exec_time);
    times_vec[i] = elapsed_exec_time;
    acc += elapsed_exec_time;
  }
  double mean = acc / static_cast<double>(times);

  // Std Deviation
  double sum_squared_dev = 0.0;
  for (const auto &item : times_vec) {
    double deviation = item - mean;
    sum_squared_dev += deviation * deviation;
  }

  double std_dev = std::sqrt(sum_squared_dev / (N - 1));
  std::cout << "Mean: " << mean << std::endl;
  std::cout << "Std_dev: " << std_dev << std::endl;
}

// Checking variation of time based on the varbiels N and P
void sweep_n_p(const uint64_t *__restrict__ keys,
               uint64_t *__restrict__ part_id, size_t N, uint64_t P) {
  std::cout << "N;P;ELAPSED_TIME,THROUGHPUT" << std::endl;
  for (size_t n = N; n < 10000000000 + 1; n = n * 10) {
    for (size_t p = P; p < (n / 2) + 1; p = p * 2) {
      // Mapping
      TIMERSTART(exec_time);
      map_keys(keys, part_id, P, N);
      TIMERSTOP(exec_time);
      std::cout << std::fixed << std::setprecision(9) << n << ";" << p << ";"
                << elapsed_exec_time << ";"
                << (long int)(static_cast<double>(N) / elapsed_exec_time)
                << std::endl;
    }
  }
}

#endif

int main() {
  // Sizes
  size_t N = 100000;
  uint64_t P = 1024;
  size_t TIMES = 10;

#ifdef BASELINE
  std::cout << "=== PMK (BASELINE) ===" << std::endl;
#elif AVX
  std::cout << "=== PMK (AVX) ===" << std::endl;
#elif CUDA
  std::cout << "=== PMK (CUDA) ===" << std::endl;
#else
  std::cout << "=== PMK (auto vectorization) ===" << std::endl;
#endif

  // Linear allocation
  std::vector<uint64_t> keys = generate_keys(N);
  std::vector<uint64_t> part_id(N);

#ifdef CUDA
  map_keys_cuda_timing(keys.data(), part_id.data(), P, N);
#else
  // Data gathering
  mean_and_std_deviation(keys.data(), part_id.data(), TIMES, N, P);
  sweep_n_p(keys.data(), part_id.data(), N, P);
#endif

  std::cout << "Seed: " << SEED << std::endl;
  std::cout << "Keys checksum: " << checksum(keys.data(), keys.size())
            << std::endl;
  std::cout << "Part_id checksum: " << checksum(part_id.data(), part_id.size())
            << std::endl;

  return 0;
}
