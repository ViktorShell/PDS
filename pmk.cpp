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

// It seems that always_inline have close to the same execution time as inline
__attribute__((always_inline)) inline void
map_keys(const uint64_t *__restrict__ keys, uint64_t *__restrict__ part_id,
         uint64_t P, size_t N) {
  const uint64_t prime_number = 0x9E3779B97F4A7C15ULL;

#ifdef AVX
  __m256i _prime = _mm256_set1_epi64x(prime_number);
  __m256i _mask = _mm256_set1_epi64x(P - 1);

  // Precalcoliamo la parte alta del numero primo per l'emulazione
  __m256i _prime_h = _mm256_srli_epi64(_prime, 32);

  size_t i = 0;
  for (; i + 4 <= N; i += 4) {
    __m256i _keys = _mm256_loadu_si256((const __m256i *)&keys[i]);

    // --- Emulazione Moltiplicazione 64-bit (K * P) ---
    // 1. K_L * P_L
    __m256i mul_ll = _mm256_mul_epu32(_keys, _prime);

    // 2. K_H * P_L
    __m256i keys_h = _mm256_srli_epi64(_keys, 32);
    __m256i mul_hl = _mm256_mul_epu32(keys_h, _prime);

    // 3. K_L * P_H
    __m256i mul_lh = _mm256_mul_epu32(_keys, _prime_h);

    // 4. Somma incrociata ((K_H * P_L) + (K_L * P_H)) << 32
    __m256i cross_sum = _mm256_add_epi64(mul_hl, mul_lh);
    __m256i cross_shifted = _mm256_slli_epi64(cross_sum, 32);

    // 5. Risultato finale della moltiplicazione a 64-bit
    __m256i _hash = _mm256_add_epi64(mul_ll, cross_shifted);
    // --------------------------------------------------

    _hash = _mm256_srli_epi64(_hash, 32);
    __m256i _and_result = _mm256_and_si256(_hash, _mask);

    // Correzzione: storeu invece di store per prevenire i SegFault
    _mm256_storeu_si256((__m256i *)&part_id[i], _and_result);
  }

  for (; i < N; i++) {
    uint64_t key = keys[i];
    uint64_t hash = (key * prime_number) >> 32;
    part_id[i] = hash & (P - 1);
  }

  // const uint64_t mask_val = P - 1;
  //
  // // Load vectors with prime number and the mask 4 x 64-bit
  // __m256i v_prime = _mm256_set1_epi64x(prime_number);
  // __m256i v_mask = _mm256_set1_epi64x(mask_val);
  //
  // // Prepare the cross product for the mul operation
  // __m256i v_prime_swap = _mm256_shuffle_epi32(v_prime, _MM_SHUFFLE(2, 3, 0,
  // 1));
  //
  // size_t i = 0;
  //
  // for (; i + 4 <= N; i += 4) {
  //   // Load 4 keys
  //   __m256i v_keys = _mm256_loadu_si256((const __m256i *)&keys[i]);
  //   __m256i prod_hi = _mm256_mullo_epi32(v_keys, v_prime_swap);
  //   __m256i prod_hi_sum = _mm256_add_epi32(
  //       prod_hi, _mm256_shuffle_epi32(prod_hi, _MM_SHUFFLE(2, 3, 0, 1)));
  //   __m256i prod_hi_shifted = _mm256_slli_epi64(prod_hi_sum, 32);
  //   __m256i prod_lo = _mm256_mul_epu32(v_keys, v_prime);
  //   __m256i v_hash = _mm256_add_epi64(prod_lo, prod_hi_shifted);
  //   v_hash = _mm256_srli_epi64(v_hash, 32);
  //   v_hash = _mm256_and_si256(v_hash, v_mask);
  //
  //   // Conversion to 4x 32-bit
  //   __m256i shuf = _mm256_shuffle_epi32(v_hash, _MM_SHUFFLE(2, 0, 2, 0));
  //   __m128i lo = _mm256_castsi256_si128(shuf); // Prende i 128 bit inferiori
  //   __m128i hi =
  //       _mm256_extracti128_si256(shuf, 1); // Prende i 128 bit superiori
  //   __m128i v_part_id = _mm_unpacklo_epi64(lo, hi);
  //
  //   // Save to memory
  //   _mm_storeu_si128((__m128i *)&part_id[i], v_part_id);
  // }
  //
  // // Handling of the remaining item in case the keys are not multiple of 4
  // for (; i < N; i++) {
  //   uint64_t key = keys[i];
  //   uint64_t hash = (key * prime_number) >> 32;
  //   part_id[i] = hash & mask_val;
  // }

#else

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

int main() {
  // Sizes
  size_t N = 100000;
  uint64_t P = 1024;
  size_t TIMES = 10;

#ifdef BASELINE
  std::cout << "=== PMK (baseline) ===" << std::endl;
#elif AVX
  std::cout << "=== PMK (avx) ===" << std::endl;
#else
  std::cout << "=== PMK (auto vectorization) ===" << std::endl;
#endif

  // Linear allocation
  std::vector<uint64_t> keys = generate_keys(N);
  std::vector<uint64_t> part_id(N);

  // Data gathering
  mean_and_std_deviation(keys.data(), part_id.data(), TIMES, N, P);
  sweep_n_p(keys.data(), part_id.data(), N, P);

  std::cout << "Seed: " << SEED << std::endl;
  std::cout << "Keys checksum: " << checksum(keys.data(), keys.size())
            << std::endl;
  std::cout << "Part_id checksum: " << checksum(part_id.data(), part_id.size())
            << std::endl;

  return 0;
}
