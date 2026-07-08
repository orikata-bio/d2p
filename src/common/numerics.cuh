// SPDX-License-Identifier: Apache-2.0
// numerics.cuh - CUDA numerical primitives for d2p
//
// Shared across all d2p operators. Float32 only.
//
// Usage:
//   #include "common/numerics.cuh"
//   float x = d2p::common::safe_exp(val);

#pragma once
#include <cuda_runtime.h>

namespace d2p {
namespace common {

// ============================================================================
// IMPORTANT: All d2p kernels operate on float32 only.
// These constants and functions are float-specific by design.
// ============================================================================

// Infinity constants (float-specific)
// Use NINF for maximization problems (SW, NW, LCS, MAS)
// Use PINF for minimization problems (DTW, Levenshtein)
inline constexpr float NINF = -1e30f;
inline constexpr float PINF = 1e30f;

// Safe exponential thresholds (float-specific: log(FLT_MAX) ~ 88.7)
inline constexpr float EXP_CLAMP_MIN = -88.0f;
inline constexpr float EXP_CLAMP_MAX = 88.0f;

// Safe exponential - prevents overflow/underflow
// exp(-88) ~ 6e-39 (underflows gracefully to 0)
// exp(88) ~ 1.65e38 (within float32 range)
__host__ __device__ __forceinline__
float safe_exp(float x) {
    if (x < EXP_CLAMP_MIN) return 0.0f;
    if (x > EXP_CLAMP_MAX) x = EXP_CLAMP_MAX;
    return expf(x);
}

// ============================================================================
// Inline logsumexp helpers (temperature-scaled)
// These compute: T * log(sum(exp(x_i / T)))
// ============================================================================

// 2-way logsumexp
__device__ __forceinline__
float logsumexp2(float a, float b, float temp) {
    float max_v = fmaxf(a, b);
    if (max_v <= NINF) return NINF;
    float sum = safe_exp((a - max_v) / temp) + safe_exp((b - max_v) / temp);
    return max_v + temp * logf(sum);
}

// 3-way logsumexp
__device__ __forceinline__
float logsumexp3(float a, float b, float c, float temp) {
    float max_v = fmaxf(fmaxf(a, b), c);
    if (max_v <= NINF) return NINF;
    float sum = safe_exp((a - max_v) / temp)
              + safe_exp((b - max_v) / temp)
              + safe_exp((c - max_v) / temp);
    return max_v + temp * logf(sum);
}

// 4-way logsumexp (used in SW with local alignment start)
__device__ __forceinline__
float logsumexp4(float a, float b, float c, float d, float temp) {
    float max_v = fmaxf(fmaxf(a, b), fmaxf(c, d));
    if (max_v <= NINF) return NINF;
    float sum = safe_exp((a - max_v) / temp)
              + safe_exp((b - max_v) / temp)
              + safe_exp((c - max_v) / temp)
              + safe_exp((d - max_v) / temp);
    return max_v + temp * logf(sum);
}

// ============================================================================
// Softmin variants (for minimization: DTW, Levenshtein)
// These compute: -T * log(sum(exp(-x_i / T))) = min - T*log(sum(exp((min-x_i)/T)))
// ============================================================================

// 3-way softmin
__device__ __forceinline__
float softmin3(float a, float b, float c, float temp) {
    float min_v = fminf(fminf(a, b), c);
    if (min_v >= PINF) return PINF;
    float sum = safe_exp((min_v - a) / temp)
              + safe_exp((min_v - b) / temp)
              + safe_exp((min_v - c) / temp);
    return min_v - temp * logf(sum);
}

}  // namespace common
}  // namespace d2p
