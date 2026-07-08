// SPDX-License-Identifier: Apache-2.0
// numerics.h - CPU numerical primitives for d2p
//
// Shared across all d2p CPU operators. Float32 only.
// Values MUST match numerics.cuh exactly for CPU/CUDA parity.
//
// Usage:
//   #include "common/numerics.h"
//   float x = d2p::common::safe_exp(val);

#pragma once
#include <cmath>
#include <algorithm>

namespace d2p {
namespace common {

// ============================================================================
// IMPORTANT: These values MUST match numerics.cuh exactly for CPU/CUDA parity.
// All d2p operations are float32 only.
// ============================================================================

inline constexpr float NINF = -1e30f;
inline constexpr float PINF = 1e30f;
inline constexpr float EXP_CLAMP_MIN = -88.0f;
inline constexpr float EXP_CLAMP_MAX = 88.0f;

inline float safe_exp(float x) {
    if (x < EXP_CLAMP_MIN) return 0.0f;
    if (x > EXP_CLAMP_MAX) x = EXP_CLAMP_MAX;
    return std::exp(x);
}

// ============================================================================
// Kahan compensated summation for numerical stability
// Use for accumulating many small values (common in CPU DP)
// ============================================================================

struct KahanSum {
    float sum = 0.0f;
    float c = 0.0f;  // Compensation for lost low-order bits

    inline void add(float val) {
        float y = val - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    inline float result() const { return sum; }

    inline void reset() {
        sum = 0.0f;
        c = 0.0f;
    }
};

// ============================================================================
// CPU logsumexp helpers (temperature-scaled)
// Using Kahan summation for numerical stability
// ============================================================================

inline float logsumexp2(float a, float b, float temp) {
    float max_v = std::max(a, b);
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
    return max_v + temp * std::log(sum.result());
}

inline float logsumexp3(float a, float b, float c, float temp) {
    float max_v = std::max({a, b, c});
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
    sum.add(safe_exp((c - max_v) / temp));
    return max_v + temp * std::log(sum.result());
}

inline float logsumexp4(float a, float b, float c, float d, float temp) {
    float max_v = std::max({a, b, c, d});
    if (max_v <= NINF) return NINF;
    KahanSum sum;
    sum.add(safe_exp((a - max_v) / temp));
    sum.add(safe_exp((b - max_v) / temp));
    sum.add(safe_exp((c - max_v) / temp));
    sum.add(safe_exp((d - max_v) / temp));
    return max_v + temp * std::log(sum.result());
}

// ============================================================================
// CPU softmin (for minimization: DTW, Levenshtein)
// ============================================================================

inline float softmin3(float a, float b, float c, float temp) {
    float min_v = std::min({a, b, c});
    if (min_v >= PINF) return PINF;
    KahanSum sum;
    sum.add(safe_exp((min_v - a) / temp));
    sum.add(safe_exp((min_v - b) / temp));
    sum.add(safe_exp((min_v - c) / temp));
    return min_v - temp * std::log(sum.result());
}

}  // namespace common
}  // namespace d2p
