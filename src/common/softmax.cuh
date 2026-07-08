// SPDX-License-Identifier: Apache-2.0
// softmax.cuh - Temperature-scaled softmax/softmin with weight computation
//
// These compute both the logsumexp result AND the normalized weights,
// which are needed for backward pass computations in DP algorithms.
//
// Usage:
//   #include "common/softmax.cuh"
//   float wa, wb, wc;
//   d2p::common::softmax3_weights(a, b, c, temp, wa, wb, wc);

#pragma once
#include <cuda_runtime.h>
#include "numerics.cuh"

namespace d2p {
namespace common {

// ============================================================================
// Softmax with weights (for maximization: SW, NW, LCS, MAS)
// Computes normalized weights: w_i = exp((x_i - max) / T) / sum
// These weights are the gradient contributions in backward pass
// ============================================================================

// 3-way softmax weights
__device__ __forceinline__
void softmax3_weights(float a, float b, float c, float temp,
                      float& wa, float& wb, float& wc) {
    float max_v = fmaxf(fmaxf(a, b), c);
    if (max_v <= NINF) {
        wa = wb = wc = 0.0f;
        return;
    }
    wa = safe_exp((a - max_v) / temp);
    wb = safe_exp((b - max_v) / temp);
    wc = safe_exp((c - max_v) / temp);
    float sum = wa + wb + wc;
    float inv_sum = 1.0f / sum;
    wa *= inv_sum;
    wb *= inv_sum;
    wc *= inv_sum;
}

// 4-way softmax weights (for SW with local alignment start from 0)
__device__ __forceinline__
void softmax4_weights(float a, float b, float c, float d, float temp,
                      float& wa, float& wb, float& wc, float& wd) {
    float max_v = fmaxf(fmaxf(a, b), fmaxf(c, d));
    if (max_v <= NINF) {
        wa = wb = wc = wd = 0.0f;
        return;
    }
    wa = safe_exp((a - max_v) / temp);
    wb = safe_exp((b - max_v) / temp);
    wc = safe_exp((c - max_v) / temp);
    wd = safe_exp((d - max_v) / temp);
    float sum = wa + wb + wc + wd;
    float inv_sum = 1.0f / sum;
    wa *= inv_sum;
    wb *= inv_sum;
    wc *= inv_sum;
    wd *= inv_sum;
}

// 2-way softmax weights
__device__ __forceinline__
void softmax2_weights(float a, float b, float temp,
                      float& wa, float& wb) {
    float max_v = fmaxf(a, b);
    if (max_v <= NINF) {
        wa = wb = 0.0f;
        return;
    }
    wa = safe_exp((a - max_v) / temp);
    wb = safe_exp((b - max_v) / temp);
    float sum = wa + wb;
    float inv_sum = 1.0f / sum;
    wa *= inv_sum;
    wb *= inv_sum;
}

// ============================================================================
// Softmin with weights (for minimization: DTW, Levenshtein, OSA)
// Computes normalized weights: w_i = exp((min - x_i) / T) / sum
// ============================================================================

// 3-way softmin weights
__device__ __forceinline__
void softmin3_weights(float a, float b, float c, float temp,
                      float& wa, float& wb, float& wc) {
    float min_v = fminf(fminf(a, b), c);
    if (min_v >= PINF) {
        wa = wb = wc = 0.0f;
        return;
    }
    wa = safe_exp((min_v - a) / temp);
    wb = safe_exp((min_v - b) / temp);
    wc = safe_exp((min_v - c) / temp);
    float sum = wa + wb + wc;
    float inv_sum = 1.0f / sum;
    wa *= inv_sum;
    wb *= inv_sum;
    wc *= inv_sum;
}

// 4-way softmin weights
__device__ __forceinline__
void softmin4_weights(float a, float b, float c, float d, float temp,
                      float& wa, float& wb, float& wc, float& wd) {
    float min_v = fminf(fminf(a, b), fminf(c, d));
    if (min_v >= PINF) {
        wa = wb = wc = wd = 0.0f;
        return;
    }
    wa = safe_exp((min_v - a) / temp);
    wb = safe_exp((min_v - b) / temp);
    wc = safe_exp((min_v - c) / temp);
    wd = safe_exp((min_v - d) / temp);
    float sum = wa + wb + wc + wd;
    float inv_sum = 1.0f / sum;
    wa *= inv_sum;
    wb *= inv_sum;
    wc *= inv_sum;
    wd *= inv_sum;
}

}  // namespace common
}  // namespace d2p
