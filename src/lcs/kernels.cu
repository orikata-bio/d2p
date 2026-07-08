// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft LCS (Longest Common Subsequence) CUDA Kernel Implementations
 *
 * Implements anti-diagonal wavefront parallelization for LCS.
 * Uses softmax (maximization) - simpler than Levenshtein as no gap costs.
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>

namespace d2p {
namespace lcs {

// ============================================================================
// Constants and Device Helpers
// ============================================================================

#define WARP_SIZE 32

// Safe exponential with bounds for float32
template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < (T)-88.0f) return (T)0.0f;
    if (x > (T)88.0f) x = (T)88.0f;
    return exp(x);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum_lcs(T v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum_lcs(T v) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum_lcs(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : (T)0.0f;
    if (wid == 0) v = warp_reduce_sum_lcs(v);
    return v;
}

// Softmax for 3 values: max + T * log(sum exp((x - max)/T))
__device__ __forceinline__ float softmax3(float a, float b, float c, float T) {
    float m = fmaxf(fmaxf(a, b), c);
    if (m <= NINF) return NINF;

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;
    float ec = (c > NINF) ? safe_exp((c - m) / T) : 0.0f;

    float sum = ea + eb + ec;
    if (sum <= 0.0f) return NINF;
    return m + T * logf(sum);
}

// Softmax weights: w_k = exp((a_k - m)/T) / sum exp((a_j - m)/T)
__device__ __forceinline__ void softmax3_weights(
    float a, float b, float c, float T,
    float& wa, float& wb, float& wc
) {
    float m = fmaxf(fmaxf(a, b), c);
    if (m <= NINF) {
        wa = wb = wc = 0.0f;
        return;
    }

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;
    float ec = (c > NINF) ? safe_exp((c - m) / T) : 0.0f;

    float total = ea + eb + ec;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
        wc = ec / total;
    } else {
        wa = wb = wc = 0.0f;
    }
}

// ============================================================================
// FORWARD PASS KERNELS
// ============================================================================

/**
 * Initialize alpha with LCS base cases:
 * alpha(0,0) = 0
 * alpha(i,0) = 0 for i > 0 (no matches possible with empty seq2)
 * alpha(0,j) = 0 for j > 0 (no matches possible with empty seq1)
 * All other cells = -inf (will be filled by DP)
 */
__global__ void lcs_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / (max_L2 + 1);
    int j   = rem % (max_L2 + 1);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Cells outside actual lengths get -inf
    if (i > L1 || j > L2) {
        alpha[idx] = NINF;
        return;
    }

    // Base cases for LCS: all boundary cells are 0
    if (i == 0 || j == 0) {
        alpha[idx] = 0.0f;
    } else {
        // Interior cells initialized to -inf (will be computed by DP)
        alpha[idx] = NINF;
    }
}

/**
 * Forward DP for one anti-diagonal k = i + j
 * Three transitions: match (diagonal), skip1 (up), skip2 (left)
 */
__global__ void lcs_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride_alpha = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - cells outside actual lengths get -inf
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * ((size_t)max_L2 + 1) + j;
            a[idx] = NINF;
            continue;
        }

        size_t stride = (size_t)max_L2 + 1;
        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float match_score = s[score_idx];

        // Get predecessor values
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        // LCS recurrence:
        // v_match = alpha[i-1,j-1] + scores[i-1,j-1] (match/add to LCS)
        // v_skip1 = alpha[i-1,j] (skip seq1[i])
        // v_skip2 = alpha[i,j-1] (skip seq2[j])
        float v_match = a_diag + match_score;
        float v_skip1 = a_up;
        float v_skip2 = a_left;

        a[idx] = softmax3(v_match, v_skip1, v_skip2, T);
    }
}

/**
 * Extract LCS score: S = alpha[L1, L2]
 */
__global__ void lcs_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ lcs_score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = (size_t)max_L2 + 1;
    size_t total_stride = ((size_t)max_L1 + 1) * stride;

    size_t final_idx = (size_t)L1 * stride + L2;
    lcs_score[b] = alpha[b * total_stride + final_idx];
}

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

/**
 * Initialize beta: beta[L1, L2] = 1, all others = 0
 */
__global__ void lcs_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / (max_L2 + 1);
    int j   = rem % (max_L2 + 1);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Terminal condition: beta[L1, L2] = 1
    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

/**
 * Backward DP: propagate beta and compute posteriors + temperature gradient
 */
__global__ void lcs_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    float* be = beta + (size_t)b * stride_alpha;
    float* post = posteriors + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    float local_T_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        if (beta_ij <= 1e-20f) continue;

        float match_score = s[score_idx];

        // Compute option values
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float v_match = a_diag + match_score;
        float v_skip1 = a_up;
        float v_skip2 = a_left;

        float w_match, w_skip1, w_skip2;
        softmax3_weights(v_match, v_skip1, v_skip2, T, w_match, w_skip1, w_skip2);

        // Posteriors for match: gradient of LCS w.r.t. scores
        atomicAdd(&post[score_idx], beta_ij * w_match);

        // Temperature gradient: dS/dT = sum beta * (softmax - E[v]) / T
        float alpha_ij = a[idx];
        if (alpha_ij > NINF) {
            float E_v = w_match * v_match + w_skip1 * v_skip1 + w_skip2 * v_skip2;
            local_T_grad += beta_ij * (alpha_ij - E_v) / T;
        }

        // Propagate beta to predecessors
        if (w_match > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_match);
        }
        if (w_skip1 > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_skip1);
        }
        if (w_skip2 > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_skip2);
        }
    }

    // Block reduce temperature gradient
    float block_T_grad = block_reduce_sum_lcs(local_T_grad);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_T_grad);
    }
}

// ============================================================================
// HVP (Hessian-Vector Product) KERNELS
// ============================================================================

/**
 * Forward tangent pass: compute d_alpha
 */
__global__ void lcs_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    const float* v = V + (size_t)b * stride_scores;
    float* da = d_alpha + (size_t)b * stride_alpha;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride + j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float match_score = s[score_idx];
        float v_ij = v[score_idx];

        // Get predecessor alpha values
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_match = a_diag + match_score;
        float val_skip1 = a_up;
        float val_skip2 = a_left;

        float w_match, w_skip1, w_skip2;
        softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

        // Get predecessor d_alpha values
        float da_diag = da[idx_diag];
        float da_up   = da[idx_up];
        float da_left = da[idx_left];

        // Tangent of option values
        float dv_match = da_diag + v_ij;  // d(a_diag + match_score) = da_diag + V
        float dv_skip1 = da_up;            // d(a_up + 0) = da_up
        float dv_skip2 = da_left;          // d(a_left + 0) = da_left

        // d(alpha[i,j]) = sum w_k * dv_k
        da[idx] = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;
    }
}

/**
 * Compute d_lcs_score
 */
__global__ void lcs_hvp_score_kernel(
    const float* __restrict__ d_alpha,
    float* __restrict__ d_lcs_score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = (size_t)max_L2 + 1;
    size_t total_stride = ((size_t)max_L1 + 1) * stride;

    size_t final_idx = (size_t)L1 * stride + L2;
    d_lcs_score[b] = d_alpha[b * total_stride + final_idx];
}

/**
 * Backward tangent pass: compute HVP
 */
__global__ void lcs_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    const float* da = d_alpha + (size_t)b * stride_alpha;
    const float* v = V + (size_t)b * stride_scores;
    float* be = beta + (size_t)b * stride_alpha;
    float* dbe = d_beta + (size_t)b * stride_alpha;
    float* H = H_scores + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        float dbeta_ij = dbe[idx];
        float match_score = s[score_idx];
        float v_ij = v[score_idx];

        if (beta_ij <= 1e-20f && (dbeta_ij <= 1e-20f && dbeta_ij >= -1e-20f)) continue;

        // Compute weights
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_match = a_diag + match_score;
        float val_skip1 = a_up;
        float val_skip2 = a_left;

        float w_match, w_skip1, w_skip2;
        softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

        // Compute weight tangents for softmax
        // For softmax: dw_k = w_k * (dv_k - E[dv]) / T
        float da_diag_v = da[idx_diag];
        float da_up_v   = da[idx_up];
        float da_left_v = da[idx_left];

        float dv_match = da_diag_v + v_ij;
        float dv_skip1 = da_up_v;
        float dv_skip2 = da_left_v;

        float E_dv = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;

        float dw_match = w_match * (dv_match - E_dv) / T;
        float dw_skip1 = w_skip1 * (dv_skip1 - E_dv) / T;
        float dw_skip2 = w_skip2 * (dv_skip2 - E_dv) / T;

        // HVP for scores: d(posteriors) = dbeta * w_match + beta * dw_match
        atomicAdd(&H[score_idx], dbeta_ij * w_match + beta_ij * dw_match);

        // Propagate beta and dbeta
        if (w_match > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_match);
            atomicAdd(&dbe[idx_diag], dbeta_ij * w_match + beta_ij * dw_match);
        }
        if (w_skip1 > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_skip1);
            atomicAdd(&dbe[idx_up], dbeta_ij * w_skip1 + beta_ij * dw_skip1);
        }
        if (w_skip2 > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_skip2);
            atomicAdd(&dbe[idx_left], dbeta_ij * w_skip2 + beta_ij * dw_skip2);
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT KERNELS (dP/dT)
// ============================================================================

/**
 * Initialize U for temperature (all zeros for LCS)
 */
__global__ void lcs_init_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    // For LCS, all boundary U values are 0 (base cases don't depend on T)
    U[idx] = 0.0f;
}

/**
 * Forward U pass: compute U = d(alpha)/dT
 */
__global__ void lcs_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    float* u = U + (size_t)b * stride_alpha;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride + j;
            u[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float match_score = s[score_idx];

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_match = a_diag + match_score;
        float val_skip1 = a_up;
        float val_skip2 = a_left;

        float w_match, w_skip1, w_skip2;
        softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

        float u_diag = u[idx_diag];
        float u_up   = u[idx_up];
        float u_left = u[idx_left];

        // U = d(alpha)/dT
        // dv_k = U_k (no direct term since scores don't depend on T)
        float du_match = u_diag;
        float du_skip1 = u_up;
        float du_skip2 = u_left;

        // U[i,j] = sum w_k * du_k + direct temperature term
        float U_val = w_match * du_match + w_skip1 * du_skip1 + w_skip2 * du_skip2;

        // Add direct temperature derivative: (softmax - E[v]) / T
        float alpha_ij = a[idx];
        float E_v = w_match * val_match + w_skip1 * val_skip1 + w_skip2 * val_skip2;
        U_val += (alpha_ij - E_v) / T;

        u[idx] = U_val;
    }
}

/**
 * Backward W pass: compute W = d(beta)/dT and accumulate dP/dT
 */
__global__ void lcs_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    const float* u_buf = U + (size_t)b * stride_alpha;
    float* be = beta + (size_t)b * stride_alpha;
    float* w_buf = W + (size_t)b * stride_alpha;
    float* dP = dP_dT + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + j;
        size_t idx_left  = (size_t)i * stride + (j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        float W_ij = w_buf[idx];
        float match_score = s[score_idx];

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_match = a_diag + match_score;
        float val_skip1 = a_up;
        float val_skip2 = a_left;

        float w_match, w_skip1, w_skip2;
        softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

        // Accumulate dP/dT = W * w_match
        atomicAdd(&dP[score_idx], W_ij * w_match);

        if (beta_ij <= 1e-20f) continue;

        // Compute dw/dT using U values
        float u_diag = u_buf[idx_diag];
        float u_up   = u_buf[idx_up];
        float u_left = u_buf[idx_left];

        float dv_match = u_diag;
        float dv_skip1 = u_up;
        float dv_skip2 = u_left;

        float E_dv = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;

        // For softmax: dw_k/d(param) = w_k * (dv_k - E[dv]) / T
        float dw_match = w_match * (dv_match - E_dv) / T;
        float dw_skip1 = w_skip1 * (dv_skip1 - E_dv) / T;
        float dw_skip2 = w_skip2 * (dv_skip2 - E_dv) / T;

        // Add direct temperature derivative: dw_k/dT = w_k * (E[v] - v_k) / T^2
        float E_v = w_match * val_match + w_skip1 * val_skip1 + w_skip2 * val_skip2;
        float inv_T2 = 1.0f / (T * T);
        dw_match += w_match * (E_v - val_match) * inv_T2;
        dw_skip1 += w_skip1 * (E_v - val_skip1) * inv_T2;
        dw_skip2 += w_skip2 * (E_v - val_skip2) * inv_T2;

        // Add beta * dw to dP
        atomicAdd(&dP[score_idx], beta_ij * dw_match);

        // Propagate beta and W
        if (w_match > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_match);
            atomicAdd(&w_buf[idx_diag], W_ij * w_match + beta_ij * dw_match);
        }
        if (w_skip1 > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_skip1);
            atomicAdd(&w_buf[idx_up], W_ij * w_skip1 + beta_ij * dw_skip1);
        }
        if (w_skip2 > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_skip2);
            atomicAdd(&w_buf[idx_left], W_ij * w_skip2 + beta_ij * dw_skip2);
        }
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

void lcs_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_lcs_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;

    // Initialize alpha
    int blocks_init = (total_alpha + threads - 1) / threads;
    lcs_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Wavefront DP
    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        lcs_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Extract score
    int blocks_score = (B + threads - 1) / threads;
    lcs_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_lcs_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lcs_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)d_lcs_score;  // unused but kept for interface consistency

    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    // Zero outputs
    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    // Initialize beta
    int blocks_init = (total_alpha + threads - 1) / threads;
    lcs_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward DP
    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        lcs_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_T,
            d_lengths, B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lcs_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_lcs_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)d_lcs_score;  // unused but kept for interface consistency

    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_lcs_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward tangent pass
    for (int k = 2; k <= max_diag; ++k) {
        lcs_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Compute d_lcs_score
    int blocks_score = (B + threads - 1) / threads;
    lcs_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_d_alpha, d_d_lcs_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Initialize beta for backward
    int blocks_init = (total_alpha + threads - 1) / threads;
    lcs_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward tangent pass
    for (int k = max_diag; k >= 2; --k) {
        lcs_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lcs_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)d_lcs_score;  // unused but kept for interface consistency

    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    // Zero W and output, initialize U
    int blocks_init = (total_alpha + threads - 1) / threads;
    lcs_init_U_kernel<<<blocks_init, threads, 0, stream>>>(d_U, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dT, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward U pass
    for (int k = 2; k <= max_diag; ++k) {
        lcs_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Initialize beta for backward
    lcs_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward W pass
    for (int k = max_diag; k >= 2; --k) {
        lcs_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_beta, d_W, d_dP_dT, d_lengths,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace lcs
}  // namespace d2p
