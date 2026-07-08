// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft Levenshtein (Edit Distance) CUDA Kernel Implementations
 *
 * Implements anti-diagonal wavefront parallelization for edit distance.
 * Uses softmin instead of logsumexp (minimization rather than maximization).
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>

namespace d2p {
namespace lev {

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
__device__ __forceinline__ T warp_reduce_sum_lev(T v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum_lev(T v) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum_lev(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : (T)0.0f;
    if (wid == 0) v = warp_reduce_sum_lev(v);
    return v;
}

// Softmin for 3 values: min - T * log(sum exp(-(x - min)/T))
__device__ __forceinline__ float softmin3(float a, float b, float c, float T) {
    float m = fminf(fminf(a, b), c);
    if (m >= PINF) return PINF;

    float ea = (a < PINF) ? safe_exp(-(a - m) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp(-(b - m) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp(-(c - m) / T) : 0.0f;

    float sum = ea + eb + ec;
    if (sum <= 0.0f) return PINF;
    return m - T * logf(sum);
}

// Softmin weights: w_k = exp(-(a_k - m)/T) / sum exp(-(a_j - m)/T)
__device__ __forceinline__ void softmin3_weights(
    float a, float b, float c, float T,
    float& wa, float& wb, float& wc
) {
    float m = fminf(fminf(a, b), c);
    if (m >= PINF) {
        wa = wb = wc = 0.0f;
        return;
    }

    float ea = (a < PINF) ? safe_exp(-(a - m) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp(-(b - m) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp(-(c - m) / T) : 0.0f;

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
 * Initialize alpha with Levenshtein base cases:
 * alpha(0,0) = 0
 * alpha(i,0) = i * del_cost for i > 0
 * alpha(0,j) = j * ins_cost for j > 0
 * All other cells = +inf (will be filled by DP)
 */
__global__ void lev_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t alpha_cols = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    size_t i = rem / alpha_cols;
    size_t j = rem % alpha_cols;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Cells outside actual lengths get +inf
    if (i > L1 || j > L2) {
        alpha[idx] = PINF;
        return;
    }

    // Base cases for edit distance
    if (i == 0 && j == 0) {
        alpha[idx] = 0.0f;
    } else if (i == 0) {
        // First row: j insertions
        alpha[idx] = static_cast<float>(j) * ins_cost;
    } else if (j == 0) {
        // First column: i deletions
        alpha[idx] = static_cast<float>(i) * del_cost;
    } else {
        // Interior cells initialized to +inf (will be computed by DP)
        alpha[idx] = PINF;
    }
}

/**
 * Forward DP for one anti-diagonal k = i + j
 * Three transitions: substitute (diagonal), delete (up), insert (left)
 */
__global__ void lev_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

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

        // Cells outside actual lengths get +inf
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * alpha_cols + (size_t)j;
            a[idx] = PINF;
            continue;
        }

        size_t idx       = (size_t)i * alpha_cols + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * alpha_cols + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * alpha_cols + (size_t)j;
        size_t idx_left  = (size_t)i * alpha_cols + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float sub_cost = s[score_idx];

        // Get predecessor values
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        // Levenshtein recurrence:
        // v_sub = alpha[i-1,j-1] + scores[i-1,j-1] (substitute)
        // v_del = alpha[i-1,j] + del_cost (delete from seq1)
        // v_ins = alpha[i,j-1] + ins_cost (insert into seq1)
        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;

        a[idx] = softmin3(v_sub, v_del, v_ins, T);
    }
}

/**
 * Extract distance: D = alpha[L1, L2]
 */
__global__ void lev_distance_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ distance,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = (size_t)max_L2 + 1;
    size_t total_stride = ((size_t)max_L1 + 1) * stride;

    size_t final_idx = (size_t)L1 * stride + (size_t)L2;
    distance[b] = alpha[(size_t)b * total_stride + final_idx];
}

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

/**
 * Initialize beta: beta[L1, L2] = 1, all others = 0
 */
__global__ void lev_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t alpha_cols = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    size_t i = rem / alpha_cols;
    size_t j = rem % alpha_cols;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Terminal condition: beta[L1, L2] = 1
    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

/**
 * Backward DP: propagate beta and compute posteriors + parameter gradients
 */
__global__ void lev_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_ins,
    float* __restrict__ grad_del,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

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

    float local_ins_grad = 0.0f;
    float local_del_grad = 0.0f;
    float local_T_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + (size_t)j;
        size_t idx_left  = (size_t)i * stride + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float beta_ij = be[idx];
        if (beta_ij <= 1e-20f) continue;

        float sub_cost = s[score_idx];

        // Compute option values
        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;

        float w_sub, w_del, w_ins;
        softmin3_weights(v_sub, v_del, v_ins, T, w_sub, w_del, w_ins);

        // Posteriors for substitution
        atomicAdd(&post[score_idx], beta_ij * w_sub);

        // Parameter gradients
        local_del_grad += beta_ij * w_del;
        local_ins_grad += beta_ij * w_ins;

        // Temperature gradient
        float alpha_ij = a[idx];
        if (alpha_ij < PINF) {
            float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins;
            local_T_grad += beta_ij * (alpha_ij - E_v) / T;
        }

        // Propagate beta to predecessors
        if (w_sub > 0.0f) atomicAdd(&be[idx_diag], beta_ij * w_sub);
        if (w_del > 0.0f) atomicAdd(&be[idx_up], beta_ij * w_del);
        if (w_ins > 0.0f) atomicAdd(&be[idx_left], beta_ij * w_ins);
    }

    // Block reduce gradients
    float block_ins_grad = block_reduce_sum_lev(local_ins_grad);
    float block_del_grad = block_reduce_sum_lev(local_del_grad);
    float block_T_grad = block_reduce_sum_lev(local_T_grad);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_ins[b], block_ins_grad);
        atomicAdd(&grad_del[b], block_del_grad);
        atomicAdd(&grad_T[b], block_T_grad);
    }
}

// ============================================================================
// HVP KERNELS
// ============================================================================

/**
 * Forward tangent pass: compute d_alpha
 */
__global__ void lev_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

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

        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride + (size_t)j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + (size_t)j;
        size_t idx_left  = (size_t)i * stride + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float sub_cost = s[score_idx];
        float v_ij = v[score_idx];

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;

        float w_sub, w_del, w_ins;
        softmin3_weights(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

        // Tangent of option values
        float dv_sub = da[idx_diag] + v_ij;
        float dv_del = da[idx_up];
        float dv_ins = da[idx_left];

        da[idx] = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;
    }
}

/**
 * Compute d_distance
 */
__global__ void lev_hvp_distance_kernel(
    const float* __restrict__ d_alpha,
    float* __restrict__ d_distance,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = (size_t)max_L2 + 1;
    size_t total_stride = ((size_t)max_L1 + 1) * stride;

    size_t final_idx = (size_t)L1 * stride + (size_t)L2;
    d_distance[b] = d_alpha[(size_t)b * total_stride + final_idx];
}

/**
 * Backward tangent pass: compute HVP
 */
__global__ void lev_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

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

        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + (size_t)j;
        size_t idx_left  = (size_t)i * stride + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float beta_ij = be[idx];
        float dbeta_ij = dbe[idx];
        float sub_cost = s[score_idx];
        float v_ij = v[score_idx];

        if (beta_ij <= 1e-20f) {
            if (dbeta_ij > 1e-20f || dbeta_ij < -1e-20f) {
                float a_diag = a[idx_diag];
                float a_up   = a[idx_up];
                float a_left = a[idx_left];

                float val_sub = a_diag + sub_cost;
                float val_del = a_up + del_cost;
                float val_ins = a_left + ins_cost;

                float w_sub, w_del, w_ins;
                softmin3_weights(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

                atomicAdd(&H[score_idx], dbeta_ij * w_sub);

                if (w_sub > 0.0f) atomicAdd(&dbe[idx_diag], dbeta_ij * w_sub);
                if (w_del > 0.0f) atomicAdd(&dbe[idx_up], dbeta_ij * w_del);
                if (w_ins > 0.0f) atomicAdd(&dbe[idx_left], dbeta_ij * w_ins);
            }
            continue;
        }

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;

        float w_sub, w_del, w_ins;
        softmin3_weights(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

        // Weight tangents
        float da_diag_v = da[idx_diag];
        float da_up_v   = da[idx_up];
        float da_left_v = da[idx_left];

        float dv_sub = da_diag_v + v_ij;
        float dv_del = da_up_v;
        float dv_ins = da_left_v;

        float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;

        float dw_sub = w_sub * (-dv_sub + E_dv) / T;
        float dw_del = w_del * (-dv_del + E_dv) / T;
        float dw_ins = w_ins * (-dv_ins + E_dv) / T;

        atomicAdd(&H[score_idx], dbeta_ij * w_sub + beta_ij * dw_sub);

        if (w_sub > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_sub);
            atomicAdd(&dbe[idx_diag], dbeta_ij * w_sub + beta_ij * dw_sub);
        }
        if (w_del > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_del);
            atomicAdd(&dbe[idx_up], dbeta_ij * w_del + beta_ij * dw_del);
        }
        if (w_ins > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_ins);
            atomicAdd(&dbe[idx_left], dbeta_ij * w_ins + beta_ij * dw_ins);
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT KERNELS
// ============================================================================

/**
 * Initialize U with boundary derivatives
 */
__global__ void lev_init_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    int param_type
) {
    size_t stride = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t alpha_cols = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    size_t i = rem / alpha_cols;
    size_t j = rem % alpha_cols;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float val = 0.0f;

    if (i <= L1 && j <= L2) {
        if (param_type == LEV_PARAM_DEL && j == 0 && i > 0) {
            val = (float)i;
        } else if (param_type == LEV_PARAM_INS && i == 0 && j > 0) {
            val = (float)j;
        }
    }

    U[idx] = val;
}

/**
 * Forward U pass: compute U = d(alpha)/d(param)
 */
__global__ void lev_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int param_type,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

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

        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride + (size_t)j;
            u[idx] = 0.0f;
            continue;
        }

        size_t idx       = (size_t)i * stride + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + (size_t)j;
        size_t idx_left  = (size_t)i * stride + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float sub_cost = s[score_idx];

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;

        float w_sub, w_del, w_ins;
        softmin3_weights(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

        float u_diag = u[idx_diag];
        float u_up   = u[idx_up];
        float u_left = u[idx_left];

        float du_sub = u_diag;
        float du_del = u_up;
        float du_ins = u_left;

        if (param_type == LEV_PARAM_INS) {
            du_ins += 1.0f;
        } else if (param_type == LEV_PARAM_DEL) {
            du_del += 1.0f;
        }

        float U_val = w_sub * du_sub + w_del * du_del + w_ins * du_ins;

        if (param_type == LEV_PARAM_TEMPERATURE) {
            float alpha_ij = a[idx];
            float E_v = w_sub * val_sub + w_del * val_del + w_ins * val_ins;
            U_val += (alpha_ij - E_v) / T;
        }

        u[idx] = U_val;
    }
}

/**
 * Backward W pass: compute W = d(beta)/d(param) and accumulate dP/d(param)
 */
__global__ void lev_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dparam,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int param_type,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    const float* u_buf = U + (size_t)b * stride_alpha;
    float* be = beta + (size_t)b * stride_alpha;
    float* w_buf = W + (size_t)b * stride_alpha;
    float* dP = dP_dparam + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) continue;

        size_t idx       = (size_t)i * stride + (size_t)j;
        size_t idx_diag  = (size_t)(i - 1) * stride + (size_t)(j - 1);
        size_t idx_up    = (size_t)(i - 1) * stride + (size_t)j;
        size_t idx_left  = (size_t)i * stride + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * max_L2 + (size_t)(j - 1);

        float beta_ij = be[idx];
        float W_ij = w_buf[idx];
        float sub_cost = s[score_idx];

        float a_diag = a[idx_diag];
        float a_up   = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;

        float w_sub, w_del, w_ins;
        softmin3_weights(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

        atomicAdd(&dP[score_idx], W_ij * w_sub);

        if (beta_ij <= 1e-20f) continue;

        float u_diag = u_buf[idx_diag];
        float u_up   = u_buf[idx_up];
        float u_left = u_buf[idx_left];

        float dv_sub = u_diag;
        float dv_del = u_up;
        float dv_ins = u_left;

        if (param_type == LEV_PARAM_INS) {
            dv_ins += 1.0f;
        } else if (param_type == LEV_PARAM_DEL) {
            dv_del += 1.0f;
        }

        float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;

        float dw_sub = w_sub * (-dv_sub + E_dv) / T;
        float dw_del = w_del * (-dv_del + E_dv) / T;
        float dw_ins = w_ins * (-dv_ins + E_dv) / T;

        if (param_type == LEV_PARAM_TEMPERATURE) {
            float E_v = w_sub * val_sub + w_del * val_del + w_ins * val_ins;
            float inv_T2 = 1.0f / (T * T);
            dw_sub += w_sub * (val_sub - E_v) * inv_T2;
            dw_del += w_del * (val_del - E_v) * inv_T2;
            dw_ins += w_ins * (val_ins - E_v) * inv_T2;
        }

        atomicAdd(&dP[score_idx], beta_ij * dw_sub);

        if (w_sub > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_sub);
            atomicAdd(&w_buf[idx_diag], W_ij * w_sub + beta_ij * dw_sub);
        }
        if (w_del > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_del);
            atomicAdd(&w_buf[idx_up], W_ij * w_del + beta_ij * dw_del);
        }
        if (w_ins > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_ins);
            atomicAdd(&w_buf[idx_left], W_ij * w_ins + beta_ij * dw_ins);
        }
    }
}

// ============================================================================
// Boundary Gradient Accumulation
// ============================================================================

/**
 * Accumulate boundary insertion/deletion cost gradients.
 *
 * Boundary cells carry direct cost dependence: alpha[i,0] = i * del_cost,
 * alpha[0,j] = j * ins_cost. The backward diag kernel only processes cells
 * with i >= 1, j >= 1, so the beta mass that propagates to boundary cells
 * (column 0 and row 0) is never converted into cost gradients. This kernel
 * adds those missing contributions after the backward diag loop completes.
 */
__global__ void lev_boundary_grad_kernel(
    const float* __restrict__ beta,
    float* __restrict__ grad_ins,
    float* __restrict__ grad_del,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x;
    if (b >= B) return;

    size_t stride = (size_t)max_L2 + 1;
    size_t stride_alpha = ((size_t)max_L1 + 1) * stride;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* be = beta + (size_t)b * stride_alpha;

    float del_acc = 0.0f;
    float ins_acc = 0.0f;

    // Column 0: beta[i, 0] * i contributes to grad_del
    for (int i = 1 + (int)threadIdx.x; i <= L1; i += (int)blockDim.x) {
        del_acc += be[(size_t)i * stride] * static_cast<float>(i);
    }
    // Row 0: beta[0, j] * j contributes to grad_ins
    for (int j = 1 + (int)threadIdx.x; j <= L2; j += (int)blockDim.x) {
        ins_acc += be[j] * static_cast<float>(j);
    }

    // Block reduce
    del_acc = block_reduce_sum_lev(del_acc);
    ins_acc = block_reduce_sum_lev(ins_acc);

    if (threadIdx.x == 0) {
        atomicAdd(&grad_del[b], del_acc);
        atomicAdd(&grad_ins[b], ins_acc);
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void lev_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_distance,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;

    int blocks_init = (total_alpha + threads - 1) / threads;
    lev_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_lengths, B, max_L1, max_L2, ins_cost, del_cost
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        lev_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2,
            ins_cost, del_cost, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_dist = (B + threads - 1) / threads;
    lev_distance_kernel<<<blocks_dist, threads, 0, stream>>>(
        d_alpha, d_distance, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lev_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_ins,
    float* d_grad_del,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_ins, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_del, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    int blocks_init = (total_alpha + threads - 1) / threads;
    lev_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        lev_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_ins, d_grad_del, d_grad_T,
            d_lengths, B, max_L1, max_L2, ins_cost, del_cost, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    lev_boundary_grad_kernel<<<B, threads, 0, stream>>>(
        d_beta, d_grad_ins, d_grad_del, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lev_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_distance,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_distance, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    for (int k = 2; k <= max_diag; ++k) {
        lev_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, ins_cost, del_cost, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_dist = (B + threads - 1) / threads;
    lev_hvp_distance_kernel<<<blocks_dist, threads, 0, stream>>>(
        d_d_alpha, d_d_distance, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_init = (total_alpha + threads - 1) / threads;
    lev_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        lev_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_L1, max_L2, ins_cost, del_cost, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void lev_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dparam,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int param_type
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dparam, 0, sizeof(float) * score_elems, stream));

    int blocks_init = (total_alpha + threads - 1) / threads;
    lev_init_U_kernel<<<blocks_init, threads, 0, stream>>>(d_U, d_lengths, B, max_L1, max_L2, param_type);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;

    for (int k = 2; k <= max_diag; ++k) {
        lev_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths,
            B, max_L1, max_L2, ins_cost, del_cost, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    lev_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        lev_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_beta, d_W, d_dP_dparam, d_lengths,
            B, max_L1, max_L2, ins_cost, del_cost, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace lev
}  // namespace d2p
