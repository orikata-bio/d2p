// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft OSA CUDA Kernel Implementations
 *
 * Optimal String Alignment (Restricted Damerau-Levenshtein) using SOFTMIN.
 * Four operations: substitute, insert, delete, transpose (adjacent only).
 *
 * Input:
 *   - sub_costs[B, L1, L2]: Substitution cost at each position
 *   - trans_mask[B, L1, L2]: Boolean mask for valid transpositions
 *
 * Parameters: ins_cost, del_cost, trans_cost, temperature
 */

#include <cuda_runtime.h>
#include <math.h>
#include <limits>
#include "kernels.cuh"
#include "common/cuda_utils.h"

namespace d2p {
namespace osa {

// ============================================================================
// Device Helper Functions
// ============================================================================

#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < (T)-88.0f) return (T)0.0f;
    if (x > (T)88.0f) x = (T)88.0f;
    return exp(x);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum_osa(T v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum_osa(T v) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum_osa(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : (T)0.0f;
    if (wid == 0) v = warp_reduce_sum_osa(v);
    return v;
}

// Softmin for 4 values: -T * log(sum exp(-x/T))
__device__ __forceinline__ float softmin4(float a, float b, float c, float d, float T) {
    float m = fminf(fminf(a, b), fminf(c, d));
    if (m >= PINF) return PINF;

    float ea = (a < PINF) ? safe_exp(-(a - m) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp(-(b - m) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp(-(c - m) / T) : 0.0f;
    float ed = (d < PINF) ? safe_exp(-(d - m) / T) : 0.0f;

    float sum = ea + eb + ec + ed;
    if (sum <= 0.0f) return PINF;
    return m - T * logf(sum);
}

// Softmin weights for 4 values
__device__ __forceinline__ void softmin4_weights(
    float a, float b, float c, float d, float T,
    float& wa, float& wb, float& wc, float& wd
) {
    float m = fminf(fminf(a, b), fminf(c, d));
    if (m >= PINF) {
        wa = wb = wc = wd = 0.0f;
        return;
    }

    float ea = (a < PINF) ? safe_exp(-(a - m) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp(-(b - m) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp(-(c - m) / T) : 0.0f;
    float ed = (d < PINF) ? safe_exp(-(d - m) / T) : 0.0f;

    float total = ea + eb + ec + ed;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
        wc = ec / total;
        wd = ed / total;
    } else {
        wa = wb = wc = wd = 0.0f;
    }
}

// ============================================================================
// FORWARD PASS KERNELS
// ============================================================================

__global__ void osa_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    float del_cost,
    float ins_cost,
    int B, int max_L1, int max_L2
) {
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = static_cast<int>(rem / alpha_cols);
    int j   = static_cast<int>(rem % alpha_cols);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i > L1 || j > L2) {
        alpha[idx] = PINF;
        return;
    }

    if (i == 0 && j == 0) {
        alpha[idx] = 0.0f;
    } else if (i == 0) {
        alpha[idx] = j * ins_cost;
    } else if (j == 0) {
        alpha[idx] = i * del_cost;
    } else {
        alpha[idx] = PINF;
    }
}

__global__ void osa_forward_diag_kernel(
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * alpha_cols + j;
            a[idx] = PINF;
            continue;
        }

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float sub_cost = s[score_idx];

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;

        float v_trans = PINF;
        if (i >= 2 && j >= 2) {
            float trans_valid = tm[score_idx];
            if (trans_valid > 0.5f) {
                size_t idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
                v_trans = a[idx_trans] + trans_cost;
            }
        }

        a[idx] = softmin4(v_sub, v_del, v_ins, v_trans, T);
    }
}

__global__ void osa_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ osa_score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t total_stride = (static_cast<size_t>(max_L1) + 1) * stride;

    size_t final_idx = static_cast<size_t>(L1) * stride + L2;
    osa_score[b] = alpha[b * total_stride + final_idx];
}

// ============================================================================
// BACKWARD PASS KERNELS
// ============================================================================

__global__ void osa_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = static_cast<int>(rem / alpha_cols);
    int j   = static_cast<int>(rem % alpha_cols);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

__global__ void osa_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_T,
    float* __restrict__ grad_ins,
    float* __restrict__ grad_del,
    float* __restrict__ grad_trans,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;
    float* be = beta + (size_t)b * stride_alpha;
    float* post = posteriors + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    float local_T_grad = 0.0f;
    float local_ins_grad = 0.0f;
    float local_del_grad = 0.0f;
    float local_trans_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;
        if (i > L1 || j > L2) continue;

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        if (beta_ij <= 1e-20f) continue;

        float sub_cost = s[score_idx];
        float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        if (trans_valid > 0.5f) {
            size_t idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            v_trans = a[idx_trans] + trans_cost;
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

        // Posteriors for substitution
        atomicAdd(&post[score_idx], beta_ij * w_sub);

        // Temperature gradient
        float alpha_ij = a[idx];
        if (alpha_ij < PINF) {
            float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
            local_T_grad += beta_ij * (alpha_ij - E_v) / T;
        }

        // Cost parameter gradients
        local_ins_grad += beta_ij * w_ins;
        local_del_grad += beta_ij * w_del;
        local_trans_grad += beta_ij * w_trans;

        // Propagate beta
        if (w_sub > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_sub);
        }
        if (w_del > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_del);
        }
        if (w_ins > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_ins);
        }
        if (w_trans > 0.0f && trans_valid > 0.5f) {
            size_t idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            atomicAdd(&be[idx_trans], beta_ij * w_trans);
        }
    }

    // Block reduce parameter gradients
    float block_T_grad = block_reduce_sum_osa(local_T_grad);
    float block_ins_grad = block_reduce_sum_osa(local_ins_grad);
    float block_del_grad = block_reduce_sum_osa(local_del_grad);
    float block_trans_grad = block_reduce_sum_osa(local_trans_grad);

    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_T_grad);
        atomicAdd(&grad_ins[b], block_ins_grad);
        atomicAdd(&grad_del[b], block_del_grad);
        atomicAdd(&grad_trans[b], block_trans_grad);
    }
}

// ============================================================================
// HVP KERNELS
// ============================================================================

__global__ void osa_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;
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
            size_t idx = static_cast<size_t>(i) * alpha_cols + j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float sub_cost = s[score_idx];
        float v_ij = v[score_idx];
        float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;
        float val_trans = PINF;
        float da_trans = 0.0f;
        if (trans_valid > 0.5f) {
            size_t idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            val_trans = a[idx_trans] + trans_cost;
            da_trans = da[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

        float dv_sub = da[idx_diag] + v_ij;
        float dv_del = da[idx_up];
        float dv_ins = da[idx_left];
        float dv_trans = (trans_valid > 0.5f) ? da_trans : 0.0f;

        da[idx] = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins + w_trans * dv_trans;
    }
}

__global__ void osa_hvp_score_kernel(
    const float* __restrict__ d_alpha,
    float* __restrict__ d_score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t total_stride = (static_cast<size_t>(max_L1) + 1) * stride;

    size_t final_idx = static_cast<size_t>(L1) * stride + L2;
    d_score[b] = d_alpha[b * total_stride + final_idx];
}

__global__ void osa_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;
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

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        float dbeta_ij = dbe[idx];
        float sub_cost = s[score_idx];
        float v_ij = v[score_idx];
        float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

        if (beta_ij <= 1e-20f && fabsf(dbeta_ij) < 1e-20f) continue;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;
        float val_trans = PINF;
        float da_trans = 0.0f;
        size_t idx_trans = 0;
        bool has_trans = false;
        if (trans_valid > 0.5f) {
            idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            has_trans = true;
            val_trans = a[idx_trans] + trans_cost;
            da_trans = da[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

        float dv_sub = da[idx_diag] + v_ij;
        float dv_del = da[idx_up];
        float dv_ins = da[idx_left];
        float dv_trans = (trans_valid > 0.5f) ? da_trans : 0.0f;

        float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins + w_trans * dv_trans;

        // Weight tangents for softmin: dw_k = -w_k * (dv_k - E[dv]) / T
        float dw_sub = -w_sub * (dv_sub - E_dv) / T;
        float dw_del = -w_del * (dv_del - E_dv) / T;
        float dw_ins = -w_ins * (dv_ins - E_dv) / T;
        float dw_trans = -w_trans * (dv_trans - E_dv) / T;

        // HVP for substitution costs
        atomicAdd(&H[score_idx], dbeta_ij * w_sub + beta_ij * dw_sub);

        // Propagate beta and dbeta
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
        if (w_trans > 0.0f && has_trans) {
            atomicAdd(&be[idx_trans], beta_ij * w_trans);
            atomicAdd(&dbe[idx_trans], dbeta_ij * w_trans + beta_ij * dw_trans);
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
__global__ void osa_boundary_grad_kernel(
    const float* __restrict__ beta,
    float* __restrict__ grad_ins,
    float* __restrict__ grad_del,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x;
    if (b >= B) return;

    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* be = beta + (size_t)b * stride_alpha;

    float del_acc = 0.0f;
    float ins_acc = 0.0f;

    // Column 0: beta[i, 0] * i contributes to grad_del.
    for (int i = 1 + (int)threadIdx.x; i <= L1; i += (int)blockDim.x) {
        del_acc += be[static_cast<size_t>(i) * alpha_cols] * static_cast<float>(i);
    }
    // Row 0: beta[0, j] * j contributes to grad_ins.
    for (int j = 1 + (int)threadIdx.x; j <= L2; j += (int)blockDim.x) {
        ins_acc += be[j] * static_cast<float>(j);
    }

    del_acc = block_reduce_sum_osa(del_acc);
    ins_acc = block_reduce_sum_osa(ins_acc);

    if (threadIdx.x == 0) {
        atomicAdd(&grad_del[b], del_acc);
        atomicAdd(&grad_ins[b], ins_acc);
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

namespace {

size_t osa_alpha_elems(int max_L1, int max_L2) {
    return (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
}

int osa_blocks_for(size_t elements, int threads, const char* kernel_name) {
    size_t blocks = (elements + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
    TORCH_CHECK(
        blocks <= static_cast<size_t>(std::numeric_limits<int>::max()),
        kernel_name,
        ": launch grid is too large"
    );
    return static_cast<int>(blocks);
}

}  // namespace

void osa_forward(
    const float* d_sub_costs,
    const float* d_trans_mask,
    float* d_alpha,
    float* d_osa_score,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = osa_alpha_elems(max_L1, max_L2);
    size_t total_alpha = (size_t)B * alpha_elems;

    int blocks_init = osa_blocks_for(total_alpha, threads, "osa_init_alpha_kernel");
    osa_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_lengths, del_cost, ins_cost, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        osa_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_sub_costs, d_trans_mask, d_alpha, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = osa_blocks_for(static_cast<size_t>(B), threads, "osa_score_kernel");
    osa_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_osa_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void osa_backward(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    float* d_grad_ins,
    float* d_grad_del,
    float* d_grad_trans,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = osa_alpha_elems(max_L1, max_L2);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_ins, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_del, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_trans, 0, sizeof(float) * B, stream));

    int blocks_init = osa_blocks_for(total_alpha, threads, "osa_init_beta_kernel");
    osa_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        osa_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_mask, d_beta, d_posteriors,
            d_grad_T, d_grad_ins, d_grad_del, d_grad_trans,
            d_lengths, ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    osa_boundary_grad_kernel<<<B, threads, 0, stream>>>(
        d_beta, d_grad_ins, d_grad_del, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void osa_hvp(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = osa_alpha_elems(max_L1, max_L2);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward tangent pass
    for (int k = 2; k <= max_diag; ++k) {
        osa_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_mask, d_V, d_d_alpha, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = osa_blocks_for(static_cast<size_t>(B), threads, "osa_hvp_score_kernel");
    osa_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_d_alpha, d_d_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_init = osa_blocks_for(total_alpha, threads, "osa_init_beta_kernel");
    osa_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward tangent pass
    for (int k = max_diag; k >= 2; --k) {
        osa_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_mask, d_V, d_d_alpha,
            d_beta, d_d_beta, d_H_scores, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================================
// PARAMETER GRADIENT KERNELS
// ============================================================================

__global__ void osa_init_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    int param_type
) {
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = static_cast<int>(rem / alpha_cols);
    int j   = static_cast<int>(rem % alpha_cols);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float val = 0.0f;
    if (i <= L1 && j <= L2) {
        if (param_type == OSA_PARAM_DEL && j == 0 && i > 0) {
            val = (float)i;
        } else if (param_type == OSA_PARAM_INS && i == 0 && j > 0) {
            val = (float)j;
        }
    }
    U[idx] = val;
}

__global__ void osa_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int param_type,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;
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
            u[static_cast<size_t>(i) * alpha_cols + j] = 0.0f;
            continue;
        }

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float sub_cost = s[score_idx];
        float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        float u_trans_val = 0.0f;
        if (trans_valid > 0.5f) {
            size_t idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            v_trans = a[idx_trans] + trans_cost;
            u_trans_val = u[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

        float du_sub = u[idx_diag];
        float du_del = u[idx_up];
        float du_ins = u[idx_left];
        float du_trans = u_trans_val;

        if (param_type == OSA_PARAM_INS) du_ins += 1.0f;
        else if (param_type == OSA_PARAM_DEL) du_del += 1.0f;
        else if (param_type == OSA_PARAM_TRANS && trans_valid > 0.5f) du_trans += 1.0f;

        float U_val = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

        if (param_type == OSA_PARAM_TEMPERATURE) {
            float alpha_ij = a[idx];
            if (alpha_ij < PINF) {
                float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
                U_val += (alpha_ij - E_v) / T;
            }
        }

        u[idx] = U_val;
    }
}

__global__ void osa_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const float* __restrict__ trans_mask,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dparam,
    const int* __restrict__ lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T,
    int param_type,
    int k_diag
) {
    int b = blockIdx.x;
    size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const float* tm = trans_mask + (size_t)b * stride_scores;
    const float* u = U + (size_t)b * stride_alpha;
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

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float beta_ij = be[idx];
        float W_ij = w_buf[idx];
        float sub_cost = s[score_idx];
        float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

        if (beta_ij <= 1e-20f && fabsf(W_ij) < 1e-20f) continue;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        float u_trans_val = 0.0f;
        size_t idx_trans = 0;
        bool has_trans = false;
        if (trans_valid > 0.5f) {
            idx_trans = static_cast<size_t>(i - 2) * alpha_cols + (j - 2);
            has_trans = true;
            v_trans = a[idx_trans] + trans_cost;
            u_trans_val = u[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

        atomicAdd(&dP[score_idx], W_ij * w_sub);

        float du_sub = u[idx_diag];
        float du_del = u[idx_up];
        float du_ins = u[idx_left];
        float du_trans = u_trans_val;

        if (param_type == OSA_PARAM_INS) du_ins += 1.0f;
        else if (param_type == OSA_PARAM_DEL) du_del += 1.0f;
        else if (param_type == OSA_PARAM_TRANS && trans_valid > 0.5f) du_trans += 1.0f;

        float E_dv = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

        float dw_sub = w_sub * (-du_sub + E_dv) / T;
        float dw_del = w_del * (-du_del + E_dv) / T;
        float dw_ins = w_ins * (-du_ins + E_dv) / T;
        float dw_trans = w_trans * (-du_trans + E_dv) / T;

        if (param_type == OSA_PARAM_TEMPERATURE) {
            float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
            float inv_T2 = 1.0f / (T * T);
            dw_sub += w_sub * (v_sub - E_v) * inv_T2;
            dw_del += w_del * (v_del - E_v) * inv_T2;
            dw_ins += w_ins * (v_ins - E_v) * inv_T2;
            dw_trans += w_trans * (v_trans - E_v) * inv_T2;
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
        if (w_trans > 0.0f && has_trans) {
            atomicAdd(&be[idx_trans], beta_ij * w_trans);
            atomicAdd(&w_buf[idx_trans], W_ij * w_trans + beta_ij * dw_trans);
        }
    }
}

void osa_param_grad(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dparam,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float trans_cost, float T,
    int param_type
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = osa_alpha_elems(max_L1, max_L2);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    int blocks_init = osa_blocks_for(total_alpha, threads, "osa_init_U_kernel");
    osa_init_U_kernel<<<blocks_init, threads, 0, stream>>>(
        d_U, d_lengths, B, max_L1, max_L2, param_type
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dparam, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));

    int max_diag = max_L1 + max_L2;

    for (int k = 2; k <= max_diag; ++k) {
        osa_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_mask, d_U, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    osa_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        osa_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_mask, d_U,
            d_beta, d_W, d_dP_dparam, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace osa
}  // namespace d2p
