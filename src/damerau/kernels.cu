// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft True Damerau-Levenshtein CUDA Kernels
 *
 * GPU implementation of differentiable Damerau-Levenshtein edit distance.
 *
 * Key differences from OSA:
 *   - OSA: transposition only considers adjacent characters (alpha[i-2,j-2])
 *   - Damerau: transposition can span variable distances based on character positions
 *   - Uses precomputed trans_src tensor specifying source indices for each transposition
 *
 * Transposition cost at (i,j) from (k,l):
 *   D[k,l] + trans_cost + (i-k-1)*del_cost + (j-l-1)*ins_cost
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <limits>
#include <math.h>

namespace d2p {
namespace damerau {

// ============================================================================
// Device Helpers
// ============================================================================

#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < (T)-88.0f) return (T)0.0f;
    if (x > (T)88.0f) x = (T)88.0f;
    return exp(x);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum(T v) {
    __shared__ T shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : (T)0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
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
// Forward Kernels
// ============================================================================

__global__ void damerau_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    float del_cost,
    float ins_cost,
    int B, int max_L1, int max_L2
) {
    size_t stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / (static_cast<size_t>(max_L2) + 1);
    int j   = rem % (static_cast<size_t>(max_L2) + 1);

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

__global__ void damerau_forward_diag_kernel(
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;

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

        // Transposition: look up source position from trans_src
        float v_trans = PINF;
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];

        if (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j) {
            size_t idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            float a_trans = a[idx_trans];
            v_trans = a_trans + trans_cost +
                      (i - trans_k - 1) * del_cost +
                      (j - trans_l - 1) * ins_cost;
        }

        a[idx] = softmin4(v_sub, v_del, v_ins, v_trans, T);
    }
}

__global__ void damerau_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ damerau_score,
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
    damerau_score[b] = alpha[static_cast<size_t>(b) * total_stride + final_idx];
}

// ============================================================================
// Backward Kernels
// ============================================================================

__global__ void damerau_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / (static_cast<size_t>(max_L2) + 1);
    int j   = rem % (static_cast<size_t>(max_L2) + 1);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

__global__ void damerau_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;
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
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];
        bool trans_valid = (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j);

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        size_t idx_trans = 0;
        int extra_del = 0;
        int extra_ins = 0;

        if (trans_valid) {
            idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            extra_del = i - trans_k - 1;
            extra_ins = j - trans_l - 1;
            v_trans = a[idx_trans] + trans_cost + extra_del * del_cost + extra_ins * ins_cost;
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
        local_ins_grad += beta_ij * (w_ins + w_trans * extra_ins);
        local_del_grad += beta_ij * (w_del + w_trans * extra_del);
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
        if (w_trans > 0.0f && trans_valid) {
            atomicAdd(&be[idx_trans], beta_ij * w_trans);
        }
    }

    // Block reduce parameter gradients
    float block_T_grad = block_reduce_sum(local_T_grad);
    float block_ins_grad = block_reduce_sum(local_ins_grad);
    float block_del_grad = block_reduce_sum(local_del_grad);
    float block_trans_grad = block_reduce_sum(local_trans_grad);

    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_T_grad);
        atomicAdd(&grad_ins[b], block_ins_grad);
        atomicAdd(&grad_del[b], block_del_grad);
        atomicAdd(&grad_trans[b], block_trans_grad);
    }
}

// ============================================================================
// HVP Kernels
// ============================================================================

__global__ void damerau_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;
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
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];
        bool trans_valid = (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j);

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float val_sub = a_diag + sub_cost;
        float val_del = a_up + del_cost;
        float val_ins = a_left + ins_cost;
        float val_trans = PINF;
        float da_trans = 0.0f;

        if (trans_valid) {
            size_t idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            float a_trans = a[idx_trans];
            da_trans = da[idx_trans];
            val_trans = a_trans + trans_cost +
                        (i - trans_k - 1) * del_cost +
                        (j - trans_l - 1) * ins_cost;
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

        // Tangent values
        float dv_sub = da[idx_diag] + v_ij;
        float dv_del = da[idx_up];
        float dv_ins = da[idx_left];
        float dv_trans = trans_valid ? da_trans : 0.0f;

        da[idx] = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins + w_trans * dv_trans;
    }
}

__global__ void damerau_hvp_score_kernel(
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
    d_score[b] = d_alpha[static_cast<size_t>(b) * total_stride + final_idx];
}

__global__ void damerau_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;
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
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];
        bool trans_valid = (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j);

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

        if (trans_valid) {
            idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            float a_trans = a[idx_trans];
            da_trans = da[idx_trans];
            val_trans = a_trans + trans_cost +
                        (i - trans_k - 1) * del_cost +
                        (j - trans_l - 1) * ins_cost;
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

        // Tangent values
        float dv_sub = da[idx_diag] + v_ij;
        float dv_del = da[idx_up];
        float dv_ins = da[idx_left];
        float dv_trans = trans_valid ? da_trans : 0.0f;

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
        if (w_trans > 0.0f && trans_valid) {
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
__global__ void damerau_boundary_grad_kernel(
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

    // Column 0: beta[i, 0] * i contributes to grad_del
    for (int i = 1 + (int)threadIdx.x; i <= L1; i += (int)blockDim.x) {
        del_acc += be[static_cast<size_t>(i) * alpha_cols] * static_cast<float>(i);
    }
    // Row 0: beta[0, j] * j contributes to grad_ins
    for (int j = 1 + (int)threadIdx.x; j <= L2; j += (int)blockDim.x) {
        ins_acc += be[j] * static_cast<float>(j);
    }

    // Block reduce
    del_acc = block_reduce_sum(del_acc);
    ins_acc = block_reduce_sum(ins_acc);

    if (threadIdx.x == 0) {
        atomicAdd(&grad_del[b], del_acc);
        atomicAdd(&grad_ins[b], ins_acc);
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

int checked_grid_blocks(size_t total, int threads, const char* name) {
    size_t blocks = (total + static_cast<size_t>(threads) - 1) / threads;
    TORCH_CHECK(
        blocks <= static_cast<size_t>(std::numeric_limits<int>::max()),
        name, ": launch grid is too large"
    );
    return static_cast<int>(blocks);
}

void damerau_forward(
    const float* d_sub_costs,
    const int* d_trans_src,
    float* d_alpha,
    float* d_damerau_score,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = static_cast<size_t>(B) * alpha_elems;

    int blocks_init = checked_grid_blocks(total_alpha, threads, "damerau_init_alpha_kernel");
    damerau_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_lengths, del_cost, ins_cost, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        damerau_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_sub_costs, d_trans_src, d_alpha, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    damerau_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_damerau_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void damerau_backward(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
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
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = static_cast<size_t>(B) * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_ins, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_del, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_trans, 0, sizeof(float) * B, stream));

    int blocks_init = checked_grid_blocks(total_alpha, threads, "damerau_init_beta_kernel");
    damerau_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        damerau_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_src, d_beta, d_posteriors,
            d_grad_T, d_grad_ins, d_grad_del, d_grad_trans,
            d_lengths, ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    damerau_boundary_grad_kernel<<<B, threads, 0, stream>>>(
        d_beta, d_grad_ins, d_grad_del, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void damerau_hvp(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
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
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = static_cast<size_t>(B) * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward tangent pass
    for (int k = 2; k <= max_diag; ++k) {
        damerau_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_src, d_V, d_d_alpha, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    damerau_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_d_alpha, d_d_score, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_init = checked_grid_blocks(total_alpha, threads, "damerau_init_beta_kernel");
    damerau_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward tangent pass
    for (int k = max_diag; k >= 2; --k) {
        damerau_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_src, d_V, d_d_alpha,
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

__global__ void damerau_init_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    int param_type
) {
    size_t stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(B) * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / (static_cast<size_t>(max_L2) + 1);
    int j   = rem % (static_cast<size_t>(max_L2) + 1);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float val = 0.0f;
    if (i <= L1 && j <= L2) {
        if (param_type == DAMERAU_PARAM_DEL && j == 0 && i > 0) {
            val = (float)i;
        } else if (param_type == DAMERAU_PARAM_INS && i == 0 && j > 0) {
            val = (float)j;
        }
    }
    U[idx] = val;
}

__global__ void damerau_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;
    float* u = U + (size_t)b * stride_alpha;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;
        if (i > L1 || j > L2) { u[static_cast<size_t>(i) * alpha_cols + j] = 0.0f; continue; }

        size_t idx = static_cast<size_t>(i) * alpha_cols + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + j;
        size_t idx_left = static_cast<size_t>(i) * alpha_cols + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float sub_cost_val = s[score_idx];
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];
        bool trans_valid = (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j);

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost_val;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        float u_trans_val = 0.0f;
        int extra_del = 0, extra_ins = 0;

        if (trans_valid) {
            size_t idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            extra_del = i - trans_k - 1;
            extra_ins = j - trans_l - 1;
            v_trans = a[idx_trans] + trans_cost + extra_del * del_cost + extra_ins * ins_cost;
            u_trans_val = u[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

        float du_sub = u[idx_diag];
        float du_del = u[idx_up];
        float du_ins = u[idx_left];
        float du_trans = u_trans_val;

        if (param_type == DAMERAU_PARAM_INS) {
            du_ins += 1.0f;
            if (trans_valid) du_trans += (float)extra_ins;
        } else if (param_type == DAMERAU_PARAM_DEL) {
            du_del += 1.0f;
            if (trans_valid) du_trans += (float)extra_del;
        } else if (param_type == DAMERAU_PARAM_TRANS) {
            if (trans_valid) du_trans += 1.0f;
        }

        float U_val = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

        if (param_type == DAMERAU_PARAM_TEMPERATURE) {
            float alpha_ij = a[idx];
            if (alpha_ij < PINF) {
                float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
                U_val += (alpha_ij - E_v) / T;
            }
        }

        u[idx] = U_val;
    }
}

__global__ void damerau_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sub_costs,
    const int* __restrict__ trans_src,
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
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = sub_costs + (size_t)b * stride_scores;
    const int* ts = trans_src + (size_t)b * stride_scores * 2;
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
        float sub_cost_val = s[score_idx];
        int trans_k = ts[score_idx * 2];
        int trans_l = ts[score_idx * 2 + 1];
        bool trans_valid = (trans_k >= 0 && trans_l >= 0 && trans_k < i && trans_l < j);

        if (beta_ij <= 1e-20f && fabsf(W_ij) < 1e-20f) continue;

        float a_diag = a[idx_diag];
        float a_up = a[idx_up];
        float a_left = a[idx_left];

        float v_sub = a_diag + sub_cost_val;
        float v_del = a_up + del_cost;
        float v_ins = a_left + ins_cost;
        float v_trans = PINF;
        float u_trans_val = 0.0f;
        size_t idx_trans = 0;
        int extra_del = 0, extra_ins = 0;

        if (trans_valid) {
            idx_trans = static_cast<size_t>(trans_k) * alpha_cols + trans_l;
            extra_del = i - trans_k - 1;
            extra_ins = j - trans_l - 1;
            v_trans = a[idx_trans] + trans_cost + extra_del * del_cost + extra_ins * ins_cost;
            u_trans_val = u[idx_trans];
        }

        float w_sub, w_del, w_ins, w_trans;
        softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

        atomicAdd(&dP[score_idx], W_ij * w_sub);

        float du_sub = u[idx_diag];
        float du_del = u[idx_up];
        float du_ins = u[idx_left];
        float du_trans = u_trans_val;

        if (param_type == DAMERAU_PARAM_INS) {
            du_ins += 1.0f;
            if (trans_valid) du_trans += (float)extra_ins;
        } else if (param_type == DAMERAU_PARAM_DEL) {
            du_del += 1.0f;
            if (trans_valid) du_trans += (float)extra_del;
        } else if (param_type == DAMERAU_PARAM_TRANS) {
            if (trans_valid) du_trans += 1.0f;
        }

        float E_dv = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

        float dw_sub = w_sub * (-du_sub + E_dv) / T;
        float dw_del = w_del * (-du_del + E_dv) / T;
        float dw_ins = w_ins * (-du_ins + E_dv) / T;
        float dw_trans = w_trans * (-du_trans + E_dv) / T;

        if (param_type == DAMERAU_PARAM_TEMPERATURE) {
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
        if (w_trans > 0.0f && trans_valid) {
            atomicAdd(&be[idx_trans], beta_ij * w_trans);
            atomicAdd(&w_buf[idx_trans], W_ij * w_trans + beta_ij * dw_trans);
        }
    }
}

void damerau_param_grad(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
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
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = static_cast<size_t>(B) * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    int blocks_init = checked_grid_blocks(total_alpha, threads, "damerau_init_U_kernel");
    damerau_init_U_kernel<<<blocks_init, threads, 0, stream>>>(
        d_U, d_lengths, B, max_L1, max_L2, param_type
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dparam, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));

    int max_diag = max_L1 + max_L2;

    for (int k = 2; k <= max_diag; ++k) {
        damerau_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_src, d_U, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    damerau_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        damerau_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_sub_costs, d_trans_src, d_U,
            d_beta, d_W, d_dP_dparam, d_lengths,
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, T, param_type, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace damerau
}  // namespace d2p
