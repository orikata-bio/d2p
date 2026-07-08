// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft DTW CUDA Kernel Implementations
 *
 * Dynamic Time Warping using softmin (minimization problem).
 * Uses wavefront (anti-diagonal) parallelization for the DP.
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include "common/numerics.cuh"
#include "common/reduce.cuh"
#include "common/softmax.cuh"

using namespace d2p::common;

// ============================================================================
// DTW-specific helpers
// ============================================================================

// Check if cell (i, j) is within Sakoe-Chiba band
__device__ __forceinline__
bool in_band(int i, int j, int L1, int L2, int bandwidth) {
    if (bandwidth < 0) return true;  // No band constraint
    if (L1 == 0 || L2 == 0) return true;
    double expected_j = (static_cast<double>(i) * static_cast<double>(L2))
        / static_cast<double>(L1);
    return fabs(static_cast<double>(j) - expected_j) <= static_cast<double>(bandwidth);
}

// ============================================================================
// FORWARD PASS
// ============================================================================

// Initialize alpha: alpha[0,0] = 0, all others = +inf
__global__ void dtw_init_alpha_kernel(
    float* __restrict__ alpha,
    int B, int L1, int L2
) {
    size_t stride = (static_cast<size_t>(L1) + 1) * (static_cast<size_t>(L2) + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t rem = idx - b * stride;
    int i = rem / (L2 + 1);
    int j = rem % (L2 + 1);

    alpha[idx] = (i == 0 && j == 0) ? 0.0f : PINF;
}

// Forward DP for one anti-diagonal k = i + j
__global__ void dtw_forward_diag_kernel(
    const float* __restrict__ costs,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    float* a = alpha + (size_t)b * stride_alpha;
    const float* c = costs + (size_t)b * stride_costs;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // Cells outside actual lengths get +inf
        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * (static_cast<size_t>(max_L2) + 1) + j;
            a[idx] = PINF;
            continue;
        }

        // Bandwidth constraint
        if (!in_band(i, j, L1, L2, bandwidth)) {
            size_t idx = static_cast<size_t>(i) * (static_cast<size_t>(max_L2) + 1) + j;
            a[idx] = PINF;
            continue;
        }

        size_t stride = static_cast<size_t>(max_L2) + 1;
        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float cost = c[cost_idx];

        // Get predecessor values (check bandwidth for each)
        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        // DTW recurrence: alpha[i,j] = cost[i,j] + softmin(predecessors)
        a[idx] = cost + softmin3(a_diag, a_up, a_left, T);
    }
}

// Extract score: S = alpha[L1, L2]
__global__ void dtw_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ score,
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
    score[b] = alpha[static_cast<size_t>(b) * total_stride + final_idx];
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

// Initialize beta: beta[L1, L2] = 1, all others = 0
__global__ void dtw_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t rem = idx - b * stride;
    int i = rem / (max_L2 + 1);
    int j = rem % (max_L2 + 1);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

// Backward DP: propagate beta and compute posteriors
__global__ void dtw_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ costs,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* c = costs + (size_t)b * stride_costs;
    float* be = beta + (size_t)b * stride_alpha;
    float* post = posteriors + (size_t)b * stride_costs;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    float local_T_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;
        if (i > L1 || j > L2) continue;
        if (!in_band(i, j, L1, L2, bandwidth)) continue;

        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_ij = be[idx];
        if (beta_ij <= 1e-20f) continue;

        // For DTW, posteriors = beta
        atomicAdd(&post[cost_idx], beta_ij);

        // Compute softmin weights
        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        float w_diag, w_up, w_left;
        softmin3_weights(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

        // Propagate beta to predecessors
        if (w_diag > 0.0f && in_band(i-1, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_diag], beta_ij * w_diag);
        }
        if (w_up > 0.0f && in_band(i-1, j, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_up], beta_ij * w_up);
        }
        if (w_left > 0.0f && in_band(i, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_left], beta_ij * w_left);
        }

        // Temperature gradient: dS/dT = sum beta * (softmin - E[a]) / T
        float cost = c[cost_idx];
        float softmin_val = a[idx] - cost;
        if (softmin_val < PINF) {
            float E_a = w_diag * a_diag + w_up * a_up + w_left * a_left;
            local_T_grad += beta_ij * (softmin_val - E_a) / T;
        }
    }

    // Block reduce temperature gradient
    float block_T_grad = block_reduce_sum(local_T_grad);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_T_grad);
    }
}

// ============================================================================
// HVP (Hessian-Vector Product)
// ============================================================================

// Forward tangent pass: compute d_alpha
__global__ void dtw_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ costs,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* v = V + (size_t)b * stride_costs;
    float* da = d_alpha + (size_t)b * stride_alpha;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            da[idx] = 0.0f;
            continue;
        }
        if (!in_band(i, j, L1, L2, bandwidth)) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float v_ij = v[cost_idx];

        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        float w_diag, w_up, w_left;
        softmin3_weights(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

        float da_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? da[idx_diag] : 0.0f;
        float da_up = in_band(i-1, j, L1, L2, bandwidth) ? da[idx_up] : 0.0f;
        float da_left = in_band(i, j-1, L1, L2, bandwidth) ? da[idx_left] : 0.0f;

        // d(alpha[i,j]) = v_ij + sum w_k * da[pred_k]
        da[idx] = v_ij + w_diag * da_diag + w_up * da_up + w_left * da_left;
    }
}

// Compute d_score
__global__ void dtw_hvp_score_kernel(
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

// Backward tangent pass: compute HVP
__global__ void dtw_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ costs,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_costs,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* da = d_alpha + (size_t)b * stride_alpha;
    float* be = beta + (size_t)b * stride_alpha;
    float* dbe = d_beta + (size_t)b * stride_alpha;
    float* H = H_costs + (size_t)b * stride_costs;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;
        if (i > L1 || j > L2) continue;
        if (!in_band(i, j, L1, L2, bandwidth)) continue;

        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_ij = be[idx];
        float dbeta_ij = dbe[idx];

        // HVP: d(posteriors) = d(beta) = dbeta
        atomicAdd(&H[cost_idx], dbeta_ij);

        if (beta_ij <= 1e-20f) continue;

        // Compute weights
        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        float w_diag, w_up, w_left;
        softmin3_weights(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

        // Compute weight tangents for softmin
        // dw_k = w_k * (-da_k + E[da]) / T
        float da_diag_v = in_band(i-1, j-1, L1, L2, bandwidth) ? da[idx_diag] : 0.0f;
        float da_up_v = in_band(i-1, j, L1, L2, bandwidth) ? da[idx_up] : 0.0f;
        float da_left_v = in_band(i, j-1, L1, L2, bandwidth) ? da[idx_left] : 0.0f;

        float E_da = w_diag * da_diag_v + w_up * da_up_v + w_left * da_left_v;

        float dw_diag = w_diag * (-da_diag_v + E_da) / T;
        float dw_up = w_up * (-da_up_v + E_da) / T;
        float dw_left = w_left * (-da_left_v + E_da) / T;

        // Propagate beta and dbeta
        if (w_diag > 0.0f && in_band(i-1, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_diag], beta_ij * w_diag);
            atomicAdd(&dbe[idx_diag], dbeta_ij * w_diag + beta_ij * dw_diag);
        }
        if (w_up > 0.0f && in_band(i-1, j, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_up], beta_ij * w_up);
            atomicAdd(&dbe[idx_up], dbeta_ij * w_up + beta_ij * dw_up);
        }
        if (w_left > 0.0f && in_band(i, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_left], beta_ij * w_left);
            atomicAdd(&dbe[idx_left], dbeta_ij * w_left + beta_ij * dw_left);
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT (dP/dT)
// ============================================================================

// Forward U pass: compute U = d(alpha)/dT
__global__ void dtw_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ costs,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* c = costs + (size_t)b * stride_costs;
    float* u = U + (size_t)b * stride_alpha;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            u[idx] = 0.0f;
            continue;
        }
        if (!in_band(i, j, L1, L2, bandwidth)) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            u[idx] = 0.0f;
            continue;
        }

        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float cost = c[cost_idx];

        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        float w_diag, w_up, w_left;
        softmin3_weights(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

        float u_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? u[idx_diag] : 0.0f;
        float u_up = in_band(i-1, j, L1, L2, bandwidth) ? u[idx_up] : 0.0f;
        float u_left = in_band(i, j-1, L1, L2, bandwidth) ? u[idx_left] : 0.0f;

        // d(softmin)/dT = (softmin - E[a]) / T + sum w_k * U_k
        float softmin_val = a[idx] - cost;
        float E_a = w_diag * a_diag + w_up * a_up + w_left * a_left;

        u[idx] = (softmin_val - E_a) / T + w_diag * u_diag + w_up * u_up + w_left * u_left;
    }
}

// Backward W pass: compute W = d(beta)/dT and accumulate dP/dT
__global__ void dtw_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ costs,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_costs = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* u_buf = U + (size_t)b * stride_alpha;
    float* be = beta + (size_t)b * stride_alpha;
    float* w_buf = W + (size_t)b * stride_alpha;
    float* dP = dP_dT + (size_t)b * stride_costs;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;
        if (i > L1 || j > L2) continue;
        if (!in_band(i, j, L1, L2, bandwidth)) continue;

        size_t idx = static_cast<size_t>(i) * stride + j;
        size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
        size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_ij = be[idx];
        float W_ij = w_buf[idx];

        // dP/dT = W (since posteriors = beta)
        atomicAdd(&dP[cost_idx], W_ij);

        if (beta_ij <= 1e-20f) continue;

        float a_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? a[idx_diag] : PINF;
        float a_up = in_band(i-1, j, L1, L2, bandwidth) ? a[idx_up] : PINF;
        float a_left = in_band(i, j-1, L1, L2, bandwidth) ? a[idx_left] : PINF;

        float w_diag, w_up, w_left;
        softmin3_weights(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

        float u_diag_v = in_band(i-1, j-1, L1, L2, bandwidth) ? u_buf[idx_diag] : 0.0f;
        float u_up_v = in_band(i-1, j, L1, L2, bandwidth) ? u_buf[idx_up] : 0.0f;
        float u_left_v = in_band(i, j-1, L1, L2, bandwidth) ? u_buf[idx_left] : 0.0f;

        // Total derivative of the softmin weights w.r.t. temperature:
        // the predecessor alphas depend on T through U, and the
        // exp(-a / T) scaling contributes the explicit T term.
        float E_a = w_diag * a_diag + w_up * a_up + w_left * a_left;
        float E_u = w_diag * u_diag_v + w_up * u_up_v + w_left * u_left_v;
        float inv_T = 1.0f / T;
        float inv_T2 = inv_T * inv_T;
        float dw_diag_dT = w_diag * ((E_u - u_diag_v) * inv_T + (a_diag - E_a) * inv_T2);
        float dw_up_dT = w_up * ((E_u - u_up_v) * inv_T + (a_up - E_a) * inv_T2);
        float dw_left_dT = w_left * ((E_u - u_left_v) * inv_T + (a_left - E_a) * inv_T2);

        // Propagate beta and W
        if (w_diag > 0.0f && in_band(i-1, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_diag], beta_ij * w_diag);
            atomicAdd(&w_buf[idx_diag], W_ij * w_diag + beta_ij * dw_diag_dT);
        }
        if (w_up > 0.0f && in_band(i-1, j, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_up], beta_ij * w_up);
            atomicAdd(&w_buf[idx_up], W_ij * w_up + beta_ij * dw_up_dT);
        }
        if (w_left > 0.0f && in_band(i, j-1, L1, L2, bandwidth)) {
            atomicAdd(&be[idx_left], beta_ij * w_left);
            atomicAdd(&w_buf[idx_left], W_ij * w_left + beta_ij * dw_left_dT);
        }
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

void dtw_forward(
    const float* d_costs,
    float* d_alpha,
    float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = (size_t)B * alpha_elems;

    // Initialize alpha
    int blocks_init = (total_alpha + threads - 1) / threads;
    dtw_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(d_alpha, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Wavefront DP
    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        dtw_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_costs, d_alpha, d_lengths, B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Extract score
    int blocks_score = (B + threads - 1) / threads;
    dtw_score_kernel<<<blocks_score, threads, 0, stream>>>(d_alpha, d_score, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void dtw_backward(
    const float* d_alpha,
    const float* d_costs,
    const float* d_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t cost_elems = static_cast<size_t>(B) * static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    // Zero outputs
    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * cost_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    // Initialize beta
    int blocks_init = (total_alpha + threads - 1) / threads;
    dtw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward DP
    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        dtw_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_costs, d_beta, d_posteriors, d_grad_T, d_lengths,
            B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void dtw_hvp(
    const float* d_alpha,
    const float* d_costs,
    const float* d_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_costs,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t cost_elems = static_cast<size_t>(B) * static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_costs, 0, sizeof(float) * cost_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward tangent pass
    for (int k = 2; k <= max_diag; ++k) {
        dtw_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_costs, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Compute d_score
    int blocks_score = (B + threads - 1) / threads;
    dtw_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(d_d_alpha, d_d_score, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Initialize beta for backward
    int blocks_init = (total_alpha + threads - 1) / threads;
    dtw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward tangent pass
    for (int k = max_diag; k >= 2; --k) {
        dtw_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_costs, d_V, d_d_alpha, d_beta, d_d_beta, d_H_costs, d_lengths,
            B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void dtw_param_grad(
    const float* d_alpha,
    const float* d_costs,
    const float* d_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T, int bandwidth
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t cost_elems = static_cast<size_t>(B) * static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dT, 0, sizeof(float) * cost_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward U pass
    for (int k = 2; k <= max_diag; ++k) {
        dtw_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_costs, d_U, d_lengths,
            B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Initialize beta for backward
    int blocks_init = (total_alpha + threads - 1) / threads;
    dtw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward W pass
    for (int k = max_diag; k >= 2; --k) {
        dtw_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_costs, d_U, d_beta, d_W, d_dP_dT, d_lengths,
            B, max_L1, max_L2, T, bandwidth, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
