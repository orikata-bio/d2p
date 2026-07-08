// SPDX-License-Identifier: Apache-2.0
// kernels.cu
//
// Soft Needleman-Wunsch (Linear Gap) CUDA Kernels
// Global alignment algorithm using temperature-scaled softmax
//
// Key differences from Smith-Waterman:
//   - 3 transitions (diagonal+score, up+gap, left+gap) - no "sky" restart
//   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
//   - Score = alpha[L1, L2], not logsumexp over all cells
//   - Beta initialized at terminal only: beta[L1, L2] = 1

#include <cuda_runtime.h>
#include <math.h>

// Shared utilities
#include "common/cuda_utils.h"
#include "common/numerics.cuh"
#include "common/reduce.cuh"
#include "common/softmax.cuh"

using namespace d2p::common;

// Parameter type enum for param gradient kernels
enum NWParamType {
    NW_PARAM_GAP = 0,
    NW_PARAM_TEMPERATURE = 1
};

// Initialize boundary U values for gap sensitivity:
// alpha[i,0] = i*gap and alpha[0,j] = j*gap, so d(alpha)/d(gap) is i or j.
__global__ void nw_init_gap_param_boundary_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    size_t cell_stride = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * cell_stride;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / cell_stride;
    int j   = rem % cell_stride;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i > L1 || j > L2) return;

    if (j == 0) {
        U[idx] = static_cast<float>(i);
    } else if (i == 0) {
        U[idx] = static_cast<float>(j);
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

// Initialize alpha with NW base cases:
// alpha[0,0] = 0
// alpha[i,0] = i * gap for i > 0
// alpha[0,j] = j * gap for j > 0
// All other cells = -inf
__global__ void nw_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,  // [B, 2]
    int B, int max_L1, int max_L2,
    float gap
) {
    size_t cell_stride = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * cell_stride;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / cell_stride;
    int j   = rem % cell_stride;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Cells outside actual lengths get -inf
    if (i > L1 || j > L2) {
        alpha[idx] = NINF;
        return;
    }

    // Base cases for global alignment
    if (i == 0 && j == 0) {
        alpha[idx] = 0.0f;
    } else if (i == 0) {
        alpha[idx] = j * gap;  // alpha[0,j] = j * gap
    } else if (j == 0) {
        alpha[idx] = i * gap;  // alpha[i,0] = i * gap
    } else {
        alpha[idx] = NINF;  // Will be computed in forward pass
    }
}

// Forward DP for one anti-diagonal k = i + j
// Three transitions: diagonal+score, up+gap, left+gap (global alignment - no restart)
__global__ void nw_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t stride_scores = (size_t)max_L1 * max_L2;

    // Per-batch actual lengths
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
            size_t idx = static_cast<size_t>(i) * (static_cast<size_t>(max_L2) + 1) + j;
            a[idx] = NINF;
            continue;
        }

        size_t stride = static_cast<size_t>(max_L2) + 1;
        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float score = s[score_idx];

        // Three logits for soft-max (option-additive: score/gap added to options)
        float v_diag = a[idx_diag] + score;  // diagonal + match score
        float v_up   = a[idx_up]   + gap;    // up + gap penalty
        float v_left = a[idx_left] + gap;    // left + gap penalty

        // NW recurrence: alpha[i,j] = logsumexp(v_diag, v_up, v_left)
        a[idx] = logsumexp3(v_diag, v_up, v_left, T);
    }
}

// Extract score: S = alpha[L1, L2] (global alignment endpoint)
__global__ void nw_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ score,
    const int* __restrict__ lengths,  // [B, 2]
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
__global__ void nw_init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,  // [B, 2]
    int B, int max_L1, int max_L2
) {
    size_t cell_stride = static_cast<size_t>(max_L2) + 1;
    size_t stride = (static_cast<size_t>(max_L1) + 1) * cell_stride;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = rem / cell_stride;
    int j   = rem % cell_stride;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Terminal condition: beta[L1, L2] = 1
    beta[idx] = (i == L1 && j == L2) ? 1.0f : 0.0f;
}

// Backward DP: propagate beta and compute posteriors, grad_gap
__global__ void nw_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_gap,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * stride;
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

    float local_gap_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) continue;

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_ij = be[idx];
        if (beta_ij <= 1e-20f) continue;

        float score = s[score_idx];

        // Recompute logits (must match forward exactly)
        float v_diag = a[idx_diag] + score;
        float v_up   = a[idx_up]   + gap;
        float v_left = a[idx_left] + gap;

        // Compute softmax weights
        float w_diag, w_up, w_left;
        softmax3_weights(v_diag, v_up, v_left, T, w_diag, w_up, w_left);

        // Posteriors: P[i,j] = beta * w_diag (option-additive)
        atomicAdd(&post[score_idx], beta_ij * w_diag);

        // Gap gradient: sum beta * (w_up + w_left)
        local_gap_grad += beta_ij * (w_up + w_left);

        // Propagate beta to predecessors
        if (w_diag > 0.0f) atomicAdd(&be[idx_diag], beta_ij * w_diag);
        if (w_up > 0.0f)   atomicAdd(&be[idx_up],   beta_ij * w_up);
        if (w_left > 0.0f) atomicAdd(&be[idx_left], beta_ij * w_left);
    }

    // Block reduce gap gradient
    float block_gap_grad = block_reduce_sum(local_gap_grad);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_gap[b], block_gap_grad);
    }
}

// Boundary gap-gradient accumulation: beta mass at boundary cells (i,0) and
// (0,j) encodes gap-only paths.  alpha[i,0] = i*gap, alpha[0,j] = j*gap, so
// d(alpha)/d(gap) = i or j respectively.  The backward diag kernel never
// visits these cells because it requires i>=1 AND j>=1.
__global__ void nw_boundary_gap_grad_kernel(
    const float* __restrict__ beta,
    float* __restrict__ grad_gap,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (size_t)(max_L1 + 1) * stride;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* be = beta + (size_t)b * stride_alpha;

    float local_sum = 0.0f;

    // Column boundary: beta[i, 0] for i = 1..L1
    for (int i = 1 + (int)threadIdx.x; i <= L1; i += (int)blockDim.x) {
        local_sum += be[static_cast<size_t>(i) * stride] * static_cast<float>(i);
    }

    // Row boundary: beta[0, j] for j = 1..L2
    for (int j = 1 + (int)threadIdx.x; j <= L2; j += (int)blockDim.x) {
        local_sum += be[j] * static_cast<float>(j);
    }

    float block_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_gap[b], block_sum);
    }
}

// Compute dS/dT = (S - E[total_score]) / T
__global__ void nw_grad_T_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ posteriors,
    const float* __restrict__ grad_gap,
    const float* __restrict__ partition,
    float* __restrict__ grad_T,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int b = blockIdx.x;
    size_t size = (size_t)max_L1 * max_L2;

    const float* p = posteriors + (size_t)b * size;
    const float* s = scores + (size_t)b * size;

    // E[match_score] = sum_ij posteriors[i,j] * scores[i,j]
    float local_sum = 0.0f;
    for (size_t idx = threadIdx.x; idx < size; idx += blockDim.x) {
        local_sum += p[idx] * s[idx];
    }
    float match_sum = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        float expected_gap_cost = grad_gap[b] * gap;
        float expected_total = match_sum + expected_gap_cost;
        grad_T[b] = (partition[b] - expected_total) / T;
    }
}

// ============================================================================
// HVP (Hessian-Vector Product)
// ============================================================================

// Forward tangent pass: compute d_alpha
__global__ void nw_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * stride;
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

        // BOUNDS CHECK
        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float score = s[score_idx];
        float v_ij = v[score_idx];

        // Recompute logits
        float log_diag = a[idx_diag] + score;
        float log_up   = a[idx_up]   + gap;
        float log_left = a[idx_left] + gap;

        float w_diag, w_up, w_left;
        softmax3_weights(log_diag, log_up, log_left, T, w_diag, w_up, w_left);

        // Tangent of logits: diagonal includes V, gap options don't
        float dv_diag = da[idx_diag] + v_ij;  // d(alpha_diag + score) = da_diag + V
        float dv_up   = da[idx_up];            // d(alpha_up + gap) = da_up
        float dv_left = da[idx_left];          // d(alpha_left + gap) = da_left

        // d(alpha[i,j]) = sum w_k * dv_k
        da[idx] = w_diag * dv_diag + w_up * dv_up + w_left * dv_left;
    }
}

// Compute d_score
__global__ void nw_hvp_score_kernel(
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
__global__ void nw_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * stride;
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

        // BOUNDS CHECK
        if (i > L1 || j > L2) continue;

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_ij = be[idx];
        float dbeta_ij = dbe[idx];

        if (beta_ij <= 1e-20f && fabsf(dbeta_ij) <= 1e-20f) continue;

        float score = s[score_idx];
        float v_ij = v[score_idx];

        // Recompute logits and weights
        float log_diag = a[idx_diag] + score;
        float log_up   = a[idx_up]   + gap;
        float log_left = a[idx_left] + gap;

        float w_diag, w_up, w_left;
        softmax3_weights(log_diag, log_up, log_left, T, w_diag, w_up, w_left);

        // Tangent of logits
        float dv_diag = da[idx_diag] + v_ij;
        float dv_up   = da[idx_up];
        float dv_left = da[idx_left];

        // Compute weight tangents for softmax
        // dw_k = w_k * (dv_k - E[dv]) / T
        float E_dv = w_diag * dv_diag + w_up * dv_up + w_left * dv_left;

        float dw_diag = w_diag * (dv_diag - E_dv) / T;
        float dw_up   = w_up   * (dv_up   - E_dv) / T;
        float dw_left = w_left * (dv_left - E_dv) / T;

        // HVP: d(posteriors) = dbeta * w_diag + beta * dw_diag
        atomicAdd(&H[score_idx], dbeta_ij * w_diag + beta_ij * dw_diag);

        // Propagate beta and dbeta
        if (w_diag > 0.0f) {
            atomicAdd(&be[idx_diag], beta_ij * w_diag);
            atomicAdd(&dbe[idx_diag], dbeta_ij * w_diag + beta_ij * dw_diag);
        }
        if (w_up > 0.0f) {
            atomicAdd(&be[idx_up], beta_ij * w_up);
            atomicAdd(&dbe[idx_up], dbeta_ij * w_up + beta_ij * dw_up);
        }
        if (w_left > 0.0f) {
            atomicAdd(&be[idx_left], beta_ij * w_left);
            atomicAdd(&dbe[idx_left], dbeta_ij * w_left + beta_ij * dw_left);
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT (dP/d{gap, T})
// ============================================================================

// Forward U pass: compute U = d(alpha)/d{gap, T}
__global__ void nw_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * stride;
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

        // BOUNDS CHECK
        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            u[idx] = 0.0f;
            continue;
        }

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float score = s[score_idx];

        // Recompute logits
        float v_diag = a[idx_diag] + score;
        float v_up   = a[idx_up]   + gap;
        float v_left = a[idx_left] + gap;

        float w_diag, w_up, w_left;
        softmax3_weights(v_diag, v_up, v_left, T, w_diag, w_up, w_left);

        // Get predecessor U values
        float u_diag = u[idx_diag];
        float u_up   = u[idx_up];
        float u_left = u[idx_left];

        // For gap parameter: add +1 tangent to gap transitions
        if (param_type == NW_PARAM_GAP) {
            u_up   += 1.0f;
            u_left += 1.0f;
        }

        // Propagate tangent: U[idx] = sum w_k * u_k
        float U_val = w_diag * u_diag + w_up * u_up + w_left * u_left;

        // For temperature: add (alpha - E[v]) / T term
        if (param_type == NW_PARAM_TEMPERATURE) {
            float alpha_ij = a[idx];
            float E_v = w_diag * v_diag + w_up * v_up + w_left * v_left;
            U_val += (alpha_ij - E_v) / T;
        }

        u[idx] = U_val;
    }
}

// Backward W pass: compute W = d(beta)/d{gap, T} and accumulate dP/d{gap, T}
__global__ void nw_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ U,
    const float* __restrict__ dS_dtheta,  // [B] pre-computed dS/dtheta
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dtheta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (size_t)(max_L1 + 1) * stride;
    size_t stride_scores = (size_t)max_L1 * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    const float* u_buf = U + (size_t)b * stride_alpha;
    float* b_buf = beta + (size_t)b * stride_alpha;
    float* w_buf = W + (size_t)b * stride_alpha;
    float* dP = dP_dtheta + (size_t)b * stride_scores;

    float S = partition[b];
    float dS = dS_dtheta[b];

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

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + (j - 1);

        float beta_curr = b_buf[idx];

        // For global alignment, beta = 1 at terminal, so initial beta_init depends on distance from terminal
        // Here we use the accumulated beta and compute W tangent
        float w_curr = w_buf[idx];

        if (beta_curr <= 1e-20f && fabsf(w_curr) <= 1e-20f) continue;

        float score = s[score_idx];

        // Recompute logits and weights
        float v_diag = a[idx_diag] + score;
        float v_up   = a[idx_up]   + gap;
        float v_left = a[idx_left] + gap;

        float w_diag, w_up, w_left;
        softmax3_weights(v_diag, v_up, v_left, T, w_diag, w_up, w_left);

        // Get predecessor U values
        float u_diag = u_buf[idx_diag];
        float u_up   = u_buf[idx_up];
        float u_left = u_buf[idx_left];

        // Add gap tangent
        if (param_type == NW_PARAM_GAP) {
            u_up   += 1.0f;
            u_left += 1.0f;
        }

        // Compute weight tangents via softmax Jacobian
        // dw_k = w_k * (du_k - E[du]) / T
        float E_du = w_diag * u_diag + w_up * u_up + w_left * u_left;
        float dw_diag = w_diag * (u_diag - E_du) / T;
        float dw_up   = w_up   * (u_up   - E_du) / T;
        float dw_left = w_left * (u_left - E_du) / T;

        // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
        if (param_type == NW_PARAM_TEMPERATURE) {
            float E_v = w_diag * v_diag + w_up * v_up + w_left * v_left;
            float inv_T2 = 1.0f / (T * T);
            dw_diag += w_diag * (E_v - v_diag) * inv_T2;
            dw_up   += w_up   * (E_v - v_up)   * inv_T2;
            dw_left += w_left * (E_v - v_left) * inv_T2;
        }

        // Accumulate dP/dtheta: posteriors = beta * w_diag
        // d(posteriors) = w_curr * w_diag + beta_curr * dw_diag
        atomicAdd(&dP[score_idx], w_curr * w_diag + beta_curr * dw_diag);

        // Propagate beta and W to predecessors
        if (w_diag > 0.0f) {
            atomicAdd(&b_buf[idx_diag], beta_curr * w_diag);
            atomicAdd(&w_buf[idx_diag], w_curr * w_diag + beta_curr * dw_diag);
        }
        if (w_up > 0.0f) {
            atomicAdd(&b_buf[idx_up], beta_curr * w_up);
            atomicAdd(&w_buf[idx_up], w_curr * w_up + beta_curr * dw_up);
        }
        if (w_left > 0.0f) {
            atomicAdd(&b_buf[idx_left], beta_curr * w_left);
            atomicAdd(&w_buf[idx_left], w_curr * w_left + beta_curr * dw_left);
        }
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

extern "C" void nw_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;

    // Initialize alpha with NW base cases
    int blocks_init = (total_alpha + threads - 1) / threads;
    nw_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(d_alpha, d_lengths, B, max_L1, max_L2, gap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Wavefront DP
    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        nw_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Extract score
    int blocks_score = (B + threads - 1) / threads;
    nw_score_kernel<<<blocks_score, threads, 0, stream>>>(d_alpha, d_score, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_gap,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    // Zero outputs
    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_gap, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    // Initialize beta
    int blocks_init = (total_alpha + threads - 1) / threads;
    nw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward DP
    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        nw_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_gap, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Boundary gap-gradient: beta mass at (i,0) and (0,j) cells
    nw_boundary_gap_grad_kernel<<<B, threads, 0, stream>>>(
        d_beta, d_grad_gap, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Temperature gradient
    nw_grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_grad_gap, d_score, d_grad_T,
        B, max_L1, max_L2, gap, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward tangent pass
    for (int k = 2; k <= max_diag; ++k) {
        nw_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Compute d_score
    int blocks_score = (B + threads - 1) / threads;
    nw_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(d_d_alpha, d_d_score, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Initialize beta for backward
    int blocks_init = (total_alpha + threads - 1) / threads;
    nw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward tangent pass
    for (int k = max_diag; k >= 2; --k) {
        nw_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_score,
    const float* d_dS_dtheta,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dtheta,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int param_type
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (size_t)(max_L1 + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = (size_t)B * max_L1 * max_L2;

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dtheta, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;
    int blocks_init = (total_alpha + threads - 1) / threads;

    if (param_type == NW_PARAM_GAP) {
        nw_init_gap_param_boundary_U_kernel<<<blocks_init, threads, 0, stream>>>(
            d_U, d_lengths, B, max_L1, max_L2
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Forward U pass
    for (int k = 2; k <= max_diag; ++k) {
        nw_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths,
            B, max_L1, max_L2, gap, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Initialize beta for backward
    nw_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(d_beta, d_lengths, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward W pass
    for (int k = max_diag; k >= 2; --k) {
        nw_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_score, d_U, d_dS_dtheta,
            d_beta, d_W, d_dP_dtheta, d_lengths,
            B, max_L1, max_L2, gap, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
