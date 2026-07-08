// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft Needleman-Wunsch Affine Gap CUDA Kernels
 *
 * Global alignment algorithm using SOFTMAX (maximization) with gap open/extend penalties
 *
 * Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
 *
 * Key differences from SW affine:
 *   - No "sky" restart transition in M state (global alignment)
 *   - Base cases: M(0,0)=0, I(i,0)=g_o+(i-1)*g_e, D(0,j)=g_o+(j-1)*g_e
 *   - Score = LSE(M(L1,L2), I(L1,L2), D(L1,L2)), not over all cells
 *   - Beta initialized at terminal only
 */

#include <cuda_runtime.h>
#include <math.h>

// Shared utilities
#include "common/cuda_utils.h"
#include "common/numerics.cuh"
#include "common/reduce.cuh"

using namespace d2p::common;

// Parameter types for param gradient kernels
enum NWAffineParamType {
    NW_AFFINE_PARAM_GAP_OPEN = 0,
    NW_AFFINE_PARAM_GAP_EXT = 1,
    NW_AFFINE_PARAM_TEMPERATURE = 2
};

// ============================================================================
// Softmax Helpers (3-way for NW affine)
// ============================================================================

// Compute soft-max weights for 3 logits, return logsumexp value
__device__ __forceinline__ float softmax3_with_lse(
    float v1, float v2, float v3, float T,
    float& w1, float& w2, float& w3
) {
    float max_v = fmaxf(fmaxf(v1, v2), v3);
    if (max_v <= NINF) {
        w1 = w2 = w3 = 0.0f;
        return NINF;
    }
    w1 = safe_exp((v1 - max_v) / T);
    w2 = safe_exp((v2 - max_v) / T);
    w3 = safe_exp((v3 - max_v) / T);
    float sum_w = w1 + w2 + w3;
    float inv = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;
    w1 *= inv;
    w2 *= inv;
    w3 *= inv;
    return max_v + T * logf(fmaxf(sum_w, 1e-30f));
}

// Compute tangent of 3-way soft-max weights (softmax Jacobian)
__device__ __forceinline__ float softmax3_tangent(
    float w1, float w2, float w3,
    float dv1, float dv2, float dv3,
    float T,
    float& dw1, float& dw2, float& dw3
) {
    float sum_w_dv = w1 * dv1 + w2 * dv2 + w3 * dv3;
    dw1 = w1 * (dv1 - sum_w_dv) / T;
    dw2 = w2 * (dv2 - sum_w_dv) / T;
    dw3 = w3 * (dv3 - sum_w_dv) / T;
    return sum_w_dv;
}

// ============================================================================
// FORWARD PASS
// ============================================================================

// Initialize alpha with NW affine base cases
__global__ void nw_affine_init_alpha_kernel(
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext
) {
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride_all;
    if (idx >= total) return;

    size_t b = idx / stride_all;
    size_t rem = idx - b * stride_all;
    int state = (int)(rem / stride_state);
    size_t off = rem - (size_t)state * stride_state;
    int i = (int)(off / stride_row);
    int j = (int)(off % stride_row);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Cells outside actual lengths get -inf
    if (i > L1 || j > L2) {
        alpha[idx] = NINF;
        return;
    }

    // State 0 = M, State 1 = I, State 2 = D
    if (state == 0) {
        // M state: M(0,0) = 0, all others = -inf
        alpha[idx] = (i == 0 && j == 0) ? 0.0f : NINF;
    } else if (state == 1) {
        // I state: I(i,0) = gap_open + (i-1)*gap_ext for i > 0
        if (j == 0 && i > 0) {
            alpha[idx] = gap_open + (i - 1) * gap_ext;
        } else {
            alpha[idx] = NINF;
        }
    } else {
        // D state: D(0,j) = gap_open + (j-1)*gap_ext for j > 0
        if (i == 0 && j > 0) {
            alpha[idx] = gap_open + (j - 1) * gap_ext;
        } else {
            alpha[idx] = NINF;
        }
    }
}

// Forward DP for one anti-diagonal
__global__ void nw_affine_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t score_row = (size_t)max_L2;

    float* M = alpha + (size_t)b * stride_all;
    float* I = M + stride_state;
    float* D = I + stride_state;
    const float* s = scores + (size_t)b * stride_scores;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK
        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            M[idx] = NINF;
            I[idx] = NINF;
            D[idx] = NINF;
            continue;
        }

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float score = s[score_idx];

        // M state: from M/I/D at diagonal + score (NO SKY)
        M[idx] = score + logsumexp3(M[idx_diag], I[idx_diag], D[idx_diag], T);

        // I state (gap in seq2, vertical)
        I[idx] = logsumexp3(M[idx_up] + gap_open, I[idx_up] + gap_ext, D[idx_up] + gap_open, T);

        // D state (gap in seq1, horizontal)
        D[idx] = logsumexp3(M[idx_left] + gap_open, I[idx_left] + gap_open, D[idx_left] + gap_ext, T);
    }
}

// Score kernel: S = LSE(M[L1,L2], I[L1,L2], D[L1,L2])
__global__ void nw_affine_score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;

    const float* M = alpha + (size_t)b * stride_all;
    const float* I = M + stride_state;
    const float* D = I + stride_state;

    size_t final_idx = (size_t)L1 * stride_row + (size_t)L2;
    score[b] = logsumexp3(M[final_idx], I[final_idx], D[final_idx], T);
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

// Initialize beta at terminal
__global__ void nw_affine_init_beta_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2, float T
) {
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride_all;
    if (idx >= total) return;

    size_t b = idx / stride_all;
    size_t rem = idx - b * stride_all;
    int state = (int)(rem / stride_state);
    size_t off = rem - (size_t)state * stride_state;
    int i = (int)(off / stride_row);
    int j = (int)(off % stride_row);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i == L1 && j == L2) {
        const float* M_base = alpha + (size_t)b * stride_all;
        const float* I_base = M_base + stride_state;
        const float* D_base = I_base + stride_state;

        size_t final_idx = (size_t)L1 * stride_row + (size_t)L2;
        float m_val = M_base[final_idx];
        float i_val = I_base[final_idx];
        float d_val = D_base[final_idx];

        float w_m, w_i, w_d;
        softmax3_with_lse(m_val, i_val, d_val, T, w_m, w_i, w_d);

        if (state == 0) {
            beta[idx] = w_m;
        } else if (state == 1) {
            beta[idx] = w_i;
        } else {
            beta[idx] = w_d;
        }
    } else {
        beta[idx] = 0.0f;
    }
}

// Initialize d_beta at terminal for HVP
__global__ void nw_affine_init_d_beta_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ d_alpha,
    float* __restrict__ d_beta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2, float T
) {
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride_all;
    if (idx >= total) return;

    size_t b = idx / stride_all;
    size_t rem = idx - b * stride_all;
    int state = (int)(rem / stride_state);
    size_t off = rem - (size_t)state * stride_state;
    int i = (int)(off / stride_row);
    int j = (int)(off % stride_row);

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (i == L1 && j == L2) {
        const float* A_M = alpha + (size_t)b * stride_all;
        const float* A_I = A_M + stride_state;
        const float* A_D = A_I + stride_state;

        const float* dA_M = d_alpha + (size_t)b * stride_all;
        const float* dA_I = dA_M + stride_state;
        const float* dA_D = dA_I + stride_state;

        size_t final_idx = (size_t)L1 * stride_row + (size_t)L2;
        float m_val = A_M[final_idx];
        float i_val = A_I[final_idx];
        float d_val = A_D[final_idx];

        float dm_val = dA_M[final_idx];
        float di_val = dA_I[final_idx];
        float dd_val = dA_D[final_idx];

        float w_m, w_i, w_d;
        softmax3_with_lse(m_val, i_val, d_val, T, w_m, w_i, w_d);

        float dw_m, dw_i, dw_d;
        softmax3_tangent(w_m, w_i, w_d, dm_val, di_val, dd_val, T, dw_m, dw_i, dw_d);

        if (state == 0) {
            d_beta[idx] = dw_m;
        } else if (state == 1) {
            d_beta[idx] = dw_i;
        } else {
            d_beta[idx] = dw_d;
        }
    } else {
        d_beta[idx] = 0.0f;
    }
}

// Backward DP: propagate beta, accumulate gradients
__global__ void nw_affine_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_open,
    float* __restrict__ grad_ext,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t score_row = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    float* post = posteriors + (size_t)b * stride_scores;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    float local_open = 0.0f;
    float local_ext = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) continue;

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float betaM = B_M[idx];
        float betaI = B_I[idx];
        float betaD = B_D[idx];

        // M state backprop
        if (betaM > 1e-20f) {
            float m1 = A_M[idx_diag];
            float m2 = A_I[idx_diag];
            float m3 = A_D[idx_diag];
            float w1, w2, w3;
            if (softmax3_with_lse(m1, m2, m3, T, w1, w2, w3) > NINF) {
                if (m1 > NINF) atomicAdd(&B_M[idx_diag], betaM * w1);
                if (m2 > NINF) atomicAdd(&B_I[idx_diag], betaM * w2);
                if (m3 > NINF) atomicAdd(&B_D[idx_diag], betaM * w3);
                atomicAdd(&post[score_idx], betaM);
            }
        }

        // I state backprop
        if (betaI > 1e-20f) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float i3 = A_D[idx_up] + gap_open;
            float w1, w2, w3;
            if (softmax3_with_lse(i1, i2, i3, T, w1, w2, w3) > NINF) {
                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(&B_I[idx_up], betaI * w2);
                atomicAdd(&B_D[idx_up], betaI * w3);
                local_open += betaI * (w1 + w3);
                local_ext += betaI * w2;
            }
        }

        // D state backprop
        if (betaD > 1e-20f) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;
            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(&B_D[idx_left], betaD * w3);
                local_open += betaD * (w1 + w2);
                local_ext += betaD * w3;
            }
        }
    }

    float block_open = block_reduce_sum(local_open);
    float block_ext = block_reduce_sum(local_ext);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_open[b], block_open);
        atomicAdd(&grad_ext[b], block_ext);
    }
}

// Boundary gap-gradient accumulation for leading-gap states.
// I(i,0) = gap_open + (i-1)*gap_ext, D(0,j) = gap_open + (j-1)*gap_ext.
// The backward diag kernel never visits i=0 or j=0 cells, so accumulated
// B_I / B_D beta mass at boundaries is lost without this kernel.
__global__ void nw_affine_boundary_gap_grad_kernel(
    const float* __restrict__ beta,
    float* __restrict__ grad_open,
    float* __restrict__ grad_ext,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2
) {
    int b = blockIdx.x;
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* B_I = beta + (size_t)b * stride_all + stride_state;
    const float* B_D = B_I + stride_state;

    float local_open = 0.0f;
    float local_ext = 0.0f;

    // Column boundary: B_I[i, 0] for i = 1..L1
    for (int i = 1 + (int)threadIdx.x; i <= L1; i += (int)blockDim.x) {
        float beta_boundary = B_I[(size_t)i * stride_row];
        if (beta_boundary > 1e-20f) {
            local_open += beta_boundary;
            local_ext += beta_boundary * static_cast<float>(i - 1);
        }
    }

    // Row boundary: B_D[0, j] for j = 1..L2
    for (int j = 1 + (int)threadIdx.x; j <= L2; j += (int)blockDim.x) {
        float beta_boundary = B_D[j];
        if (beta_boundary > 1e-20f) {
            local_open += beta_boundary;
            local_ext += beta_boundary * static_cast<float>(j - 1);
        }
    }

    float block_open = block_reduce_sum(local_open);
    float block_ext = block_reduce_sum(local_ext);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_open[b], block_open);
        atomicAdd(&grad_ext[b], block_ext);
    }
}

// Compute dS/dT
__global__ void nw_affine_grad_T_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ posteriors,
    const float* __restrict__ grad_open,
    const float* __restrict__ grad_ext,
    const float* __restrict__ partition,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t score_row = (size_t)max_L2;

    const float* p = posteriors + (size_t)b * stride_scores;
    const float* s = scores + (size_t)b * stride_scores;

    float local_sum = 0.0f;
    size_t valid_size = (size_t)L1 * (size_t)L2;
    for (size_t t = (size_t)threadIdx.x; t < valid_size; t += (size_t)blockDim.x) {
        int i = (int)(t / (size_t)L2);
        int j = (int)(t % (size_t)L2);
        size_t idx = (size_t)i * score_row + (size_t)j;
        local_sum += p[idx] * s[idx];
    }
    float match_sum = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        float expected_gap_cost = grad_open[b] * gap_open + grad_ext[b] * gap_ext;
        float expected_total = match_sum + expected_gap_cost;
        grad_T[b] = (partition[b] - expected_total) / T;
    }
}

// ============================================================================
// HVP (Hessian-Vector Product)
// ============================================================================

// Forward tangent pass
__global__ void nw_affine_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t score_row = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;
    const float* v = V + (size_t)b * stride_scores;

    float* dA_M = d_alpha + (size_t)b * stride_all;
    float* dA_I = dA_M + stride_state;
    float* dA_D = dA_I + stride_state;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            dA_M[idx] = 0.0f;
            dA_I[idx] = 0.0f;
            dA_D[idx] = 0.0f;
            continue;
        }

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float V_s = v[score_idx];

        // M state tangent
        {
            float m1 = A_M[idx_diag];
            float m2 = A_I[idx_diag];
            float m3 = A_D[idx_diag];
            float w1, w2, w3;
            if (softmax3_with_lse(m1, m2, m3, T, w1, w2, w3) > NINF) {
                float dm1 = dA_M[idx_diag];
                float dm2 = dA_I[idx_diag];
                float dm3 = dA_D[idx_diag];
                dA_M[idx] = V_s + w1 * dm1 + w2 * dm2 + w3 * dm3;
            } else {
                dA_M[idx] = 0.0f;
            }
        }

        // I state tangent
        {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float i3 = A_D[idx_up] + gap_open;
            float w1, w2, w3;
            if (softmax3_with_lse(i1, i2, i3, T, w1, w2, w3) > NINF) {
                float di1 = dA_M[idx_up];
                float di2 = dA_I[idx_up];
                float di3 = dA_D[idx_up];
                dA_I[idx] = w1 * di1 + w2 * di2 + w3 * di3;
            } else {
                dA_I[idx] = 0.0f;
            }
        }

        // D state tangent
        {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;
            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                float dd1 = dA_M[idx_left];
                float dd2 = dA_I[idx_left];
                float dd3 = dA_D[idx_left];
                dA_D[idx] = w1 * dd1 + w2 * dd2 + w3 * dd3;
            } else {
                dA_D[idx] = 0.0f;
            }
        }
    }
}

// Compute tangent of score
__global__ void nw_affine_hvp_score_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ d_alpha,
    float* __restrict__ d_score,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    const float* dA_M = d_alpha + (size_t)b * stride_all;
    const float* dA_I = dA_M + stride_state;
    const float* dA_D = dA_I + stride_state;

    size_t final_idx = (size_t)L1 * stride_row + (size_t)L2;

    float m_val = A_M[final_idx];
    float i_val = A_I[final_idx];
    float d_val = A_D[final_idx];

    float w_m, w_i, w_d;
    softmax3_with_lse(m_val, i_val, d_val, T, w_m, w_i, w_d);

    d_score[b] = w_m * dA_M[final_idx] + w_i * dA_I[final_idx] + w_d * dA_D[final_idx];
}

// Backward tangent pass
__global__ void nw_affine_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t score_row = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    const float* dA_M = d_alpha + (size_t)b * stride_all;
    const float* dA_I = dA_M + stride_state;
    const float* dA_D = dA_I + stride_state;

    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    float* dB_M = d_beta + (size_t)b * stride_all;
    float* dB_I = dB_M + stride_state;
    float* dB_D = dB_I + stride_state;

    float* H = H_scores + (size_t)b * stride_scores;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) continue;

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float betaM = B_M[idx];
        float betaI = B_I[idx];
        float betaD = B_D[idx];
        float dbetaM = dB_M[idx];
        float dbetaI = dB_I[idx];
        float dbetaD = dB_D[idx];

        // M state
        if (betaM > 1e-20f || fabsf(dbetaM) > 1e-20f) {
            float m1 = A_M[idx_diag];
            float m2 = A_I[idx_diag];
            float m3 = A_D[idx_diag];
            float w1, w2, w3;

            if (softmax3_with_lse(m1, m2, m3, T, w1, w2, w3) > NINF) {
                float dm1 = dA_M[idx_diag];
                float dm2 = dA_I[idx_diag];
                float dm3 = dA_D[idx_diag];
                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, dm1, dm2, dm3, T, dw1, dw2, dw3);

                if (m1 > NINF) {
                    atomicAdd(&B_M[idx_diag], betaM * w1);
                    atomicAdd(&dB_M[idx_diag], dbetaM * w1 + betaM * dw1);
                }
                if (m2 > NINF) {
                    atomicAdd(&B_I[idx_diag], betaM * w2);
                    atomicAdd(&dB_I[idx_diag], dbetaM * w2 + betaM * dw2);
                }
                if (m3 > NINF) {
                    atomicAdd(&B_D[idx_diag], betaM * w3);
                    atomicAdd(&dB_D[idx_diag], dbetaM * w3 + betaM * dw3);
                }

                atomicAdd(&H[score_idx], dbetaM);
            }
        }

        // I state
        if (betaI > 1e-20f || fabsf(dbetaI) > 1e-20f) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float i3 = A_D[idx_up] + gap_open;
            float w1, w2, w3;

            if (softmax3_with_lse(i1, i2, i3, T, w1, w2, w3) > NINF) {
                float di1 = dA_M[idx_up];
                float di2 = dA_I[idx_up];
                float di3 = dA_D[idx_up];
                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, di1, di2, di3, T, dw1, dw2, dw3);

                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(&dB_M[idx_up], dbetaI * w1 + betaI * dw1);
                atomicAdd(&B_I[idx_up], betaI * w2);
                atomicAdd(&dB_I[idx_up], dbetaI * w2 + betaI * dw2);
                atomicAdd(&B_D[idx_up], betaI * w3);
                atomicAdd(&dB_D[idx_up], dbetaI * w3 + betaI * dw3);
            }
        }

        // D state
        if (betaD > 1e-20f || fabsf(dbetaD) > 1e-20f) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;

            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                float dd1 = dA_M[idx_left];
                float dd2 = dA_I[idx_left];
                float dd3 = dA_D[idx_left];
                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, dd1, dd2, dd3, T, dw1, dw2, dw3);

                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(&dB_M[idx_left], dbetaD * w1 + betaD * dw1);
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(&dB_I[idx_left], dbetaD * w2 + betaD * dw2);
                atomicAdd(&B_D[idx_left], betaD * w3);
                atomicAdd(&dB_D[idx_left], dbetaD * w3 + betaD * dw3);
            }
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT (dP/dtheta)
// ============================================================================

// Initialize boundary U values for affine leading-gap base cases.
__global__ void nw_affine_param_grad_init_boundary_U_kernel(
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    int param_type
) {
    if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
        return;
    }

    int b = blockIdx.x;
    if (b >= B) return;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;

    float* U_M = U + (size_t)b * stride_all;
    float* U_I = U_M + stride_state;
    float* U_D = U_I + stride_state;

    int boundary_len = max(L1, L2);
    for (int t = threadIdx.x + 1; t <= boundary_len; t += blockDim.x) {
        if (t <= L1) {
            size_t idx = (size_t)t * stride_row;
            U_I[idx] = (param_type == NW_AFFINE_PARAM_GAP_OPEN)
                ? 1.0f
                : static_cast<float>(t - 1);
        }
        if (t <= L2) {
            size_t idx = (size_t)t;
            U_D[idx] = (param_type == NW_AFFINE_PARAM_GAP_OPEN)
                ? 1.0f
                : static_cast<float>(t - 1);
        }
    }
}

// Forward U pass: compute U = d(alpha)/d(theta)
__global__ void nw_affine_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    float* U_M = U + (size_t)b * stride_all;
    float* U_I = U_M + stride_state;
    float* U_D = U_I + stride_state;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) {
            size_t idx = (size_t)i * stride_row + (size_t)j;
            U_M[idx] = 0.0f;
            U_I[idx] = 0.0f;
            U_D[idx] = 0.0f;
            continue;
        }

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);

        // M state U
        {
            float m1 = A_M[idx_diag];
            float m2 = A_I[idx_diag];
            float m3 = A_D[idx_diag];
            float w1, w2, w3;

            if (softmax3_with_lse(m1, m2, m3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_diag];
                float u2 = U_I[idx_diag];
                float u3 = U_D[idx_diag];

                float dU_M = w1 * u1 + w2 * u2 + w3 * u3;

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float alpha_M = A_M[idx];
                    float E_v = w1 * m1 + w2 * m2 + w3 * m3;
                    size_t score_idx = (size_t)b * stride_scores
                        + (size_t)(i - 1) * (size_t)max_L2
                        + (size_t)(j - 1);
                    dU_M += (alpha_M - (E_v + scores[score_idx])) / T;
                }

                U_M[idx] = dU_M;
            } else {
                U_M[idx] = 0.0f;
            }
        }

        // I state U
        {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float i3 = A_D[idx_up] + gap_open;
            float w1, w2, w3;

            if (softmax3_with_lse(i1, i2, i3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_up];
                float u2 = U_I[idx_up];
                float u3 = U_D[idx_up];

                float direct1 = 0.0f, direct2 = 0.0f, direct3 = 0.0f;
                if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                    direct1 = 1.0f;
                    direct3 = 1.0f;
                } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                    direct2 = 1.0f;
                }

                float dU_I = w1 * (u1 + direct1) + w2 * (u2 + direct2) + w3 * (u3 + direct3);

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float alpha_I = A_I[idx];
                    float E_v = w1 * i1 + w2 * i2 + w3 * i3;
                    dU_I += (alpha_I - E_v) / T;
                }

                U_I[idx] = dU_I;
            } else {
                U_I[idx] = 0.0f;
            }
        }

        // D state U
        {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;

            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_left];
                float u2 = U_I[idx_left];
                float u3 = U_D[idx_left];

                float direct1 = 0.0f, direct2 = 0.0f, direct3 = 0.0f;
                if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                    direct1 = 1.0f;
                    direct2 = 1.0f;
                } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                    direct3 = 1.0f;
                }

                float dU_D = w1 * (u1 + direct1) + w2 * (u2 + direct2) + w3 * (u3 + direct3);

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float alpha_D = A_D[idx];
                    float E_v = w1 * d1 + w2 * d2 + w3 * d3;
                    dU_D += (alpha_D - E_v) / T;
                }

                U_D[idx] = dU_D;
            } else {
                U_D[idx] = 0.0f;
            }
        }
    }
}

// Initialize W at the terminal cell using the terminal softmax Jacobian.
__global__ void nw_affine_param_grad_init_terminal_W_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ U,
    float* __restrict__ W,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float T,
    int param_type
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_row = (size_t)max_L2 + 1;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    const float* U_M = U + (size_t)b * stride_all;
    const float* U_I = U_M + stride_state;
    const float* U_D = U_I + stride_state;

    float* W_M = W + (size_t)b * stride_all;
    float* W_I = W_M + stride_state;
    float* W_D = W_I + stride_state;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];
    size_t final_idx = (size_t)L1 * stride_row + (size_t)L2;

    float m_val = A_M[final_idx];
    float i_val = A_I[final_idx];
    float d_val = A_D[final_idx];
    float w_m, w_i, w_d;
    softmax3_with_lse(m_val, i_val, d_val, T, w_m, w_i, w_d);

    float dw_m, dw_i, dw_d;
    softmax3_tangent(
        w_m, w_i, w_d,
        U_M[final_idx], U_I[final_idx], U_D[final_idx],
        T, dw_m, dw_i, dw_d
    );

    if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
        float E_v = w_m * m_val + w_i * i_val + w_d * d_val;
        float inv_T2 = 1.0f / (T * T);
        dw_m += w_m * (E_v - m_val) * inv_T2;
        dw_i += w_i * (E_v - i_val) * inv_T2;
        dw_d += w_d * (E_v - d_val) * inv_T2;
    }

    W_M[final_idx] = dw_m;
    W_I[final_idx] = dw_i;
    W_D[final_idx] = dw_d;
}

// Backward W pass
__global__ void nw_affine_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ U,
    const float* __restrict__ dS_dtheta,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dtheta,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t stride_scores = (size_t)max_L1 * (size_t)max_L2;
    size_t stride_row = (size_t)max_L2 + 1;
    size_t score_row = (size_t)max_L2;

    const float* A_M = alpha + (size_t)b * stride_all;
    const float* A_I = A_M + stride_state;
    const float* A_D = A_I + stride_state;

    const float* U_M = U + (size_t)b * stride_all;
    const float* U_I = U_M + stride_state;
    const float* U_D = U_I + stride_state;

    float* B_M = beta + (size_t)b * stride_all;
    float* B_I = B_M + stride_state;
    float* B_D = B_I + stride_state;

    float* W_M = W + (size_t)b * stride_all;
    float* W_I = W_M + stride_state;
    float* W_D = W_I + stride_state;

    float* dP = dP_dtheta + (size_t)b * stride_scores;

    int i_start = max(1, k_diag - max_L2);
    int i_end = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        if (i > L1 || j > L2) continue;

        size_t idx = (size_t)i * stride_row + (size_t)j;
        size_t idx_diag = (size_t)(i - 1) * stride_row + (size_t)(j - 1);
        size_t idx_up = (size_t)(i - 1) * stride_row + (size_t)j;
        size_t idx_left = (size_t)i * stride_row + (size_t)(j - 1);
        size_t score_idx = (size_t)(i - 1) * score_row + (size_t)(j - 1);

        float betaM = B_M[idx];
        float betaI = B_I[idx];
        float betaD = B_D[idx];
        float wM = W_M[idx];
        float wI = W_I[idx];
        float wD = W_D[idx];

        // M state
        if (betaM > 1e-20f || fabsf(wM) > 1e-20f) {
            float m1 = A_M[idx_diag];
            float m2 = A_I[idx_diag];
            float m3 = A_D[idx_diag];
            float w1, w2, w3;

            if (softmax3_with_lse(m1, m2, m3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_diag];
                float u2 = U_I[idx_diag];
                float u3 = U_D[idx_diag];

                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, u1, u2, u3, T, dw1, dw2, dw3);

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float E_v = w1 * m1 + w2 * m2 + w3 * m3;
                    float inv_T2 = 1.0f / (T * T);
                    dw1 += w1 * (E_v - m1) * inv_T2;
                    dw2 += w2 * (E_v - m2) * inv_T2;
                    dw3 += w3 * (E_v - m3) * inv_T2;
                }

                if (m1 > NINF) {
                    atomicAdd(&B_M[idx_diag], betaM * w1);
                    atomicAdd(&W_M[idx_diag], wM * w1 + betaM * dw1);
                }
                if (m2 > NINF) {
                    atomicAdd(&B_I[idx_diag], betaM * w2);
                    atomicAdd(&W_I[idx_diag], wM * w2 + betaM * dw2);
                }
                if (m3 > NINF) {
                    atomicAdd(&B_D[idx_diag], betaM * w3);
                    atomicAdd(&W_D[idx_diag], wM * w3 + betaM * dw3);
                }

                atomicAdd(&dP[score_idx], wM);
            }
        }

        // I state
        if (betaI > 1e-20f || fabsf(wI) > 1e-20f) {
            float i1 = A_M[idx_up] + gap_open;
            float i2 = A_I[idx_up] + gap_ext;
            float i3 = A_D[idx_up] + gap_open;
            float w1, w2, w3;

            if (softmax3_with_lse(i1, i2, i3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_up];
                float u2 = U_I[idx_up];
                float u3 = U_D[idx_up];

                if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                    u1 += 1.0f;
                    u3 += 1.0f;
                } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                    u2 += 1.0f;
                }

                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, u1, u2, u3, T, dw1, dw2, dw3);

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float E_v = w1 * i1 + w2 * i2 + w3 * i3;
                    float inv_T2 = 1.0f / (T * T);
                    dw1 += w1 * (E_v - i1) * inv_T2;
                    dw2 += w2 * (E_v - i2) * inv_T2;
                    dw3 += w3 * (E_v - i3) * inv_T2;
                }

                atomicAdd(&B_M[idx_up], betaI * w1);
                atomicAdd(&W_M[idx_up], wI * w1 + betaI * dw1);
                atomicAdd(&B_I[idx_up], betaI * w2);
                atomicAdd(&W_I[idx_up], wI * w2 + betaI * dw2);
                atomicAdd(&B_D[idx_up], betaI * w3);
                atomicAdd(&W_D[idx_up], wI * w3 + betaI * dw3);
            }
        }

        // D state
        if (betaD > 1e-20f || fabsf(wD) > 1e-20f) {
            float d1 = A_M[idx_left] + gap_open;
            float d2 = A_I[idx_left] + gap_open;
            float d3 = A_D[idx_left] + gap_ext;
            float w1, w2, w3;

            if (softmax3_with_lse(d1, d2, d3, T, w1, w2, w3) > NINF) {
                float u1 = U_M[idx_left];
                float u2 = U_I[idx_left];
                float u3 = U_D[idx_left];

                if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                    u1 += 1.0f;
                    u2 += 1.0f;
                } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                    u3 += 1.0f;
                }

                float dw1, dw2, dw3;
                softmax3_tangent(w1, w2, w3, u1, u2, u3, T, dw1, dw2, dw3);

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float E_v = w1 * d1 + w2 * d2 + w3 * d3;
                    float inv_T2 = 1.0f / (T * T);
                    dw1 += w1 * (E_v - d1) * inv_T2;
                    dw2 += w2 * (E_v - d2) * inv_T2;
                    dw3 += w3 * (E_v - d3) * inv_T2;
                }

                atomicAdd(&B_M[idx_left], betaD * w1);
                atomicAdd(&W_M[idx_left], wD * w1 + betaD * dw1);
                atomicAdd(&B_I[idx_left], betaD * w2);
                atomicAdd(&W_I[idx_left], wD * w2 + betaD * dw2);
                atomicAdd(&B_D[idx_left], betaD * w3);
                atomicAdd(&W_D[idx_left], wD * w3 + betaD * dw3);
            }
        }
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

extern "C" void nw_affine_forward(
    const float* d_scores, float* d_alpha, float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t total_alpha = (size_t)B * stride_all;

    size_t blocks_init = (total_alpha + threads - 1) / threads;
    nw_affine_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_lengths, B, max_L1, max_L2, gap_open, gap_ext
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        nw_affine_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    nw_affine_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_score, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_affine_backward(
    const float* d_alpha, const float* d_scores, const float* d_score,
    float* d_beta, float* d_posteriors,
    float* d_grad_open, float* d_grad_ext, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t total_alpha = (size_t)B * stride_all;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_open, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_ext, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    size_t blocks_init = (total_alpha + threads - 1) / threads;
    nw_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        nw_affine_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_open, d_grad_ext,
            d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Boundary gap-gradient: B_I at (i,0) and B_D at (0,j)
    nw_affine_boundary_gap_grad_kernel<<<B, threads, 0, stream>>>(
        d_beta, d_grad_open, d_grad_ext, d_lengths, B, max_L1, max_L2
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nw_affine_grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_grad_open, d_grad_ext, d_score, d_grad_T,
        d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_affine_hvp(
    const float* d_alpha, const float* d_scores, const float* d_score,
    const float* d_V, float* d_d_alpha, float* d_d_score,
    float* d_beta, float* d_d_beta, float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t total_alpha = (size_t)B * stride_all;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    for (int k = 2; k <= max_diag; ++k) {
        nw_affine_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    nw_affine_hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_d_alpha, d_d_score, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t blocks_init = (total_alpha + threads - 1) / threads;
    nw_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nw_affine_init_d_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_d_alpha, d_d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        nw_affine_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores,
            d_lengths, B, max_L1, max_L2, gap_open, gap_ext, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void nw_affine_param_grad(
    const float* d_alpha, const float* d_scores, const float* d_score,
    const float* d_dS_dtheta,
    float* d_U, float* d_beta, float* d_W, float* d_dP_dtheta,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int param_type
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t stride_state = ((size_t)max_L1 + 1) * ((size_t)max_L2 + 1);
    size_t stride_all = 3 * stride_state;
    size_t total_alpha = (size_t)B * stride_all;
    size_t score_elems = (size_t)B * (size_t)max_L1 * (size_t)max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dtheta, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    nw_affine_param_grad_init_boundary_U_kernel<<<B, threads, 0, stream>>>(
        d_U, d_lengths, B, max_L1, max_L2, param_type
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = 2; k <= max_diag; ++k) {
        nw_affine_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    size_t blocks_init = (total_alpha + threads - 1) / threads;
    nw_affine_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_terminal = (B + threads - 1) / threads;
    nw_affine_param_grad_init_terminal_W_kernel<<<blocks_terminal, threads, 0, stream>>>(
        d_alpha, d_U, d_W, d_lengths, B, max_L1, max_L2, T, param_type
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 2; --k) {
        nw_affine_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_score, d_U, d_dS_dtheta,
            d_beta, d_W, d_dP_dtheta, d_lengths,
            B, max_L1, max_L2, gap_open, gap_ext, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
