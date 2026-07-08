// SPDX-License-Identifier: Apache-2.0
// soft_sw_regular.cu
//
// Direct (simple) Soft Smith-Waterman in CUDA
// Single-state DP with linear gap penalty for local alignment
//
// Operations:
//   - Forward:  Compute partition function S = logsumexp over all alignments
//   - Backward: Compute all gradients (dS/dscores, dS/dgap, dS/dT)
//   - HVP:      Hessian-vector product d^2S/dscores^2 * V (forward-mode)
//
// Shapes:
//   scores:      [B, L1, L2]       - similarity scores
//   alpha:       [B, (L1+1)*(L2+1)] - DP table (row-major, 1-indexed logic)
//   partition:   [B]               - partition function values
//   posteriors:  [B, L1, L2]       - alignment marginals (dS/dscores)
//   grad_gap:    [B]               - expected number of gap steps
//   grad_T:      [B]               - temperature gradient

#include <cuda_runtime.h>
#include <math.h>
#include <limits>

// Shared utilities
#include "common/cuda_utils.h"
#include "common/numerics.cuh"
#include "common/reduce.cuh"

using namespace d2p::common;

// Parameter type enum for param gradient kernels
enum ParamType {
    PARAM_GAP = 0,
    PARAM_TEMPERATURE = 1
};

constexpr int SW_WARP_SIZE = 32;
constexpr int SW_WARPS_PER_BLOCK = 8;

namespace {

int checked_grid_blocks(size_t elements, int threads, const char* kernel_name) {
    size_t blocks = (elements + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
    TORCH_CHECK(
        blocks <= static_cast<size_t>(std::numeric_limits<int>::max()),
        kernel_name, ": launch grid is too large: ", blocks, " blocks"
    );
    return static_cast<int>(blocks);
}

}  // namespace

// ============================================================================
// FORWARD PASS
// ============================================================================

// Initialize alpha: alpha[0,0] = 0, all others = -inf
__global__ void sw_regular_init_alpha_kernel(
    float* __restrict__ alpha,
    int B, int L1, int L2
) {
    size_t cols = static_cast<size_t>(L2) + 1;
    size_t stride = (static_cast<size_t>(L1) + 1) * cols;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b   = idx / stride;
    size_t rem = idx - b * stride;
    int i   = static_cast<int>(rem / cols);
    int j   = static_cast<int>(rem % cols);

    alpha[idx] = (i == 0 && j == 0) ? 0.0f : NINF;
}

// Forward DP for one anti-diagonal k = i + j
// Four transitions: align (diagonal+score), gap_up, gap_left, sky (fresh start)
__global__ void sw_regular_forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,    // max padded dimensions
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t row_stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (static_cast<size_t>(max_L1) + 1) * row_stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

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
            size_t idx = static_cast<size_t>(i) * row_stride + j;
            a[idx] = NINF;
            continue;
        }

        size_t idx       = static_cast<size_t>(i) * row_stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * row_stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * row_stride + j;
        size_t idx_left  = static_cast<size_t>(i) * row_stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float score = s[score_idx];

        // Four logits for soft-max
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;  // local alignment: can start fresh

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));

        if (max_v <= NINF) {
            a[idx] = NINF;
        } else {
            float w_align = safe_exp((v_align - max_v) / T);
            float w_up    = safe_exp((v_up    - max_v) / T);
            float w_left  = safe_exp((v_left  - max_v) / T);
            float w_sky   = safe_exp((v_sky   - max_v) / T);
            float sum_w   = w_align + w_up + w_left + w_sky;
            a[idx] = max_v + T * logf(sum_w);
        }
    }
}

// Partition function: S = T * logsumexp(alpha / T) over valid cells only
__global__ void sw_regular_partition_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ partition,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t total_stride = (static_cast<size_t>(max_L1) + 1) * stride;
    const float* a = alpha + (size_t)b * total_stride;

    // Per-batch actual lengths
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Find max over valid region [0, L1] x [0, L2] (includes boundary alpha[0,0]=0)
    float local_max = NINF;
    size_t valid_elems = (static_cast<size_t>(L1) + 1) * (L2 + 1);
    size_t valid_cols = static_cast<size_t>(L2) + 1;
    for (size_t t = threadIdx.x; t < valid_elems; t += blockDim.x) {
        int i = static_cast<int>(t / valid_cols);
        int j = static_cast<int>(t % valid_cols);
        if (i <= L1 && j <= L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            local_max = fmaxf(local_max, a[idx]);
        }
    }
    float max_val = block_reduce_max(local_max);

    __shared__ float sh_max;
    if (threadIdx.x == 0) sh_max = max_val;
    __syncthreads();
    max_val = sh_max;

    if (max_val <= NINF) {
        if (threadIdx.x == 0) partition[b] = NINF;
        return;
    }

    // Sum exp over valid region only
    float local_sum = 0.0f;
    for (size_t t = threadIdx.x; t < valid_elems; t += blockDim.x) {
        int i = static_cast<int>(t / valid_cols);
        int j = static_cast<int>(t % valid_cols);
        if (i <= L1 && j <= L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            local_sum += safe_exp((a[idx] - max_val) / T);
        }
    }
    float sum_exp = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        partition[b] = max_val + T * logf(sum_exp);
    }
}

// Warp-parallel forward kernel: one warp owns one batch element and processes
// the full anti-diagonal schedule with implicit warp synchronization.
__global__ void sw_regular_forward_warp_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    float* __restrict__ partition,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int lane = threadIdx.x & (SW_WARP_SIZE - 1);
    int warp_in_block = threadIdx.x / SW_WARP_SIZE;
    int b = blockIdx.x * SW_WARPS_PER_BLOCK + warp_in_block;
    if (b >= B) return;

    size_t row_stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * row_stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    if (lane == 0) {
        a[0] = 0.0f;
        for (int i = 1; i <= max_L1; ++i) {
            a[static_cast<size_t>(i) * row_stride] = NINF;
        }
    }
    for (int j = lane + 1; j <= max_L2; j += SW_WARP_SIZE) {
        a[j] = NINF;
    }
    __syncwarp();

    int max_diag = max_L1 + max_L2;
    for (int k_diag = 2; k_diag <= max_diag; ++k_diag) {
        int i_start = max(1, k_diag - max_L2);
        int i_end = min(max_L1, k_diag - 1);
        int diag_len = i_end - i_start + 1;
        if (diag_len > 0) {
            for (int t = lane; t < diag_len; t += SW_WARP_SIZE) {
                int i = i_start + t;
                int j = k_diag - i;
                if (j < 1 || j > max_L2) continue;

                size_t idx = static_cast<size_t>(i) * row_stride + j;
                if (i > L1 || j > L2) {
                    a[idx] = NINF;
                    continue;
                }

                size_t idx_diag = static_cast<size_t>(i - 1) * row_stride + (j - 1);
                size_t idx_up = static_cast<size_t>(i - 1) * row_stride + j;
                size_t idx_left = static_cast<size_t>(i) * row_stride + (j - 1);
                size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

                float score = s[score_idx];
                float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                float v_sky = score;

                a[idx] = logsumexp4(v_align, v_up, v_left, v_sky, T);
            }
        }
        __syncwarp();
    }

    size_t valid_elems = (static_cast<size_t>(L1) + 1) * (L2 + 1);
    size_t valid_cols = static_cast<size_t>(L2) + 1;
    float local_max = NINF;
    for (size_t t = lane; t < valid_elems; t += SW_WARP_SIZE) {
        int i = static_cast<int>(t / valid_cols);
        int j = static_cast<int>(t % valid_cols);
        size_t idx = static_cast<size_t>(i) * row_stride + j;
        local_max = fmaxf(local_max, a[idx]);
    }
    float max_val = warp_reduce_max(local_max);
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    if (max_val <= NINF) {
        if (lane == 0) {
            partition[b] = NINF;
        }
        return;
    }

    float local_sum = 0.0f;
    for (size_t t = lane; t < valid_elems; t += SW_WARP_SIZE) {
        int i = static_cast<int>(t / valid_cols);
        int j = static_cast<int>(t % valid_cols);
        size_t idx = static_cast<size_t>(i) * row_stride + j;
        local_sum += safe_exp((a[idx] - max_val) / T);
    }
    float sum_exp = warp_reduce_sum(local_sum);
    if (lane == 0) {
        partition[b] = max_val + T * logf(sum_exp);
    }
}

// ============================================================================
// BACKWARD PASS - All Gradients
// ============================================================================

__device__ __forceinline__ void sw_compute_weights(
    float v_align, float v_up, float v_left, float v_sky, float T,
    float& w_align, float& w_up, float& w_left, float& w_sky
) {
    float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));
    if (max_v <= NINF + 1.0f) {
        w_align = w_up = w_left = w_sky = 0.0f;
        return;
    }

    float w_a = safe_exp((v_align - max_v) / T);
    float w_u = safe_exp((v_up - max_v) / T);
    float w_l = safe_exp((v_left - max_v) / T);
    float w_s = safe_exp((v_sky - max_v) / T);
    float inv_sum = 1.0f / (w_a + w_u + w_l + w_s);
    w_align = w_a * inv_sum;
    w_up = w_u * inv_sum;
    w_left = w_l * inv_sum;
    w_sky = w_s * inv_sum;
}

// Initialize beta = dS/dalpha = exp((alpha - S) / T)
// Only initializes valid cells; masked cells get beta=0
__global__ void sw_regular_init_beta_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ partition,
    float* __restrict__ beta,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * alpha_stride;
    if (idx >= total) return;

    size_t b = idx / alpha_stride;
    size_t rem = idx - b * alpha_stride;
    int i = static_cast<int>(rem / stride);
    int j = static_cast<int>(rem % stride);

    // Per-batch actual lengths
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // Masked cells get beta=0
    if (i > L1 || j > L2) {
        beta[idx] = 0.0f;
        return;
    }

    float a = alpha[idx];
    float S = partition[b];

    // Fix #10: Posterior clamping - clamp log-posterior to <= 0 to ensure beta <= 1
    // Numerical errors can cause alpha > S slightly, resulting in posteriors > 1
    if (a > NINF) {
        float log_post = (a - S) / T;
        log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
        beta[idx] = safe_exp(log_post);
    } else {
        beta[idx] = 0.0f;
    }
}

// Backward DP: propagate beta and accumulate gradients
// Computes: posteriors (dS/dscores), grad_gap (dS/dgap)
__global__ void sw_regular_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_gap,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    // Per-batch actual lengths
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    float* g_beta  = beta + (size_t)b * stride_alpha;
    float* g_scores = posteriors + (size_t)b * stride_scores;

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    float local_gap_grad = 0.0f;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float b_curr = g_beta[idx];
        if (b_curr <= 1e-20f) continue;

        float score = s[score_idx];

        // Recompute logits and weights
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));
        if (max_v <= NINF) continue;

        float w_align = safe_exp((v_align - max_v) / T);
        float w_up    = safe_exp((v_up    - max_v) / T);
        float w_left  = safe_exp((v_left  - max_v) / T);
        float w_sky   = safe_exp((v_sky   - max_v) / T);
        float sum_w = w_align + w_up + w_left + w_sky;
        float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

        w_align *= inv_sum;
        w_up    *= inv_sum;
        w_left  *= inv_sum;
        w_sky   *= inv_sum;

        // Propagate beta to predecessors
        if (v_align > NINF) atomicAdd(&g_beta[idx_diag], b_curr * w_align);
        if (v_up    > NINF) atomicAdd(&g_beta[idx_up],   b_curr * w_up);
        if (v_left  > NINF) atomicAdd(&g_beta[idx_left], b_curr * w_left);

        // Accumulate score gradient (align and sky both use score)
        float score_grad = b_curr * (w_align + w_sky);
        atomicAdd(&g_scores[score_idx], score_grad);

        // Accumulate gap gradient
        local_gap_grad += b_curr * (w_up + w_left);
    }

    // Block-reduce gap gradient
    float block_gap = block_reduce_sum(local_gap_grad);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_gap[b], block_gap);
    }
}

__global__ void sw_regular_backward_warp_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    float* __restrict__ grad_gap,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int lane = threadIdx.x & (SW_WARP_SIZE - 1);
    int warp_in_block = threadIdx.x / SW_WARP_SIZE;
    int b = blockIdx.x * SW_WARPS_PER_BLOCK + warp_in_block;
    if (b >= B) return;

    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a = alpha + (size_t)b * stride_alpha;
    const float* s = scores + (size_t)b * stride_scores;
    float* g_beta = beta + (size_t)b * stride_alpha;
    float* g_scores = posteriors + (size_t)b * stride_scores;

    float local_gap_grad = 0.0f;
    int max_diag = max_L1 + max_L2;
    for (int k_diag = max_diag; k_diag >= 2; --k_diag) {
        int i_start = max(1, k_diag - max_L2);
        int i_end = min(max_L1, k_diag - 1);
        int diag_len = i_end - i_start + 1;
        if (diag_len > 0) {
            for (int t = lane; t < diag_len; t += SW_WARP_SIZE) {
                int i = i_start + t;
                int j = k_diag - i;
                if (j < 1 || j > max_L2) continue;
                if (i > L1 || j > L2) continue;

                size_t idx = static_cast<size_t>(i) * stride + j;
                size_t idx_diag = static_cast<size_t>(i - 1) * stride + (j - 1);
                size_t idx_up = static_cast<size_t>(i - 1) * stride + j;
                size_t idx_left = static_cast<size_t>(i) * stride + (j - 1);
                size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

                float b_curr = g_beta[idx];
                if (b_curr <= 1.0e-20f) continue;

                float score = s[score_idx];
                float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                float v_sky = score;

                float w_align, w_up, w_left, w_sky;
                sw_compute_weights(v_align, v_up, v_left, v_sky, T, w_align, w_up, w_left, w_sky);

                if (v_align > NINF) atomicAdd(&g_beta[idx_diag], b_curr * w_align);
                if (v_up > NINF) atomicAdd(&g_beta[idx_up], b_curr * w_up);
                if (v_left > NINF) atomicAdd(&g_beta[idx_left], b_curr * w_left);

                atomicAdd(&g_scores[score_idx], b_curr * (w_align + w_sky));
                local_gap_grad += b_curr * (w_up + w_left);
            }
        }
        __syncwarp();
    }

    float warp_gap = warp_reduce_sum(local_gap_grad);
    if (lane == 0) {
        atomicAdd(&grad_gap[b], warp_gap);
    }
}

// Compute dS/dT = (S - E[total_score]) / T
// Note: posteriors are already zero for masked cells, so sum over full array is correct
__global__ void sw_regular_grad_T_kernel(
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
    // Masked cells have posteriors=0, so they don't contribute
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
// HVP (Hessian-Vector Product) - Forward Mode
// ============================================================================
// Given V = upstream gradient w.r.t. posteriors, compute H*V where H = d^2S/dscores^2
// Uses forward-mode AD: propagate tangent vectors through the gradient computation

// Forward-mode HVP: propagate tangents through forward DP
__global__ void sw_regular_hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,           // tangent input [B,L1,L2]
    float* __restrict__ d_alpha,           // tangent of alpha [B,(L1+1)*(L2+1)]
    const int* __restrict__ lengths,       // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    // Per-batch actual lengths
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

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            da[idx] = 0.0f;
            continue;
        }

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float score = s[score_idx];
        float V_s   = v[score_idx];

        // Recompute logits
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));
        if (max_v <= NINF) {
            da[idx] = 0.0f;
            continue;
        }

        float w_align = safe_exp((v_align - max_v) / T);
        float w_up    = safe_exp((v_up    - max_v) / T);
        float w_left  = safe_exp((v_left  - max_v) / T);
        float w_sky   = safe_exp((v_sky   - max_v) / T);
        float sum_w = w_align + w_up + w_left + w_sky;
        float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

        w_align *= inv_sum;
        w_up    *= inv_sum;
        w_left  *= inv_sum;
        w_sky   *= inv_sum;

        // Tangent of logits
        float dv_align = (i > 1 && j > 1) ? da[idx_diag] + V_s : 0.0f;
        float dv_up    = (i > 1)          ? da[idx_up]        : 0.0f;
        float dv_left  = (j > 1)          ? da[idx_left]      : 0.0f;
        float dv_sky   = V_s;

        // Tangent of soft-max output: d_alpha = sum_k w_k * dv_k
        da[idx] = w_align * dv_align + w_up * dv_up + w_left * dv_left + w_sky * dv_sky;
    }
}

// Compute tangent of partition function (only over valid cells)
__global__ void sw_regular_hvp_partition_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ d_alpha,
    const float* __restrict__ partition,
    float* __restrict__ d_partition,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2, float T
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t total_stride = (static_cast<size_t>(max_L1) + 1) * stride;
    const float* a  = alpha + (size_t)b * total_stride;
    const float* da = d_alpha + (size_t)b * total_stride;
    float S = partition[b];

    // Per-batch actual lengths
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    // d_S = sum_ij beta[i,j] * d_alpha[i,j] where beta = exp((alpha - S)/T)
    // Only sum over valid cells [0, L1] x [0, L2]
    float local_sum = 0.0f;
    size_t valid_elems = (static_cast<size_t>(L1) + 1) * (L2 + 1);
    size_t valid_cols = static_cast<size_t>(L2) + 1;
    for (size_t t = threadIdx.x; t < valid_elems; t += blockDim.x) {
        int i = static_cast<int>(t / valid_cols);
        int j = static_cast<int>(t % valid_cols);
        if (i <= L1 && j <= L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            if (a[idx] > NINF) {
                float beta = safe_exp((a[idx] - S) / T);
                local_sum += beta * da[idx];
            }
        }
    }
    float sum = block_reduce_sum(local_sum);

    if (threadIdx.x == 0) {
        d_partition[b] = sum;
    }
}

// Backward pass of forward-mode HVP: compute HVP output
__global__ void sw_regular_hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ d_alpha,
    const float* __restrict__ d_partition,
    const float* __restrict__ V,
    float* __restrict__ beta,     // beta accumulation buffer (like regular backward)
    float* __restrict__ d_beta,   // d_beta (tangent) accumulation buffer
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,  // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

    // Per-batch actual lengths
    int L1 = lengths[b * 2];
    int L2 = lengths[b * 2 + 1];

    const float* a  = alpha + (size_t)b * stride_alpha;
    const float* s  = scores + (size_t)b * stride_scores;
    const float* da = d_alpha + (size_t)b * stride_alpha;
    const float* v  = V + (size_t)b * stride_scores;
    float* g_beta   = beta + (size_t)b * stride_alpha;    // accumulated beta (like regular backward)
    float* db       = d_beta + (size_t)b * stride_alpha;  // d_beta accumulation
    float* H        = H_scores + (size_t)b * stride_scores;

    float S  = partition[b];
    float dS = d_partition[b];

    int i_start  = max(1, k_diag - max_L2);
    int i_end    = min(max_L1, k_diag - 1);
    int diag_len = i_end - i_start + 1;
    if (diag_len <= 0) return;

    for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
        int i = i_start + t;
        int j = k_diag - i;
        if (j < 1 || j > max_L2) continue;

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        // Read ACCUMULATED beta from buffer (like regular backward)
        float beta_curr = g_beta[idx];

        // Tangent of beta: d_beta_init = beta_init * (d_alpha - d_S) / T
        // where beta_init = exp((alpha - S)/T)
        // Fix #10: Apply posterior clamping
        float beta_init = 0.0f;
        if (a[idx] > NINF) {
            float log_post = (a[idx] - S) / T;
            log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
            beta_init = safe_exp(log_post);
        }

        // Guard against numerical overflow: clamp tangent differences
        float da_diff = da[idx] - dS;
        da_diff = fminf(fmaxf(da_diff, -1e6f), 1e6f);
        float db_init = beta_init * da_diff / T;

        // Total d_beta = accumulated from successors + initial tangent
        float db_accum = db[idx];
        // Guard against NaN/Inf in accumulated d_beta
        if (isnan(db_accum) || isinf(db_accum)) db_accum = 0.0f;
        float db_curr = db_accum + db_init;

        // Clamp db_curr to prevent overflow in subsequent computations
        db_curr = fminf(fmaxf(db_curr, -1e6f), 1e6f);

        // Skip if both beta and d_beta are negligible
        if (beta_curr <= 1e-20f && fabsf(db_curr) <= 1e-20f) continue;

        float score = s[score_idx];
        float V_s   = v[score_idx];

        // Recompute weights
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));
        if (max_v <= NINF) continue;

        float w_align = safe_exp((v_align - max_v) / T);
        float w_up    = safe_exp((v_up    - max_v) / T);
        float w_left  = safe_exp((v_left  - max_v) / T);
        float w_sky   = safe_exp((v_sky   - max_v) / T);
        float sum_w = w_align + w_up + w_left + w_sky;
        float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

        w_align *= inv_sum;
        w_up    *= inv_sum;
        w_left  *= inv_sum;
        w_sky   *= inv_sum;

        // Compute tangent of logits (dv) for softmax Jacobian
        // d_w_k = w_k * (d_logit_k - sum_j w_j * d_logit_j) / T
        float dv_align = (i > 1 && j > 1) ? da[idx_diag] + V_s : 0.0f;
        float dv_up    = (i > 1)          ? da[idx_up]        : 0.0f;
        float dv_left  = (j > 1)          ? da[idx_left]      : 0.0f;
        float dv_sky   = V_s;

        float sum_w_dv = w_align * dv_align + w_up * dv_up + w_left * dv_left + w_sky * dv_sky;

        // Compute tangent of ALL weights (needed for d_beta propagation)
        // Guard against overflow by clamping the differences
        float dv_diff_align = fminf(fmaxf(dv_align - sum_w_dv, -1e6f), 1e6f);
        float dv_diff_up    = fminf(fmaxf(dv_up    - sum_w_dv, -1e6f), 1e6f);
        float dv_diff_left  = fminf(fmaxf(dv_left  - sum_w_dv, -1e6f), 1e6f);
        float dv_diff_sky   = fminf(fmaxf(dv_sky   - sum_w_dv, -1e6f), 1e6f);

        float dw_align = w_align * dv_diff_align / T;
        float dw_up    = w_up    * dv_diff_up    / T;
        float dw_left  = w_left  * dv_diff_left  / T;
        float dw_sky   = w_sky   * dv_diff_sky   / T;

        // Propagate BOTH beta and d_beta to predecessors (like regular backward)
        // Original backward: beta[pred] += beta_curr * w[k]
        // HVP derivative:    d_beta[pred] += db_curr * w[k] + beta_curr * dw[k]
        if (v_align > NINF) {
            atomicAdd(&g_beta[idx_diag], beta_curr * w_align);
            float db_add = db_curr * w_align + beta_curr * dw_align;
            if (!isnan(db_add) && !isinf(db_add)) {
                atomicAdd(&db[idx_diag], db_add);
            }
        }
        if (v_up > NINF) {
            atomicAdd(&g_beta[idx_up], beta_curr * w_up);
            float db_add = db_curr * w_up + beta_curr * dw_up;
            if (!isnan(db_add) && !isinf(db_add)) {
                atomicAdd(&db[idx_up], db_add);
            }
        }
        if (v_left > NINF) {
            atomicAdd(&g_beta[idx_left], beta_curr * w_left);
            float db_add = db_curr * w_left + beta_curr * dw_left;
            if (!isnan(db_add) && !isinf(db_add)) {
                atomicAdd(&db[idx_left], db_add);
            }
        }

        // Contribution to H_scores from d_beta propagation
        float H_contrib = db_curr * (w_align + w_sky);

        // Additional term from d_weights contribution to score gradient
        // From: posteriors[i,j] = beta_curr * (w_align + w_sky)
        // Derivative: d_posteriors += db_curr * (w_align + w_sky) + beta_curr * (dw_align + dw_sky)
        float beta_term = beta_curr * (dw_align + dw_sky);

        // Guard against NaN/Inf before atomic add
        float H_val = H_contrib + beta_term;
        if (!isnan(H_val) && !isinf(H_val)) {
            atomicAdd(&H[score_idx], H_val);
        }
    }
}

// ============================================================================
// PARAMETER GRADIENT KERNELS (d^2S/dscores/dtheta)
// ============================================================================

// Forward U pass: compute U = dalpha/dtheta for parameter theta (gap or temperature)
// Propagates tangent through forward DP using chain rule
__global__ void sw_regular_param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag,
    int param_type  // 0=gap, 1=temperature
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

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

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) {
            size_t idx = static_cast<size_t>(i) * stride + j;
            u[idx] = 0.0f;
            continue;
        }

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        float score = s[score_idx];

        // Recompute logits (must match sw_regular_forward_diag_kernel exactly!)
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));

        if (max_v <= NINF) {
            u[idx] = 0.0f;
            continue;
        }

        // Compute softmax weights
        float w_align = safe_exp((v_align - max_v) / T);
        float w_up    = safe_exp((v_up    - max_v) / T);
        float w_left  = safe_exp((v_left  - max_v) / T);
        float w_sky   = safe_exp((v_sky   - max_v) / T);
        float sum_w = w_align + w_up + w_left + w_sky;
        float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

        w_align *= inv_sum;
        w_up    *= inv_sum;
        w_left  *= inv_sum;
        w_sky   *= inv_sum;

        // Get predecessor U values
        float u_align = (i > 1 && j > 1) ? u[idx_diag] : 0.0f;
        float u_up    = (i > 1)          ? u[idx_up]   : 0.0f;
        float u_left  = (j > 1)          ? u[idx_left] : 0.0f;
        float u_sky   = 0.0f;  // fresh start has no predecessor

        // For PARAM_GAP: add +1 tangent to transitions using gap
        if (param_type == PARAM_GAP) {
            u_up   += 1.0f;
            u_left += 1.0f;
        }

        // Propagate tangent through softmax: U[idx] = sum_k w_k * u_k
        float U_val = w_align * u_align + w_up * u_up + w_left * u_left + w_sky * u_sky;

        // For PARAM_TEMPERATURE: add (alpha - E[logits]) / T term
        // This comes from d/dT of T*log(sum exp(v/T)) = log(...) + T * d/dT[log(...)]
        // = alpha/T + (E[v] - alpha) / T = (alpha - E[v] + alpha) / T ... wait
        // Actually: d/dT [T * log(sum exp(v_k/T))] =
        //   log(sum exp(v_k/T)) + T * sum_k w_k * (-v_k/T^2) / (sum exp/sum exp)
        //   = alpha/T + (-1/T) * sum_k w_k * v_k = alpha/T - E[v]/T = (alpha - E[v])/T
        if (param_type == PARAM_TEMPERATURE) {
            float alpha_ij = a[idx];
            float E_v = w_align * v_align + w_up * v_up + w_left * v_left + w_sky * v_sky;
            U_val += (alpha_ij - E_v) / T;
        }

        u[idx] = U_val;
    }
}

// Backward W pass: compute W = dbeta/dtheta and accumulate dP/dtheta
__global__ void sw_regular_param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ partition,
    const float* __restrict__ U,
    const float* __restrict__ dS_dtheta,  // [B] pre-computed dS/dtheta
    float* __restrict__ beta,              // workspace (accumulated)
    float* __restrict__ W,                 // workspace (tangent of beta)
    float* __restrict__ dP_dtheta,         // output [B, L1, L2]
    const int* __restrict__ lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int k_diag,
    int param_type
) {
    int b = blockIdx.x;
    size_t stride = static_cast<size_t>(max_L2) + 1;
    size_t stride_alpha  = (static_cast<size_t>(max_L1) + 1) * stride;
    size_t stride_scores = static_cast<size_t>(max_L1) * max_L2;

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

        // BOUNDS CHECK - skip masked cells
        if (i > L1 || j > L2) continue;

        size_t idx       = static_cast<size_t>(i) * stride + j;
        size_t idx_diag  = static_cast<size_t>(i - 1) * stride + (j - 1);
        size_t idx_up    = static_cast<size_t>(i - 1) * stride + j;
        size_t idx_left  = static_cast<size_t>(i) * stride + (j - 1);
        size_t score_idx = static_cast<size_t>(i - 1) * max_L2 + (j - 1);

        // Read accumulated beta
        float beta_curr = b_buf[idx];

        // Compute initial W term from beta initialization tangent
        // beta_init = exp((alpha - S) / T)
        // d_beta_init/d_theta = beta_init * (U[idx] - dS) / T
        // Fix #10: Apply posterior clamping
        float beta_init = 0.0f;
        if (a[idx] > NINF) {
            float log_post = (a[idx] - S) / T;
            log_post = fminf(log_post, 0.0f);  // Clamp to <= 0
            beta_init = safe_exp(log_post);
        }
        float dW_init = beta_init * (u_buf[idx] - dS) / T;

        // For temperature: add extra term from dbeta_init/dT = ... - beta_init*(alpha-S)/T^2
        if (param_type == PARAM_TEMPERATURE && a[idx] > NINF) {
            dW_init -= beta_init * (a[idx] - S) / (T * T);
        }

        float w_curr = w_buf[idx] + dW_init;

        // Skip if both beta and W are negligible
        if (beta_curr <= 1e-20f && fabsf(w_curr) <= 1e-20f) continue;

        float score = s[score_idx];

        // Recompute logits and weights
        float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
        float v_up    = (i > 1)          ? a[idx_up]   + gap   : NINF;
        float v_left  = (j > 1)          ? a[idx_left] + gap   : NINF;
        float v_sky   = score;

        float max_v = fmaxf(fmaxf(v_align, v_up), fmaxf(v_left, v_sky));
        if (max_v <= NINF) continue;

        float w_align = safe_exp((v_align - max_v) / T);
        float w_up    = safe_exp((v_up    - max_v) / T);
        float w_left  = safe_exp((v_left  - max_v) / T);
        float w_sky   = safe_exp((v_sky   - max_v) / T);
        float sum_w = w_align + w_up + w_left + w_sky;
        float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

        w_align *= inv_sum;
        w_up    *= inv_sum;
        w_left  *= inv_sum;
        w_sky   *= inv_sum;

        // Compute tangent of logits (du values)
        float du_align = (i > 1 && j > 1) ? u_buf[idx_diag] : 0.0f;
        float du_up    = (i > 1)          ? u_buf[idx_up]   : 0.0f;
        float du_left  = (j > 1)          ? u_buf[idx_left] : 0.0f;
        float du_sky   = 0.0f;

        // For gap parameter: add +1 to gap transitions
        if (param_type == PARAM_GAP) {
            du_up   += 1.0f;
            du_left += 1.0f;
        }

        // Compute weight tangents via softmax Jacobian
        // dw_k = w_k * (du_k - E[du]) / T
        float E_du = w_align * du_align + w_up * du_up + w_left * du_left + w_sky * du_sky;
        float dw_align = w_align * (du_align - E_du) / T;
        float dw_up    = w_up    * (du_up    - E_du) / T;
        float dw_left  = w_left  * (du_left  - E_du) / T;
        float dw_sky   = w_sky   * (du_sky   - E_du) / T;

        // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
        if (param_type == PARAM_TEMPERATURE) {
            float E_v = w_align * v_align + w_up * v_up + w_left * v_left + w_sky * v_sky;
            float inv_T2 = 1.0f / (T * T);
            dw_align += w_align * (E_v - v_align) * inv_T2;
            dw_up    += w_up    * (E_v - v_up)    * inv_T2;
            dw_left  += w_left  * (E_v - v_left)  * inv_T2;
            dw_sky   += w_sky   * (E_v - v_sky)   * inv_T2;
        }

        // Propagate beta and W to predecessors
        if (v_align > NINF) {
            atomicAdd(&b_buf[idx_diag], beta_curr * w_align);
            atomicAdd(&w_buf[idx_diag], w_curr * w_align + beta_curr * dw_align);
        }
        if (v_up > NINF) {
            atomicAdd(&b_buf[idx_up], beta_curr * w_up);
            atomicAdd(&w_buf[idx_up], w_curr * w_up + beta_curr * dw_up);
        }
        if (v_left > NINF) {
            atomicAdd(&b_buf[idx_left], beta_curr * w_left);
            atomicAdd(&w_buf[idx_left], w_curr * w_left + beta_curr * dw_left);
        }

        // Accumulate dP/dtheta: posteriors = beta * (w_align + w_sky)
        // d(posteriors) = w_curr * (w_align + w_sky) + beta_curr * (dw_align + dw_sky)
        atomicAdd(&dP[score_idx], w_curr * (w_align + w_sky) + beta_curr * (dw_align + dw_sky));
    }
}

// ============================================================================
// Host Wrappers
// ============================================================================

extern "C" void sw_regular_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,     // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;

    // Initialize alpha
    int blocks_init = checked_grid_blocks(total_alpha, threads, "sw_regular_init_alpha_kernel");
    sw_regular_init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(d_alpha, B, max_L1, max_L2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Wavefront DP
    int max_diag = max_L1 + max_L2;
    for (int k = 2; k <= max_diag; ++k) {
        sw_regular_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Partition function
    sw_regular_partition_kernel<<<B, threads, 0, stream>>>(d_alpha, d_partition, d_lengths, B, max_L1, max_L2, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Synchronize to ensure all kernels complete before returning
    // This prevents stale pointer issues if caller frees input tensors
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void sw_regular_forward_warp(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = SW_WARPS_PER_BLOCK * SW_WARP_SIZE;
    int blocks = (B + SW_WARPS_PER_BLOCK - 1) / SW_WARPS_PER_BLOCK;
    cudaStream_t stream = d2p::common::get_cuda_stream();

    sw_regular_forward_warp_kernel<<<blocks, threads, 0, stream>>>(
        d_scores, d_alpha, d_partition, d_lengths,
        B, max_L1, max_L2, gap, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void sw_regular_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,          // workspace [B,(L1+1)*(L2+1)]
    float* d_posteriors,    // output [B,L1,L2]
    float* d_grad_gap,      // output [B]
    float* d_grad_T,        // output [B]
    const int* d_lengths,   // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    // Zero outputs
    C10_CUDA_CHECK(cudaMemsetAsync(d_beta,       0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_gap,   0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T,     0, sizeof(float) * B, stream));

    // Initialize beta
    int blocks_init = checked_grid_blocks(total_alpha, threads, "sw_regular_init_beta_kernel");
    sw_regular_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward DP
    int max_diag = max_L1 + max_L2;
    for (int k = max_diag; k >= 2; --k) {
        sw_regular_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_grad_gap, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Temperature gradient
    sw_regular_grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_grad_gap, d_partition, d_grad_T,
        B, max_L1, max_L2, gap, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Synchronize to ensure all kernels complete before returning
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void sw_regular_backward_warp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
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
    int warp_threads = SW_WARPS_PER_BLOCK * SW_WARP_SIZE;
    int warp_blocks = static_cast<int>(
        (static_cast<int64_t>(B) + SW_WARPS_PER_BLOCK - 1) / SW_WARPS_PER_BLOCK
    );
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * score_elems, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_gap, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    int blocks_init = checked_grid_blocks(total_alpha, threads, "sw_regular_init_beta_kernel");
    sw_regular_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sw_regular_backward_warp_kernel<<<warp_blocks, warp_threads, 0, stream>>>(
        d_alpha, d_scores, d_beta, d_posteriors, d_grad_gap,
        d_lengths, B, max_L1, max_L2, gap, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    sw_regular_grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_grad_gap, d_partition, d_grad_T,
        B, max_L1, max_L2, gap, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void sw_regular_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_V,           // input [B,L1,L2]
    float* d_d_alpha,           // workspace [B,(L1+1)*(L2+1)]
    float* d_d_partition,       // workspace [B]
    float* d_beta,              // workspace [B,(L1+1)*(L2+1)] - for beta accumulation (PASSED IN)
    float* d_d_beta,            // workspace [B,(L1+1)*(L2+1)] - for d_beta accumulation
    float* d_H_scores,          // output [B,L1,L2]
    const int* d_lengths,       // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    // Zero workspaces and output (d_beta is passed in and zeroed by caller)
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha,     0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_partition, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta,      0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores,    0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward pass: compute d_alpha (tangent of alpha w.r.t. scores in direction V)
    for (int k = 2; k <= max_diag; ++k) {
        sw_regular_hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Compute d_partition (tangent of partition function)
    sw_regular_hvp_partition_kernel<<<B, threads, 0, stream>>>(
        d_alpha, d_d_alpha, d_partition, d_d_partition, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Initialize beta = exp((alpha - S) / T) before backward pass
    int blocks_init = checked_grid_blocks(total_alpha, threads, "sw_regular_init_beta_kernel");
    sw_regular_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward pass: compute H_scores (tangent of posteriors = HVP output)
    // Now passes d_beta for accumulation
    for (int k = max_diag; k >= 2; --k) {
        sw_regular_hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_partition, d_d_alpha, d_d_partition, d_V,
            d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_L1, max_L2, gap, T, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Synchronize to ensure all kernels complete before returning
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

extern "C" void sw_regular_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    const float* d_dS_dtheta,       // pre-computed [B] - dS/dtheta from backward
    float* d_U,                     // workspace [B,(max_L1+1)*(max_L2+1)]
    float* d_beta,                  // workspace [B,(max_L1+1)*(max_L2+1)]
    float* d_W,                     // workspace [B,(max_L1+1)*(max_L2+1)]
    float* d_dP_dtheta,             // output [B,max_L1,max_L2]
    const int* d_lengths,           // [B, 2] actual lengths per batch
    int B, int max_L1, int max_L2,
    float gap, float T,
    int param_type                  // 0=gap, 1=temperature
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t alpha_elems = (static_cast<size_t>(max_L1) + 1) * (max_L2 + 1);
    size_t total_alpha = (size_t)B * alpha_elems;
    size_t score_elems = static_cast<size_t>(B) * max_L1 * max_L2;

    // Zero workspaces and output
    C10_CUDA_CHECK(cudaMemsetAsync(d_U,         0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_beta,      0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W,         0, sizeof(float) * total_alpha, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dtheta, 0, sizeof(float) * score_elems, stream));

    int max_diag = max_L1 + max_L2;

    // Forward U pass: compute U = dalpha/dtheta
    for (int k = 2; k <= max_diag; ++k) {
        sw_regular_param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_U, d_lengths, B, max_L1, max_L2,
            gap, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Initialize beta = exp((alpha - S) / T) before backward pass
    int blocks_init = checked_grid_blocks(total_alpha, threads, "sw_regular_init_beta_kernel");
    sw_regular_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_alpha, d_partition, d_beta, d_lengths, B, max_L1, max_L2, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Backward W pass: compute W = dbeta/dtheta and accumulate dP/dtheta
    for (int k = max_diag; k >= 2; --k) {
        sw_regular_param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_partition, d_U, d_dS_dtheta,
            d_beta, d_W, d_dP_dtheta, d_lengths,
            B, max_L1, max_L2, gap, T, k, param_type
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Synchronize to ensure all kernels complete before returning
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
