// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft Monotonic Alignment Search (MAS) CUDA Kernel Implementations
 *
 * MAS aligns frames (T) to text positions (S) with monotonic constraint.
 * Uses anti-diagonal wavefront parallelization for the 2D DP.
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <cmath>

namespace d2p {
namespace mas {

// =============================================================================
// Device Helpers
// =============================================================================

#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < (T)-88.0f) return (T)0.0f;
    if (x > (T)88.0f) x = (T)88.0f;
    return exp(x);
}

__device__ __forceinline__ size_t cell_offset(int t, int s, int max_S) {
    return (size_t)t * (size_t)max_S + (size_t)s;
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

__device__ __forceinline__ float softmax2(float a, float b, float T) {
    float m = fmaxf(a, b);
    if (m <= NINF) return NINF;

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;

    float sum = ea + eb;
    if (sum <= 0.0f) return NINF;
    return m + T * logf(sum);
}

__device__ __forceinline__ void softmax2_weights(
    float a, float b, float T,
    float& wa, float& wb
) {
    float m = fmaxf(a, b);
    if (m <= NINF) {
        wa = wb = 0.0f;
        return;
    }

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;

    float total = ea + eb;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
    } else {
        wa = wb = 0.0f;
    }
}

__device__ __forceinline__ void softmax2_tangent(
    float w1, float w2,
    float dv1, float dv2,
    float T,
    float& dw1, float& dw2
) {
    float sum_w_dv = w1 * dv1 + w2 * dv2;
    dw1 = w1 * (dv1 - sum_w_dv) / T;
    dw2 = w2 * (dv2 - sum_w_dv) / T;
}

// =============================================================================
// Forward Kernels
// =============================================================================

__global__ void init_alpha_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    size_t stride = (size_t)max_T * max_S;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t off = idx - b * stride;
    int t = (int)(off / max_S);
    int s = (int)(off % max_S);

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    if (t >= T || s >= S) {
        a[cell_offset(t, s, max_S)] = NINF;
        return;
    }

    if (t == 0 && s == 0) {
        a[0] = sc[0];
    } else if (s == 0 && t > 0) {
        a[cell_offset(t, 0, max_S)] = NINF;
    } else if (t == 0 && s > 0) {
        a[s] = NINF;
    } else {
        a[cell_offset(t, s, max_S)] = NINF;
    }
}

__global__ void init_first_col_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    for (int t = 1; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        size_t prev_idx = cell_offset(t - 1, 0, max_S);
        a[idx] = a[prev_idx] + sc[idx];
    }
}

__global__ void forward_diag_kernel(
    const float* __restrict__ scores,
    float* __restrict__ alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* sc = scores + b * stride;
    float* a = alpha + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        a[idx] = sc[idx] + softmax2(stay, diag, T_temp);
    }
}

__global__ void score_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ score,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    size_t final_idx = cell_offset(T - 1, S - 1, max_S);

    score[b] = alpha[b * stride + final_idx];
}

// =============================================================================
// Backward Kernels
// =============================================================================

__global__ void init_beta_kernel(
    float* __restrict__ beta,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    size_t stride = (size_t)max_T * max_S;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t off = idx - b * stride;
    int t = (int)(off / max_S);
    int s = (int)(off % max_S);

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    if (t == T - 1 && s == S - 1) {
        beta[idx] = 1.0f;
    } else {
        beta[idx] = 0.0f;
    }
}

__global__ void backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    float* be = beta + b * stride;
    float* P = posteriors + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        if (beta_ts < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * w_stay);
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * w_diag);
        }

        P[idx] = beta_ts;
    }
}

__global__ void backward_first_col_kernel(
    float* __restrict__ beta,
    float* __restrict__ posteriors,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    float* be = beta + b * stride;
    float* P = posteriors + b * stride;

    for (int t = T - 1; t >= 0; t--) {
        size_t idx = cell_offset(t, 0, max_S);
        P[idx] = be[idx];
    }
}

__global__ void grad_T_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ posteriors,
    const float* __restrict__ partition,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S, float T_temp
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    const float* sc = scores + b * stride;
    const float* P = posteriors + b * stride;

    float expected_score = 0.0f;

    size_t total_valid = (size_t)T * (size_t)S;
    for (size_t idx = threadIdx.x; idx < total_valid; idx += blockDim.x) {
        int t = (int)(idx / (size_t)S);
        int s = (int)(idx % (size_t)S);
        if (t < T && s < S) {
            size_t flat_idx = cell_offset(t, s, max_S);
            expected_score += P[flat_idx] * sc[flat_idx];
        }
    }

    expected_score = block_reduce_sum(expected_score);

    if (threadIdx.x == 0) {
        grad_T[b] = (partition[b] - expected_score) / T_temp;
    }
}

// =============================================================================
// HVP Kernels
// =============================================================================

__global__ void hvp_init_first_col_kernel(
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* v = V + b * stride;
    float* da = d_alpha + b * stride;

    da[0] = v[0];
    for (int t = 1; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        size_t prev_idx = cell_offset(t - 1, 0, max_S);
        da[idx] = da[prev_idx] + v[idx];
    }
}

__global__ void hvp_forward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ scores,
    const float* __restrict__ V,
    float* __restrict__ d_alpha,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* v = V + b * stride;
    float* da = d_alpha + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        da[idx] = v[idx] + w_stay * da[idx_stay] + w_diag * da[idx_diag];
    }
}

__global__ void hvp_score_kernel(
    const float* __restrict__ d_alpha,
    float* __restrict__ d_score,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;
    size_t final_idx = cell_offset(T - 1, S - 1, max_S);

    d_score[b] = d_alpha[b * stride + final_idx];
}

__global__ void hvp_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ V,
    const float* __restrict__ d_alpha,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* da = d_alpha + b * stride;
    float* be = beta + b * stride;
    float* dbe = d_beta + b * stride;
    float* H = H_scores + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        float dbeta_ts = dbe[idx];

        if (beta_ts < 1e-30f && fabsf(dbeta_ts) < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        float da_stay = da[idx_stay];
        float da_diag = (s >= 1) ? da[idx_diag] : 0.0f;

        float dw_stay, dw_diag;
        softmax2_tangent(w_stay, w_diag, da_stay, da_diag, T_temp, dw_stay, dw_diag);

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * w_stay);
            atomicAdd(&dbe[idx_stay], dbeta_ts * w_stay + beta_ts * dw_stay);
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * w_diag);
            atomicAdd(&dbe[idx_diag], dbeta_ts * w_diag + beta_ts * dw_diag);
        }

        H[idx] = dbeta_ts;
    }
}

__global__ void hvp_backward_first_col_kernel(
    float* __restrict__ d_beta,
    float* __restrict__ H_scores,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    float* dbe = d_beta + b * stride;
    float* H = H_scores + b * stride;

    for (int t = 0; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        H[idx] = dbe[idx];
    }
}

// =============================================================================
// Parameter Gradient Kernels
// =============================================================================

__global__ void param_grad_forward_diag_kernel(
    const float* __restrict__ alpha,
    float* __restrict__ U,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    float* u = U + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag - 1);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 1 || s >= S) continue;

        size_t idx = cell_offset(t, s, max_S);
        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = a[idx_diag];

        float w_stay, w_diag;
        softmax2_weights(stay, diag, T_temp, w_stay, w_diag);

        float u_stay = u[idx_stay];
        float u_diag = u[idx_diag];

        if (a[idx] <= NINF) {
            u[idx] = 0.0f;
            continue;
        }

        float E_v = w_stay * stay + w_diag * diag;
        float soft_value = softmax2(stay, diag, T_temp);

        // d/dT [T * logsumexp(v / T)] = sum_i w_i * dv_i/dT + (f - E[v]) / T.
        u[idx] = w_stay * u_stay + w_diag * u_diag + (soft_value - E_v) / T_temp;
    }
}

__global__ void param_grad_backward_diag_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S,
    float T_temp, int k_diag
) {
    int b = blockIdx.x;
    int T = lengths[b * 2];
    int S = lengths[b * 2 + 1];

    size_t stride = (size_t)max_T * max_S;

    const float* a = alpha + b * stride;
    const float* u = U + b * stride;
    float* be = beta + b * stride;
    float* w = W + b * stride;
    float* dP = dP_dT + b * stride;

    int t_start = max(1, k_diag - (S - 1));
    int t_end = min(T - 1, k_diag);
    int diag_len = t_end - t_start + 1;
    if (diag_len <= 0) return;

    for (int i = threadIdx.x; i < diag_len; i += blockDim.x) {
        int t = t_start + i;
        int s = k_diag - t;
        if (s < 0 || s >= S) continue;
        if (t < 1) continue;

        size_t idx = cell_offset(t, s, max_S);
        float beta_ts = be[idx];
        float w_ts = w[idx];

        if (beta_ts < 1e-30f && fabsf(w_ts) < 1e-30f) continue;

        size_t idx_stay = cell_offset(t - 1, s, max_S);
        size_t idx_diag = cell_offset(t - 1, s - 1, max_S);

        float stay = a[idx_stay];
        float diag = (s >= 1) ? a[idx_diag] : NINF;

        float wt_stay, wt_diag;
        softmax2_weights(stay, diag, T_temp, wt_stay, wt_diag);

        float u_stay = u[idx_stay];
        float u_diag = (s >= 1) ? u[idx_diag] : 0.0f;

        float dw_stay, dw_diag;
        softmax2_tangent(wt_stay, wt_diag, u_stay, u_diag, T_temp, dw_stay, dw_diag);

        float E_v = wt_stay * stay + wt_diag * diag;
        float inv_T2 = 1.0f / (T_temp * T_temp);
        dw_stay += wt_stay * (E_v - stay) * inv_T2;
        dw_diag += wt_diag * (E_v - diag) * inv_T2;

        if (stay > NINF) {
            atomicAdd(&be[idx_stay], beta_ts * wt_stay);
            atomicAdd(&w[idx_stay], w_ts * wt_stay + beta_ts * dw_stay);
        }
        if (s >= 1 && diag > NINF) {
            atomicAdd(&be[idx_diag], beta_ts * wt_diag);
            atomicAdd(&w[idx_diag], w_ts * wt_diag + beta_ts * dw_diag);
        }

        dP[idx] = w_ts;
    }
}

__global__ void param_grad_backward_first_col_kernel(
    const float* __restrict__ W,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int max_T, int max_S
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int T = lengths[b * 2];
    size_t stride = (size_t)max_T * max_S;

    const float* w = W + b * stride;
    float* dP = dP_dT + b * stride;

    for (int t = 0; t < T; t++) {
        size_t idx = cell_offset(t, 0, max_S);
        dP[idx] = w[idx];
    }
}

// =============================================================================
// Host Functions
// =============================================================================

void forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    size_t blocks_init = (total + threads - 1) / threads;
    init_alpha_kernel<<<blocks_init, threads, 0, stream>>>(
        d_scores, d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    init_first_col_kernel<<<B, 1, 0, stream>>>(
        d_scores, d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;
    for (int k = 2; k <= max_diag; ++k) {
        forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_scores, d_alpha, d_lengths, B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_alpha, d_partition, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_posteriors, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_grad_T, 0, sizeof(float) * B, stream));

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;
    for (int k = max_diag; k >= 1; --k) {
        backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_beta, d_posteriors, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_beta, d_posteriors, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    grad_T_kernel<<<B, threads, 0, stream>>>(
        d_scores, d_posteriors, d_partition, d_grad_T, d_lengths,
        B, max_T, max_S, temperature
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_d_alpha, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_score, 0, sizeof(float) * B, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_d_beta, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_H_scores, 0, sizeof(float) * total, stream));

    hvp_init_first_col_kernel<<<B, 1, 0, stream>>>(
        d_V, d_d_alpha, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int max_diag = max_T + max_S - 2;

    for (int k = 2; k <= max_diag; ++k) {
        hvp_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_scores, d_V, d_d_alpha, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_score = (B + threads - 1) / threads;
    hvp_score_kernel<<<blocks_score, threads, 0, stream>>>(
        d_d_alpha, d_d_score, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 1; --k) {
        hvp_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_V, d_d_alpha, d_beta, d_d_beta, d_H_scores, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    hvp_backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_d_beta, d_H_scores, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void param_grad(
    const float* d_alpha,
    const float* d_scores,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t total = (size_t)B * max_T * max_S;

    C10_CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_W, 0, sizeof(float) * total, stream));
    C10_CUDA_CHECK(cudaMemsetAsync(d_dP_dT, 0, sizeof(float) * total, stream));

    int max_diag = max_T + max_S - 2;

    for (int k = 2; k <= max_diag; ++k) {
        param_grad_forward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_U, d_lengths, B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    size_t blocks_init = (total + threads - 1) / threads;
    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int k = max_diag; k >= 1; --k) {
        param_grad_backward_diag_kernel<<<B, threads, 0, stream>>>(
            d_alpha, d_U, d_beta, d_W, d_dP_dT, d_lengths,
            B, max_T, max_S, temperature, k
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    param_grad_backward_first_col_kernel<<<B, 1, 0, stream>>>(
        d_W, d_dP_dT, d_lengths, B, max_T, max_S
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace mas
} // namespace d2p
