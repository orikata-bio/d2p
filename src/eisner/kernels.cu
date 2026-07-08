// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft Eisner CUDA Kernel Implementations
 *
 * Differentiable Eisner algorithm for projective dependency parsing.
 * Uses wavefront parallelization over span lengths.
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <math.h>

namespace d2p {
namespace eisner {

// ============================================================================
// Device Helpers
// ============================================================================

#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T safe_exp(T x) {
    if (x < -88.0f) return (T)0.0f;
    if (x > 88.0f) x = (T)88.0f;
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

__device__ __forceinline__ size_t chart_index(int row, int col, int n) {
    return (size_t)row * (size_t)n + (size_t)col;
}

// ============================================================================
// Forward Pass Kernels
// ============================================================================

__global__ void init_kernel(
    float* __restrict__ C_R,
    float* __restrict__ C_L,
    float* __restrict__ I_R,
    float* __restrict__ I_L,
    const int* __restrict__ lengths,
    int B, int n
) {
    size_t stride = (size_t)n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;
    if (idx >= total) return;

    size_t b = idx / stride;
    size_t rem = idx - b * stride;
    int i = rem / n;
    int j = rem % n;

    int seq_len = lengths ? lengths[b] : n;

    if (i == j && i < seq_len) {
        C_R[idx] = 0.0f;
        C_L[idx] = 0.0f;
    } else {
        C_R[idx] = NINF;
        C_L[idx] = NINF;
    }
    I_R[idx] = NINF;
    I_L[idx] = NINF;
}

__global__ void forward_incomplete_kernel(
    const float* __restrict__ arc_scores,
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    float* __restrict__ I_R,
    float* __restrict__ I_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* arc = arc_scores + b * stride;
    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    float* ir = I_R + b * stride;
    float* il = I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        // I_R[i,j] = arc[i,j] + LSE_k{ C_R[i,k] + C_L[k+1,j] }
        // I_L[i,j] = arc[j,i] + LSE_k{ C_R[i,k] + C_L[k+1,j] }

        float max_v = NINF;
        for (int k = i; k < j; k++) {
            float v = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            max_v = fmaxf(max_v, v);
        }

        if (max_v <= NINF) {
            ir[chart_index(i, j, n)] = NINF;
            il[chart_index(i, j, n)] = NINF;
            continue;
        }

        float sum_exp = 0.0f;
        for (int k = i; k < j; k++) {
            float v = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            sum_exp += safe_exp((v - max_v) / T);
        }

        float lse = max_v + T * logf(sum_exp);
        ir[chart_index(i, j, n)] = arc[chart_index(i, j, n)] + lse;
        il[chart_index(i, j, n)] = arc[chart_index(j, i, n)] + lse;
    }
}

__global__ void forward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    float* __restrict__ C_R_out,
    float* __restrict__ C_L_out,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    float* cr_out = C_R_out + b * stride;
    float* cl_out = C_L_out + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        // C_R[i,j] = LSE_k{ C_R[i,k] + I_R[k,j] }
        {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                cr_out[chart_index(i, j, n)] = NINF;
            } else {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                cr_out[chart_index(i, j, n)] = max_v + T * logf(sum_exp);
            }
        }

        // C_L[i,j] = LSE_k{ I_L[i,k] + C_L[k,j] }
        {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                cl_out[chart_index(i, j, n)] = NINF;
            } else {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                cl_out[chart_index(i, j, n)] = max_v + T * logf(sum_exp);
            }
        }
    }
}

__global__ void extract_partition_kernel(
    const float* __restrict__ C_R,
    float* __restrict__ partition,
    const int* __restrict__ lengths,
    int B, int n
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    int seq_len = lengths ? lengths[b] : n;
    partition[b] = C_R[(size_t)b * n * n + (seq_len - 1)];
}

// ============================================================================
// Backward Pass Kernels
// ============================================================================

__global__ void init_beta_kernel(
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ marginals,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int n
) {
    size_t stride = (size_t)n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;

    if (idx < total) {
        size_t b = idx / stride;
        size_t rem = idx - b * stride;
        int i = rem / n;
        int j = rem % n;

        int seq_len = lengths ? lengths[b] : n;

        if (i == 0 && j == seq_len - 1) {
            beta_C_R[idx] = 1.0f;
        } else {
            beta_C_R[idx] = 0.0f;
        }
        beta_C_L[idx] = 0.0f;
        beta_I_R[idx] = 0.0f;
        beta_I_L[idx] = 0.0f;
        marginals[idx] = 0.0f;
    }

    if (idx < (size_t)B) {
        grad_T[idx] = 0.0f;
    }
}

__global__ void backward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    float* bir = beta_I_R + b * stride;
    float* bil = beta_I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    float local_grad_T = 0.0f;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        // Backward for C_R[i,j]
        float beta_cr_ij = bcr[chart_index(i, j, n)];
        if (beta_cr_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float Zij = cr[chart_index(i, j, n)];
                float E_term = 0.0f;

                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float mass = beta_cr_ij * w;

                    atomicAdd(&bcr[chart_index(i, k, n)], mass);
                    atomicAdd(&bir[chart_index(k, j, n)], mass);

                    E_term += w * v;
                }

                local_grad_T += beta_cr_ij * (Zij - E_term) / T;
            }
        }

        // Backward for C_L[i,j]
        float beta_cl_ij = bcl[chart_index(i, j, n)];
        if (beta_cl_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float Zij = cl[chart_index(i, j, n)];
                float E_term = 0.0f;

                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float mass = beta_cl_ij * w;

                    atomicAdd(&bil[chart_index(i, k, n)], mass);
                    atomicAdd(&bcl[chart_index(k, j, n)], mass);

                    E_term += w * v;
                }

                local_grad_T += beta_cl_ij * (Zij - E_term) / T;
            }
        }
    }

    float block_grad_T = block_reduce_sum(local_grad_T);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_grad_T);
    }
}

__global__ void backward_incomplete_kernel(
    const float* __restrict__ arc_scores,
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    const float* __restrict__ beta_I_R,
    const float* __restrict__ beta_I_L,
    float* __restrict__ marginals,
    float* __restrict__ grad_T,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* arc = arc_scores + b * stride;
    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    const float* bir = beta_I_R + b * stride;
    const float* bil = beta_I_L + b * stride;
    float* marg = marginals + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    float local_grad_T = 0.0f;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        float beta_ir_ij = bir[chart_index(i, j, n)];
        float beta_il_ij = bil[chart_index(i, j, n)];

        // Arc marginals
        marg[chart_index(i, j, n)] = beta_ir_ij;
        marg[chart_index(j, i, n)] = beta_il_ij;

        float beta_combined = beta_ir_ij + beta_il_ij;

        if (beta_combined != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float lse = max_v + T * logf(sum_exp);
                float E_term = 0.0f;

                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float mass = beta_combined * w;

                    atomicAdd(&bcr[chart_index(i, k, n)], mass);
                    atomicAdd(&bcl[chart_index(k + 1, j, n)], mass);

                    E_term += w * v;
                }

                local_grad_T += beta_combined * (lse - E_term) / T;
            }
        }
    }

    float block_grad_T = block_reduce_sum(local_grad_T);
    if (threadIdx.x == 0) {
        atomicAdd(&grad_T[b], block_grad_T);
    }
}

// ============================================================================
// HVP Kernels
// ============================================================================

__global__ void hvp_init_kernel(
    const float* __restrict__ V,
    float* __restrict__ d_C_R,
    float* __restrict__ d_C_L,
    float* __restrict__ d_I_R,
    float* __restrict__ d_I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ d_beta_C_R,
    float* __restrict__ d_beta_C_L,
    float* __restrict__ d_beta_I_R,
    float* __restrict__ d_beta_I_L,
    float* __restrict__ HVP,
    const int* __restrict__ lengths,
    int B, int n
) {
    size_t stride = (size_t)n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;

    if (idx < total) {
        size_t b = idx / stride;
        size_t rem = idx - b * stride;
        int i = rem / n;
        int j = rem % n;

        int seq_len = lengths ? lengths[b] : n;

        if (i == j && i < seq_len) {
            d_C_R[idx] = 0.0f;
            d_C_L[idx] = 0.0f;
        } else {
            d_C_R[idx] = 0.0f;
            d_C_L[idx] = 0.0f;
        }
        d_I_R[idx] = 0.0f;
        d_I_L[idx] = 0.0f;

        if (i == 0 && j == seq_len - 1) {
            beta_C_R[idx] = 1.0f;
        } else {
            beta_C_R[idx] = 0.0f;
        }
        beta_C_L[idx] = 0.0f;
        beta_I_R[idx] = 0.0f;
        beta_I_L[idx] = 0.0f;

        d_beta_C_R[idx] = 0.0f;
        d_beta_C_L[idx] = 0.0f;
        d_beta_I_R[idx] = 0.0f;
        d_beta_I_L[idx] = 0.0f;

        HVP[idx] = 0.0f;
    }
}

__global__ void hvp_forward_incomplete_kernel(
    const float* __restrict__ arc_scores,
    const float* __restrict__ V,
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ d_C_R,
    const float* __restrict__ d_C_L,
    float* __restrict__ d_I_R,
    float* __restrict__ d_I_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* v = V + b * stride;
    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* dcr = d_C_R + b * stride;
    const float* dcl = d_C_L + b * stride;
    float* dir = d_I_R + b * stride;
    float* dil = d_I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        float max_v = NINF;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            max_v = fmaxf(max_v, val);
        }

        if (max_v <= NINF) {
            dir[chart_index(i, j, n)] = 0.0f;
            dil[chart_index(i, j, n)] = 0.0f;
            continue;
        }

        float sum_exp = 0.0f;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            sum_exp += safe_exp((val - max_v) / T);
        }
        float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

        float d_lse = 0.0f;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            float w = safe_exp((val - max_v) / T) * inv_sum;
            d_lse += w * (dcr[chart_index(i, k, n)] + dcl[chart_index(k + 1, j, n)]);
        }

        dir[chart_index(i, j, n)] = v[chart_index(i, j, n)] + d_lse;
        dil[chart_index(i, j, n)] = v[chart_index(j, i, n)] + d_lse;
    }
}

__global__ void hvp_forward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    const float* __restrict__ d_C_R_in,
    const float* __restrict__ d_C_L_in,
    const float* __restrict__ d_I_R,
    const float* __restrict__ d_I_L,
    float* __restrict__ d_C_R,
    float* __restrict__ d_C_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    const float* dcr_in = d_C_R_in + b * stride;
    const float* dcl_in = d_C_L_in + b * stride;
    const float* dir = d_I_R + b * stride;
    const float* dil = d_I_L + b * stride;
    float* dcr = d_C_R + b * stride;
    float* dcl = d_C_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        // d_C_R[i,j]
        {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                dcr[chart_index(i, j, n)] = 0.0f;
            } else {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float d_lse = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    d_lse += w * (dcr_in[chart_index(i, k, n)] + dir[chart_index(k, j, n)]);
                }
                dcr[chart_index(i, j, n)] = d_lse;
            }
        }

        // d_C_L[i,j]
        {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                dcl[chart_index(i, j, n)] = 0.0f;
            } else {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float d_lse = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    d_lse += w * (dil[chart_index(i, k, n)] + dcl_in[chart_index(k, j, n)]);
                }
                dcl[chart_index(i, j, n)] = d_lse;
            }
        }
    }
}

__global__ void hvp_backward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    const float* __restrict__ d_C_R,
    const float* __restrict__ d_C_L,
    const float* __restrict__ d_I_R,
    const float* __restrict__ d_I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ d_beta_C_R,
    float* __restrict__ d_beta_C_L,
    float* __restrict__ d_beta_I_R,
    float* __restrict__ d_beta_I_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    const float* dcr = d_C_R + b * stride;
    const float* dcl = d_C_L + b * stride;
    const float* dir = d_I_R + b * stride;
    const float* dil = d_I_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    float* bir = beta_I_R + b * stride;
    float* bil = beta_I_L + b * stride;
    float* dbcr = d_beta_C_R + b * stride;
    float* dbcl = d_beta_C_L + b * stride;
    float* dbir = d_beta_I_R + b * stride;
    float* dbil = d_beta_I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        // Backward for C_R[i,j]
        float beta_cr_ij = bcr[chart_index(i, j, n)];
        float d_beta_cr_ij = dbcr[chart_index(i, j, n)];

        if (beta_cr_ij != 0.0f || d_beta_cr_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_d_term = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_d_term += w * (dcr[chart_index(i, k, n)] + dir[chart_index(k, j, n)]);
                }

                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float d_term = dcr[chart_index(i, k, n)] + dir[chart_index(k, j, n)];
                    float d_w = w * (d_term - E_d_term) / T;

                    float mass = beta_cr_ij * w;
                    float d_mass = d_beta_cr_ij * w + beta_cr_ij * d_w;

                    atomicAdd(&bcr[chart_index(i, k, n)], mass);
                    atomicAdd(&bir[chart_index(k, j, n)], mass);
                    atomicAdd(&dbcr[chart_index(i, k, n)], d_mass);
                    atomicAdd(&dbir[chart_index(k, j, n)], d_mass);
                }
            }
        }

        // Backward for C_L[i,j]
        float beta_cl_ij = bcl[chart_index(i, j, n)];
        float d_beta_cl_ij = dbcl[chart_index(i, j, n)];

        if (beta_cl_ij != 0.0f || d_beta_cl_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_d_term = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_d_term += w * (dil[chart_index(i, k, n)] + dcl[chart_index(k, j, n)]);
                }

                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float d_term = dil[chart_index(i, k, n)] + dcl[chart_index(k, j, n)];
                    float d_w = w * (d_term - E_d_term) / T;

                    float mass = beta_cl_ij * w;
                    float d_mass = d_beta_cl_ij * w + beta_cl_ij * d_w;

                    atomicAdd(&bil[chart_index(i, k, n)], mass);
                    atomicAdd(&bcl[chart_index(k, j, n)], mass);
                    atomicAdd(&dbil[chart_index(i, k, n)], d_mass);
                    atomicAdd(&dbcl[chart_index(k, j, n)], d_mass);
                }
            }
        }
    }
}

__global__ void hvp_backward_incomplete_kernel(
    const float* __restrict__ V,
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ d_C_R,
    const float* __restrict__ d_C_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    const float* __restrict__ beta_I_R,
    const float* __restrict__ beta_I_L,
    float* __restrict__ d_beta_C_R,
    float* __restrict__ d_beta_C_L,
    const float* __restrict__ d_beta_I_R,
    const float* __restrict__ d_beta_I_L,
    float* __restrict__ HVP,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* v = V + b * stride;
    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* dcr = d_C_R + b * stride;
    const float* dcl = d_C_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    const float* bir = beta_I_R + b * stride;
    const float* bil = beta_I_L + b * stride;
    float* dbcr = d_beta_C_R + b * stride;
    float* dbcl = d_beta_C_L + b * stride;
    const float* dbir = d_beta_I_R + b * stride;
    const float* dbil = d_beta_I_L + b * stride;
    float* hvp_out = HVP + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;

        if (j >= seq_len) continue;

        float beta_ir = bir[chart_index(i, j, n)];
        float beta_il = bil[chart_index(i, j, n)];
        float d_beta_ir = dbir[chart_index(i, j, n)];
        float d_beta_il = dbil[chart_index(i, j, n)];

        // HVP output
        hvp_out[chart_index(i, j, n)] = d_beta_ir;
        hvp_out[chart_index(j, i, n)] = d_beta_il;

        float beta_combined = beta_ir + beta_il;
        float d_beta_combined = d_beta_ir + d_beta_il;

        if (beta_combined != 0.0f || d_beta_combined != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                max_v = fmaxf(max_v, val);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    sum_exp += safe_exp((val - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_d_term = 0.0f;
                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    float w = safe_exp((val - max_v) / T) * inv_sum;
                    E_d_term += w * (dcr[chart_index(i, k, n)] + dcl[chart_index(k + 1, j, n)]);
                }

                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    float w = safe_exp((val - max_v) / T) * inv_sum;
                    float d_term = dcr[chart_index(i, k, n)] + dcl[chart_index(k + 1, j, n)];
                    float d_w = w * (d_term - E_d_term) / T;

                    float mass = beta_combined * w;
                    float d_mass = d_beta_combined * w + beta_combined * d_w;

                    atomicAdd(&bcr[chart_index(i, k, n)], mass);
                    atomicAdd(&bcl[chart_index(k + 1, j, n)], mass);
                    atomicAdd(&dbcr[chart_index(i, k, n)], d_mass);
                    atomicAdd(&dbcl[chart_index(k + 1, j, n)], d_mass);
                }
            }
        }
    }
}

// ============================================================================
// Host API
// ============================================================================

void forward(
    const float* arc_scores,
    float* C_R,
    float* C_L,
    float* I_R,
    float* I_L,
    float* partition,
    const int* lengths,
    int B, int n, float temperature
) {
    cudaStream_t stream = d2p::common::get_cuda_stream();
    int threads = 256;
    size_t elems = (size_t)B * n * n;
    int blocks_init = (elems + threads - 1) / threads;

    init_kernel<<<blocks_init, threads, 0, stream>>>(C_R, C_L, I_R, I_L, lengths, B, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int len = 1; len < n; len++) {
        forward_incomplete_kernel<<<B, threads, 0, stream>>>(
            arc_scores, C_R, C_L, I_R, I_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        forward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L, C_R, C_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    int blocks_extract = (B + 255) / 256;
    extract_partition_kernel<<<blocks_extract, 256, 0, stream>>>(C_R, partition, lengths, B, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void backward(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* marginals,
    float* grad_T,
    const int* lengths,
    int B, int n, float temperature
) {
    cudaStream_t stream = d2p::common::get_cuda_stream();
    int threads = 256;
    size_t elems = (size_t)B * n * n;
    int blocks_init = (elems + threads - 1) / threads;

    init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        beta_C_R, beta_C_L, beta_I_R, beta_I_L, marginals, grad_T, lengths, B, n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int len = n - 1; len >= 1; len--) {
        backward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            grad_T, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        backward_incomplete_kernel<<<B, threads, 0, stream>>>(
            arc_scores, C_R, C_L, I_R, I_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            marginals, grad_T, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void hvp(
    const float* arc_scores,
    const float* V,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* d_C_R,
    float* d_C_L,
    float* d_I_R,
    float* d_I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* d_beta_C_R,
    float* d_beta_C_L,
    float* d_beta_I_R,
    float* d_beta_I_L,
    float* HVP,
    const int* lengths,
    int B, int n, float temperature
) {
    cudaStream_t stream = d2p::common::get_cuda_stream();
    int threads = 256;
    size_t elems = (size_t)B * n * n;
    int blocks_init = (elems + threads - 1) / threads;

    hvp_init_kernel<<<blocks_init, threads, 0, stream>>>(
        V, d_C_R, d_C_L, d_I_R, d_I_L,
        beta_C_R, beta_C_L, beta_I_R, beta_I_L,
        d_beta_C_R, d_beta_C_L, d_beta_I_R, d_beta_I_L,
        HVP, lengths, B, n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    for (int len = 1; len < n; len++) {
        hvp_forward_incomplete_kernel<<<B, threads, 0, stream>>>(
            arc_scores, V, C_R, C_L, d_C_R, d_C_L,
            d_I_R, d_I_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hvp_forward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L, d_C_R, d_C_L, d_I_R, d_I_L,
            d_C_R, d_C_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    for (int len = n - 1; len >= 1; len--) {
        hvp_backward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L, d_C_R, d_C_L, d_I_R, d_I_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            d_beta_C_R, d_beta_C_L, d_beta_I_R, d_beta_I_L,
            lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        hvp_backward_incomplete_kernel<<<B, threads, 0, stream>>>(
            V, C_R, C_L, d_C_R, d_C_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            d_beta_C_R, d_beta_C_L, d_beta_I_R, d_beta_I_L,
            HVP, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ============================================================================
// Parameter Gradient Kernels (dP/dT)
// ============================================================================

__global__ void param_grad_init_kernel(
    float* __restrict__ U_C_R,
    float* __restrict__ U_C_L,
    float* __restrict__ U_I_R,
    float* __restrict__ U_I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ W_C_R,
    float* __restrict__ W_C_L,
    float* __restrict__ W_I_R,
    float* __restrict__ W_I_L,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int n
) {
    size_t stride = (size_t)n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * stride;

    if (idx < total) {
        size_t b = idx / stride;
        size_t rem = idx - b * stride;
        int i = rem / n;
        int j = rem % n;
        int seq_len = lengths ? lengths[b] : n;

        U_C_R[idx] = 0.0f;
        U_C_L[idx] = 0.0f;
        U_I_R[idx] = 0.0f;
        U_I_L[idx] = 0.0f;

        if (i == 0 && j == seq_len - 1) {
            beta_C_R[idx] = 1.0f;
        } else {
            beta_C_R[idx] = 0.0f;
        }
        beta_C_L[idx] = 0.0f;
        beta_I_R[idx] = 0.0f;
        beta_I_L[idx] = 0.0f;

        W_C_R[idx] = 0.0f;
        W_C_L[idx] = 0.0f;
        W_I_R[idx] = 0.0f;
        W_I_L[idx] = 0.0f;

        dP_dT[idx] = 0.0f;
    }
}

__global__ void param_grad_forward_incomplete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ U_C_R,
    const float* __restrict__ U_C_L,
    float* __restrict__ U_I_R,
    float* __restrict__ U_I_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ucr = U_C_R + b * stride;
    const float* ucl = U_C_L + b * stride;
    float* uir = U_I_R + b * stride;
    float* uil = U_I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;
        if (j >= seq_len) continue;

        float max_v = NINF;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            max_v = fmaxf(max_v, val);
        }

        if (max_v <= NINF) {
            uir[chart_index(i, j, n)] = 0.0f;
            uil[chart_index(i, j, n)] = 0.0f;
            continue;
        }

        float sum_exp = 0.0f;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            sum_exp += safe_exp((val - max_v) / T);
        }
        float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;
        float lse = max_v + T * logf(sum_exp);

        float E_term = 0.0f;
        float E_U = 0.0f;
        for (int k = i; k < j; k++) {
            float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
            float w = safe_exp((val - max_v) / T) * inv_sum;
            E_term += w * val;
            E_U += w * (ucr[chart_index(i, k, n)] + ucl[chart_index(k + 1, j, n)]);
        }

        float U_lse = (lse - E_term) / T + E_U;
        uir[chart_index(i, j, n)] = U_lse;
        uil[chart_index(i, j, n)] = U_lse;
    }
}

__global__ void param_grad_forward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    const float* __restrict__ U_C_R_in,
    const float* __restrict__ U_C_L_in,
    const float* __restrict__ U_I_R,
    const float* __restrict__ U_I_L,
    float* __restrict__ U_C_R,
    float* __restrict__ U_C_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    const float* ucr_in = U_C_R_in + b * stride;
    const float* ucl_in = U_C_L_in + b * stride;
    const float* uir = U_I_R + b * stride;
    const float* uil = U_I_L + b * stride;
    float* ucr = U_C_R + b * stride;
    float* ucl = U_C_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;
        if (j >= seq_len) continue;

        // U_C_R[i,j]
        {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                ucr[chart_index(i, j, n)] = 0.0f;
            } else {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;
                float Zij = cr[chart_index(i, j, n)];

                float E_term = 0.0f;
                float E_U = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_term += w * v;
                    E_U += w * (ucr_in[chart_index(i, k, n)] + uir[chart_index(k, j, n)]);
                }
                ucr[chart_index(i, j, n)] = (Zij - E_term) / T + E_U;
            }
        }

        // U_C_L[i,j]
        {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v <= NINF) {
                ucl[chart_index(i, j, n)] = 0.0f;
            } else {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;
                float Zij = cl[chart_index(i, j, n)];

                float E_term = 0.0f;
                float E_U = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_term += w * v;
                    E_U += w * (uil[chart_index(i, k, n)] + ucl_in[chart_index(k, j, n)]);
                }
                ucl[chart_index(i, j, n)] = (Zij - E_term) / T + E_U;
            }
        }
    }
}

__global__ void param_grad_backward_complete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ I_R,
    const float* __restrict__ I_L,
    const float* __restrict__ U_C_R,
    const float* __restrict__ U_C_L,
    const float* __restrict__ U_I_R,
    const float* __restrict__ U_I_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    float* __restrict__ beta_I_R,
    float* __restrict__ beta_I_L,
    float* __restrict__ W_C_R,
    float* __restrict__ W_C_L,
    float* __restrict__ W_I_R,
    float* __restrict__ W_I_L,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ir = I_R + b * stride;
    const float* il = I_L + b * stride;
    const float* ucr = U_C_R + b * stride;
    const float* ucl = U_C_L + b * stride;
    const float* uir = U_I_R + b * stride;
    const float* uil = U_I_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    float* bir = beta_I_R + b * stride;
    float* bil = beta_I_L + b * stride;
    float* wcr = W_C_R + b * stride;
    float* wcl = W_C_L + b * stride;
    float* wir = W_I_R + b * stride;
    float* wil = W_I_L + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;
        if (j >= seq_len) continue;

        // Backward for C_R[i,j]
        float beta_cr_ij = bcr[chart_index(i, j, n)];
        float w_cr_ij = wcr[chart_index(i, j, n)];

        if (beta_cr_ij != 0.0f || w_cr_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_term = 0.0f;
                float E_U_child = 0.0f;
                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_term += w * v;
                    E_U_child += w * (ucr[chart_index(i, k, n)] + uir[chart_index(k, j, n)]);
                }

                for (int k = i; k < j; k++) {
                    float v = cr[chart_index(i, k, n)] + ir[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float diff = v - E_term;
                    float U_child = ucr[chart_index(i, k, n)] + uir[chart_index(k, j, n)];
                    float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                    float d_mass = w_cr_ij * w + beta_cr_ij * dw_dT;

                    atomicAdd(&bcr[chart_index(i, k, n)], beta_cr_ij * w);
                    atomicAdd(&bir[chart_index(k, j, n)], beta_cr_ij * w);
                    atomicAdd(&wcr[chart_index(i, k, n)], d_mass);
                    atomicAdd(&wir[chart_index(k, j, n)], d_mass);
                }
            }
        }

        // Backward for C_L[i,j]
        float beta_cl_ij = bcl[chart_index(i, j, n)];
        float w_cl_ij = wcl[chart_index(i, j, n)];

        if (beta_cl_ij != 0.0f || w_cl_ij != 0.0f) {
            float max_v = NINF;
            for (int k = i + 1; k <= j; k++) {
                float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                max_v = fmaxf(max_v, v);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    sum_exp += safe_exp((v - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_term = 0.0f;
                float E_U_child = 0.0f;
                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    E_term += w * v;
                    E_U_child += w * (uil[chart_index(i, k, n)] + ucl[chart_index(k, j, n)]);
                }

                for (int k = i + 1; k <= j; k++) {
                    float v = il[chart_index(i, k, n)] + cl[chart_index(k, j, n)];
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    float diff = v - E_term;
                    float U_child = uil[chart_index(i, k, n)] + ucl[chart_index(k, j, n)];
                    float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                    float d_mass = w_cl_ij * w + beta_cl_ij * dw_dT;

                    atomicAdd(&bil[chart_index(i, k, n)], beta_cl_ij * w);
                    atomicAdd(&bcl[chart_index(k, j, n)], beta_cl_ij * w);
                    atomicAdd(&wil[chart_index(i, k, n)], d_mass);
                    atomicAdd(&wcl[chart_index(k, j, n)], d_mass);
                }
            }
        }
    }
}

__global__ void param_grad_backward_incomplete_kernel(
    const float* __restrict__ C_R,
    const float* __restrict__ C_L,
    const float* __restrict__ U_C_R,
    const float* __restrict__ U_C_L,
    float* __restrict__ beta_C_R,
    float* __restrict__ beta_C_L,
    const float* __restrict__ beta_I_R,
    const float* __restrict__ beta_I_L,
    float* __restrict__ W_C_R,
    float* __restrict__ W_C_L,
    const float* __restrict__ W_I_R,
    const float* __restrict__ W_I_L,
    float* __restrict__ dP_dT,
    const int* __restrict__ lengths,
    int B, int n, float T,
    int span_len
) {
    int b = blockIdx.x;
    size_t stride = (size_t)n * n;

    const float* cr = C_R + b * stride;
    const float* cl = C_L + b * stride;
    const float* ucr = U_C_R + b * stride;
    const float* ucl = U_C_L + b * stride;
    float* bcr = beta_C_R + b * stride;
    float* bcl = beta_C_L + b * stride;
    const float* bir = beta_I_R + b * stride;
    const float* bil = beta_I_L + b * stride;
    float* wcr = W_C_R + b * stride;
    float* wcl = W_C_L + b * stride;
    const float* wir = W_I_R + b * stride;
    const float* wil = W_I_L + b * stride;
    float* dp_dt = dP_dT + b * stride;

    int seq_len = lengths ? lengths[b] : n;
    int num_spans = seq_len - span_len;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len;
        if (j >= seq_len) continue;

        float beta_ir = bir[chart_index(i, j, n)];
        float beta_il = bil[chart_index(i, j, n)];
        float w_ir = wir[chart_index(i, j, n)];
        float w_il = wil[chart_index(i, j, n)];

        // Extract dP/dT
        dp_dt[chart_index(i, j, n)] = w_ir;
        dp_dt[chart_index(j, i, n)] = w_il;

        float beta_combined = beta_ir + beta_il;
        float w_combined = w_ir + w_il;

        if (beta_combined != 0.0f || w_combined != 0.0f) {
            float max_v = NINF;
            for (int k = i; k < j; k++) {
                float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                max_v = fmaxf(max_v, val);
            }

            if (max_v > NINF) {
                float sum_exp = 0.0f;
                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    sum_exp += safe_exp((val - max_v) / T);
                }
                float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

                float E_term = 0.0f;
                float E_U_child = 0.0f;
                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    float w = safe_exp((val - max_v) / T) * inv_sum;
                    E_term += w * val;
                    E_U_child += w * (ucr[chart_index(i, k, n)] + ucl[chart_index(k + 1, j, n)]);
                }

                for (int k = i; k < j; k++) {
                    float val = cr[chart_index(i, k, n)] + cl[chart_index(k + 1, j, n)];
                    float w = safe_exp((val - max_v) / T) * inv_sum;
                    float diff = val - E_term;
                    float U_child = ucr[chart_index(i, k, n)] + ucl[chart_index(k + 1, j, n)];
                    float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                    float d_mass = w_combined * w + beta_combined * dw_dT;

                    atomicAdd(&bcr[chart_index(i, k, n)], beta_combined * w);
                    atomicAdd(&bcl[chart_index(k + 1, j, n)], beta_combined * w);
                    atomicAdd(&wcr[chart_index(i, k, n)], d_mass);
                    atomicAdd(&wcl[chart_index(k + 1, j, n)], d_mass);
                }
            }
        }
    }
}

void param_grad(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* U_C_R,
    float* U_C_L,
    float* U_I_R,
    float* U_I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* W_C_R,
    float* W_C_L,
    float* W_I_R,
    float* W_I_L,
    float* dP_dT,
    const int* lengths,
    int B, int n, float temperature
) {
    cudaStream_t stream = d2p::common::get_cuda_stream();
    int threads = 256;
    size_t elems = (size_t)B * n * n;
    int blocks_init = (elems + threads - 1) / threads;

    param_grad_init_kernel<<<blocks_init, threads, 0, stream>>>(
        U_C_R, U_C_L, U_I_R, U_I_L,
        beta_C_R, beta_C_L, beta_I_R, beta_I_L,
        W_C_R, W_C_L, W_I_R, W_I_L,
        dP_dT, lengths, B, n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Forward: compute U tables
    for (int len = 1; len < n; len++) {
        param_grad_forward_incomplete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, U_C_R, U_C_L,
            U_I_R, U_I_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        param_grad_forward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L, U_C_R, U_C_L, U_I_R, U_I_L,
            U_C_R, U_C_L, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // Backward: compute W tables and extract dP/dT
    for (int len = n - 1; len >= 1; len--) {
        param_grad_backward_complete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, I_R, I_L, U_C_R, U_C_L, U_I_R, U_I_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            W_C_R, W_C_L, W_I_R, W_I_L,
            lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        param_grad_backward_incomplete_kernel<<<B, threads, 0, stream>>>(
            C_R, C_L, U_C_R, U_C_L,
            beta_C_R, beta_C_L, beta_I_R, beta_I_L,
            W_C_R, W_C_L, W_I_R, W_I_L,
            dP_dT, lengths, B, n, temperature, len
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace eisner
} // namespace d2p
