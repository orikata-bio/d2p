// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cu
 * @brief Soft CKY CUDA Kernel Implementations
 *
 * Differentiable CKY parsing with span-based parallelization.
 * Uses logsumexp for maximization (like SW/NW, unlike DTW).
 */

#include "kernels.cuh"
#include "common/cuda_utils.h"
#include "common/numerics.cuh"
#include "common/reduce.cuh"

using namespace d2p::common;

namespace {

constexpr unsigned CKY_FULL_MASK = 0xffffffffu;
constexpr int CKY_WARP_SIZE = 32;

__device__ __forceinline__ float cky_warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = CKY_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(CKY_FULL_MASK, v, offset));
    }
    return v;
}

__device__ __forceinline__ float cky_warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = CKY_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(CKY_FULL_MASK, v, offset);
    }
    return v;
}

__device__ __forceinline__ float cky_temperature_value(
    const float* __restrict__ d_T,
    int b,
    int n,
    int i,
    int j,
    bool T_is_scalar
) {
    if (T_is_scalar) {
        return d_T[0];
    }
    return d_T[(size_t)b * n * n + (size_t)i * n + j];
}

__device__ __forceinline__ size_t cky_n3_index(int n, int i, int k, int j) {
    return (size_t)i * n * n + (size_t)k * n + j;
}

__device__ __forceinline__ float cky_span_max(
    const float* __restrict__ z,
    const float* __restrict__ merge,
    int n, int i, int j, int lane
) {
    float max_v = NINF;
    for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
        int k = k_base + lane;
        float v = NINF;
        if (k < j) {
            float left = z[i * n + k];
            float right = z[(k + 1) * n + j];
            float ms = merge[cky_n3_index(n, i, k, j)];
            v = left + right + ms;
        }
        float chunk_max = cky_warp_reduce_max(v);
        max_v = fmaxf(max_v, __shfl_sync(CKY_FULL_MASK, chunk_max, 0));
    }
    return max_v;
}

__device__ __forceinline__ float cky_span_sumexp(
    const float* __restrict__ z,
    const float* __restrict__ merge,
    int n, int i, int j, float max_v, float T, int lane
) {
    float sum_exp = 0.0f;
    for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
        int k = k_base + lane;
        float contrib = 0.0f;
        if (k < j) {
            float left = z[i * n + k];
            float right = z[(k + 1) * n + j];
            float ms = merge[cky_n3_index(n, i, k, j)];
            float v = left + right + ms;
            contrib = safe_exp((v - max_v) / T);
        }
        float chunk_sum = cky_warp_reduce_sum(contrib);
        sum_exp += __shfl_sync(CKY_FULL_MASK, chunk_sum, 0);
    }
    return sum_exp;
}

}  // namespace

// ============================================================================
// FORWARD PASS (Inside Algorithm)
// ============================================================================

// Initialize Z: leaves = leaf_scores, all others = -inf
__global__ void cky_init_kernel(
    const float* __restrict__ leaf_scores,  // [B, n]
    float* __restrict__ Z,                  // [B, n, n]
    int B, int n
) {
    size_t Z_stride = (size_t)n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * Z_stride;
    if (idx >= total) return;

    size_t b = idx / Z_stride;
    size_t rem = idx - b * Z_stride;
    int i = rem / n;
    int j = rem % n;

    if (i == j) {
        Z[idx] = leaf_scores[b * n + i];
    } else {
        Z[idx] = NINF;
    }
}

__global__ void cky_forward_persistent_kernel(
    const float* __restrict__ merge_scores,  // [B, n, n, n]
    const float* __restrict__ leaf_scores,   // [B, n]
    float* __restrict__ Z,                   // [B, n, n]
    float* __restrict__ partition,           // [B]
    const float* __restrict__ d_T,           // [1] or [B, n, n]
    int B, int n,
    bool T_is_scalar
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    const float* leaf = leaf_scores + (size_t)b * n;

    for (int idx = threadIdx.x; idx < n * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        z[idx] = (i == j) ? leaf[i] : NINF;
    }
    __syncthreads();

    for (int span_len = 2; span_len <= n; span_len++) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;
            float T = cky_temperature_value(d_T, b, n, i, j, T_is_scalar);

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) {
                if (lane == 0) z[i * n + j] = NINF;
                continue;
            }

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            if (lane == 0) {
                z[i * n + j] = max_v + T * logf(sum_exp);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partition[b] = z[n - 1];
    }
}

// ============================================================================
// BACKWARD PASS (Outside Algorithm + Gradients)
// ============================================================================

__global__ void cky_init_beta_kernel(
    float* __restrict__ beta,
    float* __restrict__ Pcond,
    float* __restrict__ Pjoint,
    float* __restrict__ grad_merge,
    float* __restrict__ grad_leaf,
    float* __restrict__ grad_T,
    int B, int n,
    bool T_is_scalar
) {
    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_Z = (size_t)B * Z_stride;
    size_t total_merge = (size_t)B * merge_stride;

    if (idx < total_Z) {
        size_t b = idx / Z_stride;
        size_t rem = idx - b * Z_stride;
        int i = rem / n;
        int j = rem % n;
        beta[idx] = (i == 0 && j == n - 1) ? 1.0f : 0.0f;
    }

    if (idx < total_merge) {
        Pcond[idx] = 0.0f;
        Pjoint[idx] = 0.0f;
        grad_merge[idx] = 0.0f;
    }

    if (idx < (size_t)B * n) {
        grad_leaf[idx] = 0.0f;
    }

    size_t total_T = T_is_scalar ? (size_t)B : (size_t)B * n * n;
    if (idx < total_T) {
        grad_T[idx] = 0.0f;
    }
}

__global__ void cky_compute_pcond_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    float* __restrict__ Pcond,
    const float* __restrict__ d_T,
    int B, int n,
    bool T_is_scalar,
    int span_len
) {
    int b = blockIdx.x;
    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + b * Z_stride;
    const float* merge = merge_scores + b * merge_stride;
    float* pc = Pcond + b * merge_stride;

    int num_spans = n - span_len + 1;
    if (num_spans <= 0) return;

    for (int t = threadIdx.x; t < num_spans; t += blockDim.x) {
        int i = t;
        int j = i + span_len - 1;
        float T = cky_temperature_value(d_T, b, n, i, j, T_is_scalar);

        float max_v = NINF;
        for (int k = i; k < j; k++) {
            float left = z[i * n + k];
            float right = z[(k + 1) * n + j];
            float ms = merge[cky_n3_index(n, i, k, j)];
            float v = left + right + ms;
            max_v = fmaxf(max_v, v);
        }

        if (max_v <= NINF) continue;

        float sum_exp = 0.0f;
        for (int k = i; k < j; k++) {
            float left = z[i * n + k];
            float right = z[(k + 1) * n + j];
            float ms = merge[cky_n3_index(n, i, k, j)];
            float v = left + right + ms;
            sum_exp += safe_exp((v - max_v) / T);
        }

        float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

        for (int k = i; k < j; k++) {
            float left = z[i * n + k];
            float right = z[(k + 1) * n + j];
            float ms = merge[cky_n3_index(n, i, k, j)];
            float v = left + right + ms;
            float w = safe_exp((v - max_v) / T) * inv_sum;
            pc[cky_n3_index(n, i, k, j)] = w;
        }
    }
}

__global__ void cky_compute_pcond_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    float* __restrict__ Pcond,
    const float* __restrict__ d_T,
    int B, int n,
    bool T_is_scalar
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    float* pc = Pcond + (size_t)b * merge_stride;

    for (int span_len = 2; span_len <= n; span_len++) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;
            float T = cky_temperature_value(d_T, b, n, i, j, T_is_scalar);

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) continue;

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    pc[cky_n3_index(n, i, k, j)] = safe_exp((v - max_v) / T) * inv_sum;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void cky_backward_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    const float* __restrict__ Pcond,
    float* __restrict__ beta,
    float* __restrict__ Pjoint,
    float* __restrict__ grad_merge,
    float* __restrict__ grad_T,
    const float* __restrict__ d_T,
    int B, int n,
    bool T_is_scalar
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    const float* pc = Pcond + (size_t)b * merge_stride;
    float* be = beta + (size_t)b * Z_stride;
    float* pj = Pjoint + (size_t)b * merge_stride;
    float* gm = grad_merge + (size_t)b * merge_stride;

    for (int span_len = n; span_len >= 2; span_len--) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;
            float T = cky_temperature_value(d_T, b, n, i, j, T_is_scalar);

            float beta_ij = be[i * n + j];
            if (beta_ij == 0.0f) continue;

            float e_term_partial = 0.0f;
            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float p = pc[cky_n3_index(n, i, k, j)];
                    float mass = beta_ij * p;

                    pj[cky_n3_index(n, i, k, j)] = mass;
                    gm[cky_n3_index(n, i, k, j)] = mass;

                    atomicAdd(&be[i * n + k], mass);
                    atomicAdd(&be[(k + 1) * n + j], mass);

                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    e_term_partial += p * (left + right + ms);
                }
            }

            float e_term = cky_warp_reduce_sum(e_term_partial);
            e_term = __shfl_sync(CKY_FULL_MASK, e_term, 0);
            if (lane == 0) {
                float Zij = z[i * n + j];
                float grad_T_ij = beta_ij * (Zij - e_term) / T;
                if (T_is_scalar) {
                    atomicAdd(&grad_T[b], grad_T_ij);
                } else {
                    grad_T[(size_t)b * n * n + (size_t)i * n + j] = grad_T_ij;
                }
            }
        }
        __syncthreads();
    }
}

__global__ void cky_finalize_grad_leaf_kernel(
    const float* __restrict__ beta,
    float* __restrict__ grad_leaf,
    int B, int n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * n) return;

    size_t b = idx / n;
    int i = idx % n;

    size_t chart_idx = b * n * n + (size_t)i * n + i;
    grad_leaf[idx] = beta[chart_idx];
}

// ============================================================================
// HVP (Hessian-Vector Product)
// ============================================================================

__global__ void cky_hvp_init_kernel(
    const float* __restrict__ V_leaf,
    float* __restrict__ d_Z,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ HVP_merge,
    float* __restrict__ HVP_leaf,
    int B, int n
) {
    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (size_t)B * Z_stride) {
        size_t b = idx / Z_stride;
        size_t rem = idx - b * Z_stride;
        int i = rem / n;
        int j = rem % n;
        d_Z[idx] = (i == j) ? V_leaf[b * n + i] : 0.0f;
        beta[idx] = (i == 0 && j == n - 1) ? 1.0f : 0.0f;
        d_beta[idx] = 0.0f;
    }

    if (idx < (size_t)B * merge_stride) {
        HVP_merge[idx] = 0.0f;
    }
    if (idx < (size_t)B * n) {
        HVP_leaf[idx] = 0.0f;
    }
}

__global__ void cky_hvp_forward_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    const float* __restrict__ V_merge,
    float* __restrict__ d_Z,
    float* __restrict__ d_partition,
    int B, int n, float T
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    const float* v_merge = V_merge + (size_t)b * merge_stride;
    float* dz = d_Z + (size_t)b * Z_stride;

    for (int span_len = 2; span_len <= n; span_len++) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) {
                if (lane == 0) dz[i * n + j] = 0.0f;
                continue;
            }

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

            float partial = 0.0f;
            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;

                    float d_left = dz[i * n + k];
                    float d_right = dz[(k + 1) * n + j];
                    float d_ms = v_merge[cky_n3_index(n, i, k, j)];
                    partial += w * (d_left + d_right + d_ms);
                }
            }

            float d_zij = cky_warp_reduce_sum(partial);
            if (lane == 0) dz[i * n + j] = d_zij;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_partition[b] = dz[n - 1];
    }
}

__global__ void cky_hvp_backward_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    const float* __restrict__ V_merge,
    const float* __restrict__ d_Z,
    float* __restrict__ beta,
    float* __restrict__ d_beta,
    float* __restrict__ HVP_merge,
    int B, int n, float T
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    const float* v_merge = V_merge + (size_t)b * merge_stride;
    const float* dz = d_Z + (size_t)b * Z_stride;
    float* be = beta + (size_t)b * Z_stride;
    float* dbe = d_beta + (size_t)b * Z_stride;
    float* hm = HVP_merge + (size_t)b * merge_stride;

    for (int span_len = n; span_len >= 2; span_len--) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;

            float beta_ij = be[i * n + j];
            float d_beta_ij = dbe[i * n + j];

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) continue;

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

            float partial_e = 0.0f;
            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;

                    float d_left = dz[i * n + k];
                    float d_right = dz[(k + 1) * n + j];
                    float d_ms = v_merge[cky_n3_index(n, i, k, j)];
                    partial_e += w * (d_left + d_right + d_ms);
                }
            }
            float E_d_term = cky_warp_reduce_sum(partial_e);
            E_d_term = __shfl_sync(CKY_FULL_MASK, E_d_term, 0);

            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;

                    float d_left = dz[i * n + k];
                    float d_right = dz[(k + 1) * n + j];
                    float d_ms = v_merge[cky_n3_index(n, i, k, j)];
                    float d_term = d_left + d_right + d_ms;
                    float d_w = w * (d_term - E_d_term) / T;

                    float mass = beta_ij * w;
                    float d_mass = d_beta_ij * w + beta_ij * d_w;

                    hm[cky_n3_index(n, i, k, j)] = d_mass;
                    atomicAdd(&be[i * n + k], mass);
                    atomicAdd(&be[(k + 1) * n + j], mass);
                    atomicAdd(&dbe[i * n + k], d_mass);
                    atomicAdd(&dbe[(k + 1) * n + j], d_mass);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void cky_hvp_finalize_leaf_kernel(
    const float* __restrict__ d_beta,
    float* __restrict__ HVP_leaf,
    int B, int n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * n) return;

    size_t b = idx / n;
    int i = idx % n;

    size_t chart_idx = b * n * n + (size_t)i * n + i;
    HVP_leaf[idx] = d_beta[chart_idx];
}

// ============================================================================
// PARAM GRAD (dP/dT)
// ============================================================================

__global__ void cky_param_grad_init_kernel(
    float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT_merge,
    float* __restrict__ dP_dT_leaf,
    int B, int n
) {
    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (size_t)B * Z_stride) {
        size_t b = idx / Z_stride;
        size_t rem = idx - b * Z_stride;
        int i = rem / n;
        int j = rem % n;
        U[idx] = 0.0f;
        beta[idx] = (i == 0 && j == n - 1) ? 1.0f : 0.0f;
        W[idx] = 0.0f;
    }

    if (idx < (size_t)B * merge_stride) {
        dP_dT_merge[idx] = 0.0f;
    }
    if (idx < (size_t)B * n) {
        dP_dT_leaf[idx] = 0.0f;
    }
}

__global__ void cky_param_grad_forward_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    float* __restrict__ U,
    int B, int n, float T
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    float* u = U + (size_t)b * Z_stride;

    for (int span_len = 2; span_len <= n; span_len++) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) {
                if (lane == 0) u[i * n + j] = 0.0f;
                continue;
            }

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

            float partial_term = 0.0f;
            float partial_u = 0.0f;
            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    partial_term += w * v;
                    partial_u += w * (u[i * n + k] + u[(k + 1) * n + j]);
                }
            }

            float E_term = cky_warp_reduce_sum(partial_term);
            float E_U = cky_warp_reduce_sum(partial_u);
            if (lane == 0) {
                float Zij = z[i * n + j];
                u[i * n + j] = (Zij - E_term) / T + E_U;
            }
        }
        __syncthreads();
    }
}

__global__ void cky_param_grad_backward_persistent_kernel(
    const float* __restrict__ Z,
    const float* __restrict__ merge_scores,
    const float* __restrict__ U,
    float* __restrict__ beta,
    float* __restrict__ W,
    float* __restrict__ dP_dT_merge,
    int B, int n, float T
) {
    int b = blockIdx.x;
    int lane = threadIdx.x & (CKY_WARP_SIZE - 1);
    int warp = threadIdx.x / CKY_WARP_SIZE;
    int warps_per_block = blockDim.x / CKY_WARP_SIZE;
    if (b >= B) return;

    size_t Z_stride = (size_t)n * n;
    size_t merge_stride = (size_t)n * n * n;

    const float* z = Z + (size_t)b * Z_stride;
    const float* merge = merge_scores + (size_t)b * merge_stride;
    const float* u = U + (size_t)b * Z_stride;
    float* be = beta + (size_t)b * Z_stride;
    float* w_buf = W + (size_t)b * Z_stride;
    float* dpm = dP_dT_merge + (size_t)b * merge_stride;

    for (int span_len = n; span_len >= 2; span_len--) {
        int num_spans = n - span_len + 1;
        for (int span_idx = warp; span_idx < num_spans; span_idx += warps_per_block) {
            int i = span_idx;
            int j = i + span_len - 1;

            float beta_ij = be[i * n + j];
            float w_ij = w_buf[i * n + j];

            float max_v = cky_span_max(z, merge, n, i, j, lane);
            if (max_v <= NINF) continue;

            float sum_exp = cky_span_sumexp(z, merge, n, i, j, max_v, T, lane);
            float inv_sum = (sum_exp > 1e-20f) ? 1.0f / sum_exp : 0.0f;

            float partial_term = 0.0f;
            float partial_dterm = 0.0f;
            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;
                    partial_term += w * v;
                    partial_dterm += w * (u[i * n + k] + u[(k + 1) * n + j]);
                }
            }
            float E_term = cky_warp_reduce_sum(partial_term);
            float E_d_term = cky_warp_reduce_sum(partial_dterm);
            E_term = __shfl_sync(CKY_FULL_MASK, E_term, 0);
            E_d_term = __shfl_sync(CKY_FULL_MASK, E_d_term, 0);

            for (int k_base = i; k_base < j; k_base += CKY_WARP_SIZE) {
                int k = k_base + lane;
                if (k < j) {
                    float left = z[i * n + k];
                    float right = z[(k + 1) * n + j];
                    float ms = merge[cky_n3_index(n, i, k, j)];
                    float v = left + right + ms;
                    float w = safe_exp((v - max_v) / T) * inv_sum;

                    float d_term = u[i * n + k] + u[(k + 1) * n + j];
                    float diff = v - E_term;
                    float dw_dT = w * (-diff / (T * T) + (d_term - E_d_term) / T);

                    float mass = beta_ij * w;
                    float d_mass = w_ij * w + beta_ij * dw_dT;
                    dpm[cky_n3_index(n, i, k, j)] = d_mass;

                    atomicAdd(&be[i * n + k], mass);
                    atomicAdd(&be[(k + 1) * n + j], mass);
                    atomicAdd(&w_buf[i * n + k], d_mass);
                    atomicAdd(&w_buf[(k + 1) * n + j], d_mass);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void cky_param_grad_finalize_leaf_kernel(
    const float* __restrict__ W,
    float* __restrict__ dP_dT_leaf,
    int B, int n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * n) return;

    size_t b = idx / n;
    int i = idx % n;

    size_t chart_idx = b * n * n + (size_t)i * n + i;
    dP_dT_leaf[idx] = W[chart_idx];
}

// ============================================================================
// Host Wrappers
// ============================================================================

void cky_forward(
    const float* d_merge_scores,
    const float* d_leaf_scores,
    float* d_Z,
    float* d_partition,
    const float* d_T,
    int B, int n,
    bool T_is_scalar
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    cky_forward_persistent_kernel<<<B, threads, 0, stream>>>(
        d_merge_scores, d_leaf_scores, d_Z, d_partition, d_T, B, n, T_is_scalar
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void cky_backward(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    float* d_beta,
    float* d_Pcond,
    float* d_Pjoint,
    float* d_grad_merge,
    float* d_grad_leaf,
    float* d_grad_T,
    const float* d_T,
    int B, int n,
    bool T_is_scalar
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t merge_elems = (size_t)B * n * n * n;
    int blocks_init = (merge_elems + threads - 1) / threads;

    cky_init_beta_kernel<<<blocks_init, threads, 0, stream>>>(
        d_beta, d_Pcond, d_Pjoint, d_grad_merge, d_grad_leaf, d_grad_T, B, n, T_is_scalar
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cky_compute_pcond_persistent_kernel<<<B, threads, 0, stream>>>(
        d_Z, d_merge_scores, d_Pcond, d_T, B, n, T_is_scalar
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cky_backward_persistent_kernel<<<B, threads, 0, stream>>>(
        d_Z, d_merge_scores, d_Pcond, d_beta, d_Pjoint, d_grad_merge, d_grad_T, d_T, B, n, T_is_scalar
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_leaf = (B * n + threads - 1) / threads;
    cky_finalize_grad_leaf_kernel<<<blocks_leaf, threads, 0, stream>>>(d_beta, d_grad_leaf, B, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void cky_hvp(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    const float* d_V_merge,
    const float* d_V_leaf,
    float* d_d_Z,
    float* d_d_partition,
    float* d_beta,
    float* d_d_beta,
    float* d_HVP_merge,
    float* d_HVP_leaf,
    int B, int n, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t merge_elems = (size_t)B * n * n * n;
    int blocks_init = (merge_elems + threads - 1) / threads;

    cky_hvp_init_kernel<<<blocks_init, threads, 0, stream>>>(
        d_V_leaf, d_d_Z, d_beta, d_d_beta, d_HVP_merge, d_HVP_leaf, B, n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cky_hvp_forward_persistent_kernel<<<B, threads, 0, stream>>>(
        d_Z, d_merge_scores, d_V_merge, d_d_Z, d_d_partition, B, n, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cky_hvp_backward_persistent_kernel<<<B, threads, 0, stream>>>(
        d_Z, d_merge_scores, d_V_merge, d_d_Z, d_beta, d_d_beta, d_HVP_merge, B, n, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_leaf = (B * n + threads - 1) / threads;
    cky_hvp_finalize_leaf_kernel<<<blocks_leaf, threads, 0, stream>>>(d_d_beta, d_HVP_leaf, B, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

void cky_param_grad(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT_merge,
    float* d_dP_dT_leaf,
    int B, int n, float T
) {
    int threads = 256;
    cudaStream_t stream = d2p::common::get_cuda_stream();
    size_t merge_elems = (size_t)B * n * n * n;
    int blocks_init = (merge_elems + threads - 1) / threads;

    cky_param_grad_init_kernel<<<blocks_init, threads, 0, stream>>>(
        d_U, d_beta, d_W, d_dP_dT_merge, d_dP_dT_leaf, B, n
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    cky_param_grad_forward_persistent_kernel<<<B, threads, 0, stream>>>(d_Z, d_merge_scores, d_U, B, n, T);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cky_param_grad_backward_persistent_kernel<<<B, threads, 0, stream>>>(
        d_Z, d_merge_scores, d_U, d_beta, d_W, d_dP_dT_merge, B, n, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    int blocks_leaf = (B * n + threads - 1) / threads;
    cky_param_grad_finalize_leaf_kernel<<<blocks_leaf, threads, 0, stream>>>(d_W, d_dP_dT_leaf, B, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}
