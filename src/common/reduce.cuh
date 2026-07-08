// SPDX-License-Identifier: Apache-2.0
// reduce.cuh - CUDA warp and block reduction primitives
//
// Shared across all d2p CUDA operators. Float32 only.
// All kernels using these reductions MUST have blockDim.x be a multiple of 32.
//
// Usage:
//   #include "common/reduce.cuh"
//   float sum = d2p::common::warp_reduce_sum(val);

#pragma once
#include <cuda_runtime.h>
#include "numerics.cuh"

namespace d2p {
namespace common {

inline constexpr int WARP_SIZE = 32;

// ============================================================================
// IMPORTANT: All kernels must use block sizes that are multiples of WARP_SIZE.
// Use D2P_STATIC_CHECK_BLOCK_SIZE for compile-time validation of constexpr sizes.
// Use D2P_CHECK_BLOCK_SIZE (in torch_utils.h) for runtime validation.
// ============================================================================

#define D2P_STATIC_CHECK_BLOCK_SIZE(block_dim) \
    static_assert(block_dim % 32 == 0, "Block size must be multiple of warp size")

// ============================================================================
// Warp-level reductions (float-only, require full warps)
// These use __shfl_down_sync with full warp mask (0xffffffff)
// ============================================================================

__device__ __forceinline__
float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__
float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

__device__ __forceinline__
float warp_reduce_min(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v = fminf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    return v;
}

// ============================================================================
// Block-level reductions (float-only, require blockDim.x % 32 == 0)
// Use shared memory to combine warp results
// ============================================================================

__device__ __forceinline__
float block_reduce_sum(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // First reduce within warps
    v = warp_reduce_sum(v);

    // Write reduced warp values to shared memory
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    // First warp reduces all warp values
    int num_warps = blockDim.x / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);

    // Barrier before returning so a back-to-back block_reduce_sum in the same
    // block cannot overwrite shared[] (write-after-read) before every thread
    // has finished the read above. Callers already invoke this block-uniformly
    // (required by the __syncthreads above), so all threads reach this barrier.
    __syncthreads();

    return v;
}

__device__ __forceinline__
float block_reduce_max(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    v = warp_reduce_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : NINF;
    if (wid == 0) v = warp_reduce_max(v);

    return v;
}

__device__ __forceinline__
float block_reduce_min(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    v = warp_reduce_min(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    v = (threadIdx.x < num_warps) ? shared[lane] : PINF;
    if (wid == 0) v = warp_reduce_min(v);

    return v;
}

}  // namespace common
}  // namespace d2p
