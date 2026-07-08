// SPDX-License-Identifier: Apache-2.0
// cuda_utils.h - CUDA utility functions for d2p
//
// Device guards, grid sizing, stream helpers, launch validation.
// Include this in PyTorch binding files that dispatch to CUDA kernels.
//
// Usage:
//   #include "common/cuda_utils.h"
//   D2P_CUDA_GUARD(scores);
//   int grid = d2p::common::compute_grid_size(n, block_size);

#pragma once
#include <algorithm>              // std::min
#include <cuda_runtime.h>
#include <c10/util/Exception.h>   // TORCH_CHECK
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>  // C10_CUDA_CHECK, C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/CUDAContext.h>

// Host-only headers for the recordStream helpers below. Kept out of device
// (.cu) translation units so nvcc never parses the ATen tensor/allocator API.
#if !defined(__CUDACC__)
#include <initializer_list>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace d2p {
namespace common {

// ============================================================================
// Grid/block sizing utilities
// ============================================================================

inline constexpr int DEFAULT_BLOCK_SIZE = 256;
inline constexpr int MAX_GRID_SIZE = 65535;

// Ceiling division (common pattern for grid sizing)
__host__ __device__ __forceinline__
int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Compute grid size, clamped to max
inline int compute_grid_size(int num_elements, int block_size) {
    int grid = ceil_div(num_elements, block_size);
    return std::min(grid, MAX_GRID_SIZE);
}

// ============================================================================
// Device guard helper
// ============================================================================

// RAII device guard - use at start of every CUDA op to ensure
// we're on the correct device for multi-GPU scenarios
// Usage: D2P_CUDA_GUARD(scores);
#define D2P_CUDA_GUARD(tensor) \
    c10::cuda::CUDAGuard device_guard((tensor).device())

// ============================================================================
// Stream helpers
// ============================================================================

inline cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream();
}

// ----------------------------------------------------------------------------
// Caching-allocator stream tracking (host translation units only)
// ----------------------------------------------------------------------------
//
// recordStream tells the PyTorch caching allocator that a tensor's storage is
// used on the current CUDA stream, so it will not free or reuse that memory
// while d2p kernels (now launched on the current stream, not the default
// stream) are still reading/writing it. This is a correctness requirement once
// launches move off stream 0 and a precondition for CUDA-graph capture.
//
// Guarded out of device (.cu) translation units: kernels.cu includes this
// header only for get_cuda_stream()/the C10 check macros and never calls the
// helpers below, so nvcc never has to parse the ATen tensor/allocator surface.
#if !defined(__CUDACC__)

// Record a single tensor's storage against the current CUDA stream.
inline void record_stream_current(const at::Tensor& t) {
    if (t.defined() && t.is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            t.storage().data_ptr(), at::cuda::getCurrentCUDAStream());
    }
}

// Record several tensors at once: record_streams_current({&a, &b, &c});
inline void record_streams_current(std::initializer_list<const at::Tensor*> tensors) {
    for (const at::Tensor* t : tensors) {
        record_stream_current(*t);
    }
}

#endif  // !defined(__CUDACC__)

// ============================================================================
// Launch validation
// ============================================================================

// Validate launch config before kernel dispatch
inline void check_launch_config(dim3 grid, dim3 block, const char* kernel_name) {
    TORCH_CHECK(grid.x > 0 && grid.y > 0 && grid.z > 0,
        kernel_name, ": grid dimensions must be positive");
    TORCH_CHECK(block.x > 0 && block.y > 0 && block.z > 0,
        kernel_name, ": block dimensions must be positive");
    TORCH_CHECK(block.x * block.y * block.z <= 1024,
        kernel_name, ": block size exceeds 1024 threads");
}

// Simplified 1D launch validation
inline void check_launch_config_1d(int grid, int block, const char* kernel_name) {
    check_launch_config(dim3(grid), dim3(block), kernel_name);
}

}  // namespace common
}  // namespace d2p
