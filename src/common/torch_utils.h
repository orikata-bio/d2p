// SPDX-License-Identifier: Apache-2.0
// torch_utils.h - PyTorch validation macros and helpers for d2p
//
// Input validation, lengths helpers, common checks.
// Include this in PyTorch binding files.
//
// Usage:
//   #include "common/torch_utils.h"
//   D2P_CHECK_INPUT_CUDA(scores);

#pragma once
#include <torch/torch.h>

namespace d2p {
namespace common {

// ============================================================================
// Block size validation (runtime) - use before kernel launch
// See reduce.cuh for D2P_STATIC_CHECK_BLOCK_SIZE (compile-time)
// ============================================================================

#define D2P_CHECK_BLOCK_SIZE(block_dim) \
    TORCH_CHECK((block_dim) % 32 == 0, \
        "Block size must be multiple of warp size (32), got ", block_dim)

// ============================================================================
// Input validation macros - device-agnostic and device-specific
// ============================================================================

// Device-agnostic checks (work for both CUDA and CPU)
#define D2P_CHECK_CONTIGUOUS(x) \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define D2P_CHECK_DIM(x, d) \
    TORCH_CHECK((x).dim() == (d), #x " must be " #d "D, got ", (x).dim(), "D")

// Dtype checks - current: float32 only
// NOTE: When adding FP16/BF16/FP64 support, replace D2P_CHECK_FLOAT with
// D2P_CHECK_FLOATING and update the error message accordingly.
#define D2P_CHECK_FLOAT(x) \
    TORCH_CHECK((x).scalar_type() == torch::kFloat32, \
        #x " must be float32, got ", (x).scalar_type())

// Future: accept all supported floating types
// #define D2P_CHECK_FLOATING(x) \
//     TORCH_CHECK((x).is_floating_point(), #x " must be a floating point tensor"); \
//     TORCH_CHECK((x).scalar_type() == torch::kFloat32 || \
//                 (x).scalar_type() == torch::kFloat64 || \
//                 (x).scalar_type() == torch::kFloat16 || \
//                 (x).scalar_type() == torch::kBFloat16, \
//                 #x " must be float32, float64, float16, or bfloat16")

// Device-specific checks
#define D2P_CHECK_CUDA(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define D2P_CHECK_CPU(x) \
    TORCH_CHECK(!(x).is_cuda(), #x " must be a CPU tensor")

// Combined checks - use these at function entry
#define D2P_CHECK_INPUT_CUDA(x) \
    D2P_CHECK_CUDA(x); D2P_CHECK_CONTIGUOUS(x); D2P_CHECK_FLOAT(x)

#define D2P_CHECK_INPUT_CPU(x) \
    D2P_CHECK_CPU(x); D2P_CHECK_CONTIGUOUS(x); D2P_CHECK_FLOAT(x)

// ============================================================================
// Lengths validation
// ============================================================================

// 2D [B, 2] for pairwise ops (SW, NW, DTW, edit distances, MAS)
#define D2P_CHECK_LENGTHS_2D(lengths, B, device) \
    TORCH_CHECK((lengths).scalar_type() == torch::kInt32, \
        "lengths must be int32, got ", (lengths).scalar_type()); \
    TORCH_CHECK((lengths).dim() == 2 && (lengths).size(1) == 2, \
        "lengths must be [B, 2], got ", (lengths).sizes()); \
    TORCH_CHECK((lengths).size(0) == (B), \
        "lengths batch size mismatch: expected ", B, ", got ", (lengths).size(0)); \
    TORCH_CHECK((lengths).device() == (device), \
        "lengths must be on same device as scores, got ", (lengths).device(), " vs ", device)

// 1D [B] for single-sequence ops (Eisner, CKY)
#define D2P_CHECK_LENGTHS_1D(lengths, B, device) \
    TORCH_CHECK((lengths).scalar_type() == torch::kInt32, \
        "lengths must be int32, got ", (lengths).scalar_type()); \
    TORCH_CHECK((lengths).dim() == 1, \
        "lengths must be [B], got ", (lengths).sizes()); \
    TORCH_CHECK((lengths).size(0) == (B), \
        "lengths batch size mismatch: expected ", B, ", got ", (lengths).size(0)); \
    TORCH_CHECK((lengths).device() == (device), \
        "lengths must be on same device as scores, got ", (lengths).device(), " vs ", device)

// ============================================================================
// Helper functions
// ============================================================================

// Create default lengths tensor (all sequences use full dimensions)
// For pairwise ops: [B, 2] where each row is [L1, L2]
inline torch::Tensor make_default_lengths_2d(
    int64_t B, int64_t L1, int64_t L2,
    torch::Device device
) {
    auto lengths = torch::empty({B, 2}, torch::TensorOptions().dtype(torch::kInt32));
    auto acc = lengths.accessor<int32_t, 2>();
    for (int64_t b = 0; b < B; b++) {
        acc[b][0] = static_cast<int32_t>(L1);
        acc[b][1] = static_cast<int32_t>(L2);
    }
    return lengths.to(device);
}

// For single-sequence ops: [B] where each element is N
inline torch::Tensor make_default_lengths_1d(
    int64_t B, int64_t N,
    torch::Device device
) {
    return torch::full({B}, static_cast<int32_t>(N),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
}

}  // namespace common
}  // namespace d2p
