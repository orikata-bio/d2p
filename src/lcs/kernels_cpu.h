// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft LCS CPU Kernel Declarations
 *
 * CPU implementation declarations mirroring CUDA interface.
 */

#pragma once

namespace d2p {
namespace lcs {
namespace cpu {

// ============================================================================
// Constants
// ============================================================================

constexpr float NINF = -1e30f;   // Negative infinity for maximization

// ============================================================================
// CPU Kernel Function Declarations
// ============================================================================

void lcs_forward_cpu(
    const float* scores,
    float* alpha,
    float* lcs_score,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

void lcs_backward_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    float* beta,
    float* posteriors,
    float* grad_T,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

void lcs_hvp_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    const float* V,
    float* d_alpha,
    float* d_lcs_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

void lcs_param_grad_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    float* U,
    float* beta,
    float* W,
    float* dP_dT,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
);

}  // namespace cpu
}  // namespace lcs
}  // namespace d2p
