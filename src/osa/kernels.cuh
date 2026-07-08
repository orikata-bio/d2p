// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft OSA CUDA Kernel Declarations
 *
 * Optimal String Alignment (Restricted Damerau-Levenshtein) using SOFTMIN.
 * Four operations: substitute, insert, delete, transpose (adjacent only).
 */

#pragma once

namespace d2p {
namespace osa {

// ============================================================================
// Constants
// ============================================================================

constexpr float PINF = 1e30f;   // Positive infinity for minimization

// ============================================================================
// CUDA Kernel Function Declarations
// ============================================================================

void osa_forward(
    const float* d_sub_costs,
    const float* d_trans_mask,
    float* d_alpha,
    float* d_osa_score,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

void osa_backward(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    float* d_grad_ins,
    float* d_grad_del,
    float* d_grad_trans,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

void osa_hvp(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

// Parameter types for OSA param_jacobian
enum OSAParamType {
    OSA_PARAM_INS = 0,
    OSA_PARAM_DEL = 1,
    OSA_PARAM_TRANS = 2,
    OSA_PARAM_TEMPERATURE = 3
};

void osa_param_grad(
    const float* d_alpha,
    const float* d_sub_costs,
    const float* d_trans_mask,
    const float* d_osa_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dparam,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float trans_cost, float T,
    int param_type
);

}  // namespace osa
}  // namespace d2p
