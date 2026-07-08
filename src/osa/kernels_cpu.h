// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft OSA CPU Kernel Declarations
 *
 * CPU implementation declarations mirroring CUDA interface.
 */

#pragma once

namespace d2p {
namespace osa {
namespace cpu {

// ============================================================================
// Constants
// ============================================================================

constexpr float PINF = 1e30f;   // Positive infinity for minimization

// ============================================================================
// CPU Kernel Function Declarations
// ============================================================================

void osa_forward_cpu(
    const float* sub_costs,
    const float* trans_mask,
    float* alpha,
    float* osa_score,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

void osa_backward_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    const float* osa_score,
    float* beta,
    float* posteriors,
    float* grad_T,
    float* grad_ins,
    float* grad_del,
    float* grad_trans,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

void osa_hvp_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    const float* osa_score,
    const float* V,
    float* d_alpha,
    float* d_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

// Parameter types for OSA param_jacobian
enum OSAParamTypeCPU {
    OSA_PARAM_INS_CPU = 0,
    OSA_PARAM_DEL_CPU = 1,
    OSA_PARAM_TRANS_CPU = 2,
    OSA_PARAM_TEMPERATURE_CPU = 3
};

/**
 * Parameter gradient: dP/d{ins,del,trans,T}
 */
void osa_param_grad_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* sub_costs,   // [B, L1, L2]
    const float* trans_mask,  // [B, L1, L2]
    const float* osa_score,   // [B]
    float* U,                 // [B, (L1+1)*(L2+1)]
    float* beta,              // [B, (L1+1)*(L2+1)]
    float* W,                 // [B, (L1+1)*(L2+1)]
    float* dP_dparam,         // [B, L1, L2]
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float trans_cost, float T,
    int param_type
);

}  // namespace cpu
}  // namespace osa
}  // namespace d2p
