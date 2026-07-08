// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft True Damerau-Levenshtein CPU Kernel Declarations
 *
 * CPU implementations that mirror the CUDA interface for seamless dispatch.
 *
 * Damerau differs from OSA in that transpositions can span variable distances
 * based on character positions, using precomputed trans_src indices.
 */

#pragma once

namespace d2p {
namespace damerau {
namespace cpu {

// Positive infinity for minimization
constexpr float PINF = 1e30f;

/**
 * @brief Forward pass for Soft Damerau-Levenshtein (CPU)
 */
void damerau_forward_cpu(
    const float* sub_costs,
    const int* trans_src,
    float* alpha,
    float* damerau_score,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * @brief Backward pass for Soft Damerau-Levenshtein (CPU)
 */
void damerau_backward_cpu(
    const float* alpha,
    const float* sub_costs,
    const int* trans_src,
    const float* damerau_score,
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

/**
 * @brief Hessian-vector product for Soft Damerau-Levenshtein (CPU)
 */
void damerau_hvp_cpu(
    const float* alpha,
    const float* sub_costs,
    const int* trans_src,
    const float* damerau_score,
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

// Parameter types for Damerau param_jacobian
enum DamerauParamTypeCPU {
    DAMERAU_PARAM_INS_CPU = 0,
    DAMERAU_PARAM_DEL_CPU = 1,
    DAMERAU_PARAM_TRANS_CPU = 2,
    DAMERAU_PARAM_TEMPERATURE_CPU = 3
};

/**
 * Parameter gradient: dP/d{ins,del,trans,T}
 */
void damerau_param_grad_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* sub_costs,   // [B, L1, L2]
    const int* trans_src,     // [B, L1, L2, 2]
    const float* damerau_score, // [B]
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
}  // namespace damerau
}  // namespace d2p
