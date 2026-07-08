// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Levenshtein (Edit Distance) CPU Kernel Declarations
 *
 * CPU implementations that mirror the CUDA kernel interface for seamless dispatch.
 */

#pragma once

#include <cstdint>

namespace d2p {
namespace lev {

// Parameter types for Levenshtein gradients (matches CUDA)
enum LevParamTypeCPU {
    LEV_PARAM_INS_CPU = 0,
    LEV_PARAM_DEL_CPU = 1,
    LEV_PARAM_TEMPERATURE_CPU = 2
};

/**
 * Forward pass: Compute soft edit distance
 */
void lev_forward_cpu(
    const float* scores,      // [B, L1, L2]
    float* alpha,             // [B, (L1+1)*(L2+1)]
    float* distance,          // [B]
    const int* lengths,       // [B, 2]
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Backward pass: Compute gradients
 */
void lev_backward_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* distance,    // [B]
    float* beta,              // [B, (L1+1)*(L2+1)]
    float* posteriors,        // [B, L1, L2]
    float* grad_ins,          // [B]
    float* grad_del,          // [B]
    float* grad_T,            // [B]
    const int* lengths,       // [B, 2]
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Hessian-vector product
 */
void lev_hvp_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* distance,    // [B]
    const float* V,           // [B, L1, L2]
    float* d_alpha,           // [B, (L1+1)*(L2+1)]
    float* d_distance,        // [B]
    float* beta,              // [B, (L1+1)*(L2+1)]
    float* d_beta,            // [B, (L1+1)*(L2+1)]
    float* H_scores,          // [B, L1, L2]
    const int* lengths,       // [B, 2]
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Parameter gradient: dP/d{ins,del,T}
 */
void lev_param_grad_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* distance,    // [B]
    float* U,                 // [B, (L1+1)*(L2+1)]
    float* beta,              // [B, (L1+1)*(L2+1)]
    float* W,                 // [B, (L1+1)*(L2+1)]
    float* dP_dparam,         // [B, L1, L2]
    const int* lengths,       // [B, 2]
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T,
    int param_type
);

}  // namespace lev
}  // namespace d2p
