// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft True Damerau-Levenshtein CUDA Kernel Declarations
 *
 * Host-side declarations for CUDA kernels. Implementation is in kernels.cu.
 *
 * Damerau differs from OSA in that transpositions can span variable distances
 * based on character positions, using precomputed trans_src indices.
 *
 * Uses SOFTMIN (minimization) with PINF = 1e30f for positive infinity.
 */

#pragma once

namespace d2p {
namespace damerau {

// Positive infinity for minimization
constexpr float PINF = 1e30f;

/**
 * @brief Forward pass for Soft Damerau-Levenshtein
 *
 * Computes soft edit distance using 4-way transitions:
 * substitute, delete, insert, transpose (with variable distances)
 *
 * @param d_sub_costs Substitution costs [B, L1, L2]
 * @param d_trans_src Transposition source indices [B, L1, L2, 2] (k, l pairs)
 * @param d_alpha Output DP table [B, (L1+1)*(L2+1)]
 * @param d_damerau_score Output scores [B]
 * @param d_lengths Sequence lengths [B, 2]
 * @param ins_cost Insertion cost
 * @param del_cost Deletion cost
 * @param trans_cost Transposition cost
 * @param B Batch size
 * @param max_L1 Maximum sequence 1 length
 * @param max_L2 Maximum sequence 2 length
 * @param T Temperature
 */
void damerau_forward(
    const float* d_sub_costs,
    const int* d_trans_src,
    float* d_alpha,
    float* d_damerau_score,
    const int* d_lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * @brief Backward pass for Soft Damerau-Levenshtein
 *
 * Computes posteriors and parameter gradients via reverse-mode autodiff.
 *
 * @param d_alpha DP table from forward [B, (L1+1)*(L2+1)]
 * @param d_sub_costs Substitution costs [B, L1, L2]
 * @param d_trans_src Transposition source indices [B, L1, L2, 2]
 * @param d_damerau_score Forward scores [B]
 * @param d_beta Output beta table [B, (L1+1)*(L2+1)]
 * @param d_posteriors Output posteriors [B, L1, L2]
 * @param d_grad_T Output temperature gradient [B]
 * @param d_grad_ins Output insertion cost gradient [B]
 * @param d_grad_del Output deletion cost gradient [B]
 * @param d_grad_trans Output transposition cost gradient [B]
 * @param d_lengths Sequence lengths [B, 2]
 * @param ins_cost Insertion cost
 * @param del_cost Deletion cost
 * @param trans_cost Transposition cost
 * @param B Batch size
 * @param max_L1 Maximum sequence 1 length
 * @param max_L2 Maximum sequence 2 length
 * @param T Temperature
 */
void damerau_backward(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
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

/**
 * @brief Hessian-vector product for Soft Damerau-Levenshtein
 *
 * Computes second-order gradients efficiently via forward-over-backward.
 *
 * @param d_alpha DP table from forward [B, (L1+1)*(L2+1)]
 * @param d_sub_costs Substitution costs [B, L1, L2]
 * @param d_trans_src Transposition source indices [B, L1, L2, 2]
 * @param d_damerau_score Forward scores [B]
 * @param d_V Tangent vector [B, L1, L2]
 * @param d_d_alpha Tangent alpha table [B, (L1+1)*(L2+1)]
 * @param d_d_score Tangent score [B]
 * @param d_beta Beta table [B, (L1+1)*(L2+1)]
 * @param d_d_beta Tangent beta table [B, (L1+1)*(L2+1)]
 * @param d_H_scores Output HVP result [B, L1, L2]
 * @param d_lengths Sequence lengths [B, 2]
 * @param ins_cost Insertion cost
 * @param del_cost Deletion cost
 * @param trans_cost Transposition cost
 * @param B Batch size
 * @param max_L1 Maximum sequence 1 length
 * @param max_L2 Maximum sequence 2 length
 * @param T Temperature
 */
void damerau_hvp(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
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

// Parameter types for Damerau param_jacobian
enum DamerauParamType {
    DAMERAU_PARAM_INS = 0,
    DAMERAU_PARAM_DEL = 1,
    DAMERAU_PARAM_TRANS = 2,
    DAMERAU_PARAM_TEMPERATURE = 3
};

void damerau_param_grad(
    const float* d_alpha,
    const float* d_sub_costs,
    const int* d_trans_src,
    const float* d_damerau_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dparam,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float trans_cost, float T,
    int param_type
);

}  // namespace damerau
}  // namespace d2p
