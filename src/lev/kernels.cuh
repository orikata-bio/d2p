// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Levenshtein (Edit Distance) CUDA Kernel Declarations
 *
 * Soft Levenshtein uses SOFTMIN (minimization) for differentiable edit distance.
 * Unlike Smith-Waterman which maximizes alignment, Levenshtein minimizes cost.
 *
 * Key features:
 *   - Asymmetric costs: separate ins_cost, del_cost
 *   - Substitution cost from input scores[i,j]
 *   - 3 transitions: substitute (diagonal), delete (up), insert (left)
 *   - Base cases: alpha(0,0)=0, alpha(i,0)=i*del_cost, alpha(0,j)=j*ins_cost
 *   - Score = alpha[L1, L2] (soft edit distance)
 *   - Posteriors = beta * w_sub (substitution weight)
 *
 * Shapes:
 *   scores:      [B, L1, L2]        - substitution costs
 *   alpha:       [B, (L1+1)*(L2+1)] - DP table (row-major, 1-indexed logic)
 *   distance:    [B]                - soft edit distance
 *   posteriors:  [B, L1, L2]        - substitution marginals
 *   grad_ins:    [B]                - insertion cost gradient
 *   grad_del:    [B]                - deletion cost gradient
 *   grad_T:      [B]                - temperature gradient
 */

#pragma once

namespace d2p {
namespace lev {

// ============================================================================
// Constants
// ============================================================================

constexpr float PINF = 1e30f;   // Positive infinity for minimization

// Parameter types for Levenshtein gradients
enum LevParamType {
    LEV_PARAM_INS = 0,
    LEV_PARAM_DEL = 1,
    LEV_PARAM_TEMPERATURE = 2
};

// ============================================================================
// Host Wrapper Function Declarations
// ============================================================================

/**
 * Forward pass: Compute soft edit distance
 *
 * @param d_scores      [B, L1, L2] substitution costs
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table workspace
 * @param d_distance    [B] output soft edit distance
 * @param d_lengths     [B, 2] per-batch sequence lengths
 * @param B             batch size
 * @param max_L1        maximum first sequence length
 * @param max_L2        maximum second sequence length
 * @param ins_cost      insertion cost
 * @param del_cost      deletion cost
 * @param T             temperature
 */
void lev_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_distance,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Backward pass: Compute gradients
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] substitution costs
 * @param d_distance    [B] soft edit distance
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_posteriors  [B, L1, L2] output substitution marginals
 * @param d_grad_ins    [B] output insertion cost gradient
 * @param d_grad_del    [B] output deletion cost gradient
 * @param d_grad_T      [B] output temperature gradient
 * @param d_lengths     [B, 2] per-batch sequence lengths
 */
void lev_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_ins,
    float* d_grad_del,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Hessian-vector product
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] substitution costs
 * @param d_distance    [B] soft edit distance
 * @param d_V           [B, L1, L2] tangent vector
 * @param d_d_alpha     [B, (L1+1)*(L2+1)] tangent DP workspace
 * @param d_d_distance  [B] tangent distance
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_d_beta      [B, (L1+1)*(L2+1)] tangent backward workspace
 * @param d_H_scores    [B, L1, L2] output HVP
 * @param d_lengths     [B, 2] per-batch sequence lengths
 */
void lev_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_distance,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T
);

/**
 * Parameter gradient: dP/d{ins,del,T}
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] substitution costs
 * @param d_distance    [B] soft edit distance
 * @param d_U           [B, (L1+1)*(L2+1)] forward sensitivity workspace
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_W           [B, (L1+1)*(L2+1)] backward sensitivity workspace
 * @param d_dP_dparam   [B, L1, L2] output parameter Jacobian
 * @param d_lengths     [B, 2] per-batch sequence lengths
 * @param param_type    0=ins, 1=del, 2=temperature
 */
void lev_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_distance,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dparam,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float T,
    int param_type
);

}  // namespace lev
}  // namespace d2p
