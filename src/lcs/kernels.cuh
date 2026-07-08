// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft LCS (Longest Common Subsequence) CUDA Kernel Declarations
 *
 * Soft LCS uses SOFTMAX (maximization) for differentiable longest common subsequence.
 * Only temperature parameter (no gap costs - skips are free).
 *
 * Key features:
 *   - Only temperature parameter (no gap costs)
 *   - Match score from input scores[i,j]
 *   - 3 transitions: match (diagonal), skip seq1 (up), skip seq2 (left)
 *   - Base cases: alpha(0,0)=0, alpha(i,0)=0, alpha(0,j)=0
 *   - Score = alpha[L1, L2] (soft LCS length)
 *   - Posteriors = beta * w_match (match weight)
 *
 * Shapes:
 *   scores:      [B, L1, L2]        - match scores (1=match, 0=mismatch)
 *   alpha:       [B, (L1+1)*(L2+1)] - DP table (row-major, 1-indexed logic)
 *   lcs_score:   [B]                - soft LCS score
 *   posteriors:  [B, L1, L2]        - match marginals
 *   grad_T:      [B]                - temperature gradient
 */

#pragma once

namespace d2p {
namespace lcs {

// ============================================================================
// Constants
// ============================================================================

constexpr float NINF = -1e30f;   // Negative infinity for maximization

// ============================================================================
// Host Wrapper Function Declarations
// ============================================================================

/**
 * Forward pass: Compute soft LCS score
 *
 * @param d_scores      [B, L1, L2] match scores
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table workspace
 * @param d_lcs_score   [B] output soft LCS score
 * @param d_lengths     [B, 2] per-batch sequence lengths
 * @param B             batch size
 * @param max_L1        maximum first sequence length
 * @param max_L2        maximum second sequence length
 * @param T             temperature
 */
void lcs_forward(
    const float* d_scores,
    float* d_alpha,
    float* d_lcs_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * Backward pass: Compute gradients
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] match scores
 * @param d_lcs_score   [B] soft LCS score
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_posteriors  [B, L1, L2] output match marginals
 * @param d_grad_T      [B] output temperature gradient
 * @param d_lengths     [B, 2] per-batch sequence lengths
 */
void lcs_backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * Hessian-vector product
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] match scores
 * @param d_lcs_score   [B] soft LCS score
 * @param d_V           [B, L1, L2] tangent vector
 * @param d_d_alpha     [B, (L1+1)*(L2+1)] tangent DP workspace
 * @param d_d_lcs_score [B] tangent score
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_d_beta      [B, (L1+1)*(L2+1)] tangent backward workspace
 * @param d_H_scores    [B, L1, L2] output HVP
 * @param d_lengths     [B, 2] per-batch sequence lengths
 */
void lcs_hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_lcs_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
);

/**
 * Parameter gradient: dP/dT
 *
 * @param d_alpha       [B, (L1+1)*(L2+1)] DP table from forward
 * @param d_scores      [B, L1, L2] match scores
 * @param d_lcs_score   [B] soft LCS score
 * @param d_U           [B, (L1+1)*(L2+1)] forward sensitivity workspace
 * @param d_beta        [B, (L1+1)*(L2+1)] backward DP workspace
 * @param d_W           [B, (L1+1)*(L2+1)] backward sensitivity workspace
 * @param d_dP_dT       [B, L1, L2] output temperature Jacobian
 * @param d_lengths     [B, 2] per-batch sequence lengths
 */
void lcs_param_grad(
    const float* d_alpha,
    const float* d_scores,
    const float* d_lcs_score,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float T
);

}  // namespace lcs
}  // namespace d2p
