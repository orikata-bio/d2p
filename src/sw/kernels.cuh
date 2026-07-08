// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Smith-Waterman CUDA Kernels (Linear Gap Penalty)
 *
 * Differentiable local sequence alignment using temperature-scaled softmax.
 * Implements forward, backward, Hessian-vector product, and parameter gradients.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Smith-Waterman finds optimal LOCAL alignments between two sequences.
 * Unlike global alignment (Needleman-Wunsch), it can start and end anywhere.
 *
 * Key properties:
 *   - Local alignment: best matching subsequence (not full sequences)
 *   - Linear gap model: each gap costs a fixed penalty
 *   - Soft version: uses temperature-scaled logsumexp instead of max
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 * For sequences of length L1 and L2, with scores[i,j] = similarity(seq1[i], seq2[j]):
 *
 *   alpha[i,j] = T * log( exp(alpha[i-1,j-1] + scores[i,j]) / T     // align
 *                       + exp(alpha[i-1,j] + gap) / T               // insertion
 *                       + exp(alpha[i,j-1] + gap) / T               // deletion
 *                       + exp(0) / T )                              // restart
 *
 * Simplified as LogSumExp (LSE):
 *
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // diagonal: align positions
 *       alpha[i-1,j] + gap,              // up: gap in sequence 2
 *       alpha[i,j-1] + gap,              // left: gap in sequence 1
 *       0                                 // sky: start new local alignment
 *   )
 *
 * Base case:
 *   alpha[0,0] = 0
 *   alpha[i,0] = -inf for i > 0  (can only restart, handled by sky option)
 *   alpha[0,j] = -inf for j > 0
 *
 * Partition function (soft alignment score):
 *   S = LSE_T(alpha[i,j] for all i,j)  -- best local alignment anywhere
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, (L1+1) * (L2+1)] flattened row-major
 *   - Index: alpha[b, i, j] = alpha[b * stride + i * (L2+1) + j]
 *   - Size: B * (L1+1) * (L2+1) floats
 *   - The +1 accounts for boundary conditions (i=0, j=0)
 *
 * Scores: [B, L1, L2] standard row-major
 *   - Index: scores[b, i, j] = scores[b * L1 * L2 + i * L2 + j]
 *   - Note: 0-indexed, so scores[i,j] corresponds to alpha[i+1,j+1]
 *
 * ============================================================================
 * CUDA PARALLELIZATION
 * ============================================================================
 *
 * Uses WAVEFRONT (anti-diagonal) parallelization:
 *
 *     j=0  j=1  j=2  j=3
 *   +----+----+----+----+
 *   | d0 | d1 | d2 | d3 |  i=0
 *   +----+----+----+----+
 *   | d1 | d2 | d3 | d4 |  i=1
 *   +----+----+----+----+
 *   | d2 | d3 | d4 | d5 |  i=2
 *   +----+----+----+----+
 *
 * Cells on the same anti-diagonal (d0, d1, ...) are independent and
 * can be computed in parallel. We process diagonals sequentially,
 * with full parallelism within each diagonal.
 *
 * Thread mapping: One thread per (batch, cell on diagonal)
 *
 * ============================================================================
 * GRADIENT COMPUTATIONS
 * ============================================================================
 *
 * Backward pass computes:
 *   - posteriors = dS/dscores [B, L1, L2]  -- soft alignment matrix
 *   - grad_gap = dS/dgap [B]               -- expected gap count
 *   - grad_T = dS/dT [B]                   -- temperature gradient
 *
 * Beta (backward) table:
 *   beta[i,j] = dS/dalpha[i,j] = exp((alpha[i,j] - S) / T)
 *
 * Posteriors (soft alignment):
 *   P[i,j] = beta[i+1,j+1] * w_align[i+1,j+1] + beta_sky_contrib
 *
 * HVP (Hessian-vector product):
 *   Computes d^2S/dscores^2 * V efficiently via forward-mode autodiff
 *   through the backward pass, without forming the O(L^4) Hessian.
 *
 * Parameter Jacobian:
 *   Computes dP/dtheta where P = posteriors, theta in {gap, temperature}
 *   Uses coupled forward-backward differentiation.
 *
 * ============================================================================
 * NUMERICAL STABILITY
 * ============================================================================
 *
 * - NINF = -1e30f (not -inf to avoid NaN in softmax)
 * - safe_exp clamps input to [-88, 88] (float32 range)
 * - LogSumExp computed as: max + T * log(sum(exp((x - max) / T)))
 *
 * ============================================================================
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward pass: compute alpha table and partition function.
 *
 * Fills the DP table using wavefront parallelization. Each cell computes
 * the logsumexp of four options: align, insert, delete, restart.
 *
 * @param d_scores    Input similarity scores [B, L1, L2] (device)
 * @param d_alpha     Output DP table [B, (L1+1)*(L2+1)] (device)
 * @param d_partition Output soft alignment score [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] or nullptr for full (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param gap         Gap penalty (typically negative, e.g., -1.0)
 * @param T           Temperature (T->0: hard max, T->inf: uniform)
 */
void sw_regular_forward(
    const float* d_scores, float* d_alpha, float* d_partition,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Warp-parallel forward pass: one warp per batch element.
 *
 * This variant collapses alpha initialization, the anti-diagonal forward pass,
 * and partition reduction into a single CUDA kernel. It is used as a forward
 * fast path for moderately large batches on CUDA.
 *
 * @param d_scores    Input similarity scores [B, L1, L2] (device)
 * @param d_alpha     Output DP table [B, (L1+1)*(L2+1)] (device)
 * @param d_partition Output soft alignment score [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param gap         Gap penalty
 * @param T           Temperature
 */
void sw_regular_forward_warp(
    const float* d_scores, float* d_alpha, float* d_partition,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * Computes the soft alignment matrix (posteriors) and gradients with respect
 * to gap penalty and temperature. Uses backward message passing through beta.
 *
 * @param d_alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_partition  Partition function from forward [B] (device)
 * @param d_beta       Workspace: beta table [B, (L1+1)*(L2+1)] (device)
 * @param d_posteriors Output: soft alignment [B, L1, L2] (device)
 * @param d_grad_gap   Output: dS/dgap [B] (device)
 * @param d_grad_T     Output: dS/dT [B] (device)
 * @param d_lengths    Sequence lengths [B, 2] or nullptr (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap          Gap penalty
 * @param T            Temperature
 */
void sw_regular_backward(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    float* d_beta, float* d_posteriors, float* d_grad_gap, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Warp-parallel backward pass: one warp per batch element.
 *
 * Uses the existing beta initialization and temperature-gradient kernels, but
 * replaces the reverse anti-diagonal traceback propagation with a persistent
 * warp kernel.
 */
void sw_regular_backward_warp(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    float* d_beta, float* d_posteriors, float* d_grad_gap, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * Computes the product of the Hessian with a vector v without explicitly
 * forming the O(L1^2 * L2^2) Hessian matrix. Uses forward-mode autodiff
 * through the backward pass.
 *
 * This is useful for:
 *   - Second-order optimization (Newton methods)
 *   - Computing curvature information
 *   - Implicit differentiation
 *
 * @param d_alpha        Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores       Input scores [B, L1, L2] (device)
 * @param d_partition    Partition function [B] (device)
 * @param d_V            Input vector [B, L1, L2] (device)
 * @param d_d_alpha      Workspace: dalpha [B, (L1+1)*(L2+1)] (device)
 * @param d_d_partition  Workspace: dpartition [B] (device)
 * @param d_beta         Workspace: beta [B, (L1+1)*(L2+1)] (device)
 * @param d_d_beta       Workspace: dbeta [B, (L1+1)*(L2+1)] (device)
 * @param d_H_scores     Output: H * v [B, L1, L2] (device)
 * @param d_lengths      Sequence lengths [B, 2] or nullptr (device)
 * @param B              Batch size
 * @param max_L1         Maximum sequence 1 length
 * @param max_L2         Maximum sequence 2 length
 * @param gap            Gap penalty
 * @param T              Temperature
 */
void sw_regular_hvp(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    const float* d_V, float* d_d_alpha, float* d_d_partition,
    float* d_beta, float* d_d_beta, float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * Computes how the soft alignment matrix changes with respect to a parameter
 * (gap or temperature). This is a [B, L1, L2] tensor showing sensitivity of
 * each alignment position to the parameter.
 *
 * Uses coupled differentiation:
 *   1. Forward U pass: compute dalpha/dtheta
 *   2. Backward W pass: compute dbeta/dtheta and accumulate dP/dtheta
 *
 * @param d_alpha      Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_partition  Partition function [B] (device)
 * @param d_dS_dtheta  Pre-computed dS/dtheta from backward [B] (device)
 * @param d_U          Workspace: dalpha/dtheta [B, (L1+1)*(L2+1)] (device)
 * @param d_beta       Workspace: beta [B, (L1+1)*(L2+1)] (device)
 * @param d_W          Workspace: dbeta/dtheta [B, (L1+1)*(L2+1)] (device)
 * @param d_dP_dtheta  Output: dP/dtheta [B, L1, L2] (device)
 * @param d_lengths    Sequence lengths [B, 2] or nullptr (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap          Gap penalty
 * @param T            Temperature
 * @param param_type   0 = gap, 1 = temperature
 */
void sw_regular_param_grad(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    const float* d_dS_dtheta,
    float* d_U, float* d_beta, float* d_W,
    float* d_dP_dtheta,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap, float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
