// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Needleman-Wunsch CUDA Kernels (Linear Gap Penalty)
 *
 * Differentiable global sequence alignment using temperature-scaled softmax.
 * Implements forward, backward, Hessian-vector product, and parameter gradients.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Needleman-Wunsch finds optimal GLOBAL alignments between two sequences.
 * Unlike local alignment (Smith-Waterman), it aligns the entire sequences
 * from start to end.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Linear gap model: each gap costs a fixed penalty
 *   - Soft version: uses temperature-scaled logsumexp instead of max
 *
 * Key differences from Smith-Waterman:
 *   - 3 transitions (align, insert, delete) - no "restart" option
 *   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
 *   - Score = alpha[L1,L2] at terminal, not max over all cells
 *   - Beta initialized at terminal only: beta[L1,L2] = 1
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 * For sequences of length L1 and L2:
 *
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // diagonal: align positions
 *       alpha[i-1,j] + gap,              // up: gap in sequence 2
 *       alpha[i,j-1] + gap               // left: gap in sequence 1
 *   )
 *
 * Base cases:
 *   alpha[0,0] = 0
 *   alpha[i,0] = i * gap  for i > 0  (leading gaps in seq2)
 *   alpha[0,j] = j * gap  for j > 0  (leading gaps in seq1)
 *
 * Alignment score:
 *   S = alpha[L1, L2]  -- score at terminal (global alignment)
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, (L1+1) * (L2+1)] flattened row-major
 *   - Index: alpha[b, i, j] = alpha[b * stride + i * (L2+1) + j]
 *   - Size: B * (L1+1) * (L2+1) floats
 *
 * Scores: [B, L1, L2] standard row-major
 *   - Index: scores[b, i, j] = scores[b * L1 * L2 + i * L2 + j]
 *
 * ============================================================================
 * CUDA PARALLELIZATION
 * ============================================================================
 *
 * Uses WAVEFRONT (anti-diagonal) parallelization:
 * Cells on the same anti-diagonal k = i + j are independent.
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
 * ============================================================================
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward pass: compute alpha table and alignment score.
 *
 * @param d_scores    Input similarity scores [B, L1, L2] (device)
 * @param d_alpha     Output DP table [B, (L1+1)*(L2+1)] (device)
 * @param d_score     Output alignment score [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] or nullptr for full (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param gap         Gap penalty (typically negative, e.g., -1.0)
 * @param T           Temperature (T->0: hard max, T->inf: uniform)
 */
void nw_forward(
    const float* d_scores, float* d_alpha, float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * @param d_alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_score      Alignment score from forward [B] (device)
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
void nw_backward(
    const float* d_alpha, const float* d_scores, const float* d_score,
    float* d_beta, float* d_posteriors, float* d_grad_gap, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * @param d_alpha        Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores       Input scores [B, L1, L2] (device)
 * @param d_score        Alignment score [B] (device)
 * @param d_V            Input vector [B, L1, L2] (device)
 * @param d_d_alpha      Workspace: dalpha [B, (L1+1)*(L2+1)] (device)
 * @param d_d_score      Workspace: dscore [B] (device)
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
void nw_hvp(
    const float* d_alpha, const float* d_scores, const float* d_score,
    const float* d_V, float* d_d_alpha, float* d_d_score,
    float* d_beta, float* d_d_beta, float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * @param d_alpha      Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_score      Alignment score [B] (device)
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
void nw_param_grad(
    const float* d_alpha, const float* d_scores, const float* d_score,
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
