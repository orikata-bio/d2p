// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Smith-Waterman CUDA Kernels (Affine Gap Penalty)
 *
 * Differentiable local sequence alignment with affine gap costs.
 * Uses 3-state DP (Match, Insert, Delete) for O(L1*L2) complexity.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Affine gap Smith-Waterman extends linear-gap SW with separate costs for
 * opening and extending gaps. This better models biological sequences where
 * starting a gap is more costly than extending an existing one.
 *
 * Key properties:
 *   - Local alignment: best matching subsequence
 *   - Affine gap model: cost = gap_open + gap_ext * (length - 1)
 *   - 3-state DP: Match (M), Insert (I), Delete (D)
 *   - Soft version: temperature-scaled logsumexp
 *
 * ============================================================================
 * STATE MACHINE
 * ============================================================================
 *
 * Three states track whether we're in a gap:
 *
 *   M[i,j] = best score ending with seq1[i] aligned to seq2[j]
 *   I[i,j] = best score ending with gap in seq2 (insertion)
 *   D[i,j] = best score ending with gap in seq1 (deletion)
 *
 *            +--gap_ext--+
 *            v           |
 *   +---+  gap_open  +---+---+
 *   | M |----------->| I |   |
 *   +---+<-----------+---+   |
 *     |    (free)            |
 *     |                      |
 *     | gap_open             |
 *     v                      |
 *   +---+<-------------------+
 *   | D |----gap_ext---------+
 *   +---+
 *
 * ============================================================================
 * RECURRENCE RELATIONS
 * ============================================================================
 *
 * Match state (aligned positions):
 *   M[i,j] = scores[i,j] + LSE_T(
 *       M[i-1,j-1],    // continue alignment
 *       I[i-1,j-1],    // end insertion, align
 *       D[i-1,j-1],    // end deletion, align
 *       0               // start new local alignment (sky)
 *   )
 *
 * Insert state (gap in seq2):
 *   I[i,j] = LSE_T(
 *       M[i-1,j] + gap_open,   // open new gap
 *       I[i-1,j] + gap_ext     // extend existing gap
 *   )
 *
 * Delete state (gap in seq1):
 *   D[i,j] = LSE_T(
 *       M[i,j-1] + gap_open,   // open new gap
 *       D[i,j-1] + gap_ext     // extend existing gap
 *   )
 *
 * Base cases:
 *   M[0,0] = 0, M[i,0] = M[0,j] = -inf
 *   I[*,*] = -inf initially
 *   D[*,*] = -inf initially
 *
 * Partition function:
 *   S = LSE_T(M[i,j], I[i,j], D[i,j] for all i,j)
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, 3, (L1+1), (L2+1)] but stored as [B, 3*(L1+1)*(L2+1)]
 *
 * State indexing within flattened array:
 *   cell_stride = (L1+1) * (L2+1)
 *   M[b,i,j] = alpha[b * 3 * cell_stride + 0 * cell_stride + i*(L2+1) + j]
 *   I[b,i,j] = alpha[b * 3 * cell_stride + 1 * cell_stride + i*(L2+1) + j]
 *   D[b,i,j] = alpha[b * 3 * cell_stride + 2 * cell_stride + i*(L2+1) + j]
 *
 * Or equivalently with state enum {M=0, I=1, D=2}:
 *   alpha[b, state, i, j] = alpha[b*3*stride + state*stride + i*(L2+1) + j]
 *
 * Scores: [B, L1, L2] standard row-major (same as linear gap)
 *
 * ============================================================================
 * CUDA PARALLELIZATION
 * ============================================================================
 *
 * Uses WAVEFRONT (anti-diagonal) parallelization, same as linear gap:
 *
 *     j=0  j=1  j=2  j=3
 *   +----+----+----+----+
 *   | d0 | d1 | d2 | d3 |  i=0
 *   +----+----+----+----+
 *   | d1 | d2 | d3 | d4 |  i=1
 *   +----+----+----+----+
 *
 * Each cell computes all 3 states (M, I, D) together since they depend
 * on each other within the cell. States are computed in order: I, D, M.
 *
 * ============================================================================
 * GRADIENT COMPUTATIONS
 * ============================================================================
 *
 * Backward pass computes:
 *   - posteriors = dS/dscores [B, L1, L2]   -- soft alignment matrix
 *   - grad_open = dS/dgap_open [B]          -- expected gap openings
 *   - grad_ext = dS/dgap_ext [B]            -- expected gap extensions
 *   - grad_T = dS/dT [B]                    -- temperature gradient
 *
 * Beta tables (one per state):
 *   beta_M[i,j] = dS/dM[i,j]
 *   beta_I[i,j] = dS/dI[i,j]
 *   beta_D[i,j] = dS/dD[i,j]
 *
 * Only beta_M contributes directly to posteriors since only M states
 * include the score term.
 *
 * ============================================================================
 * NUMERICAL STABILITY
 * ============================================================================
 *
 * - NINF = -1e30f (not -inf to avoid NaN in softmax)
 * - safe_exp clamps input to [-88, 88] (float32 range)
 * - LogSumExp computed as: max + T * log(sum(exp((x - max) / T)))
 * - States initialized to NINF except M[0,0] = 0
 *
 * ============================================================================
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward pass: compute 3-state alpha tables and partition function.
 *
 * Fills the DP tables for Match, Insert, and Delete states using wavefront
 * parallelization. Each cell updates all three states.
 *
 * @param d_scores    Input similarity scores [B, L1, L2] (device)
 * @param d_alpha     Output DP tables [B, 3*(L1+1)*(L2+1)] (device)
 *                    Layout: [M states | I states | D states]
 * @param d_partition Output soft alignment score [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] or nullptr for full (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param gap_open    Gap opening penalty (typically negative, e.g., -2.0)
 * @param gap_ext     Gap extension penalty (typically negative, e.g., -0.5)
 * @param T           Temperature (T->0: hard max, T->inf: uniform)
 */
void sw_affine_forward(
    const float* d_scores, float* d_alpha, float* d_partition,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap_open, float gap_ext, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * Computes soft alignment matrix and gradients w.r.t. gap_open, gap_ext,
 * and temperature. Uses backward message passing through 3-state beta.
 *
 * @param d_alpha      Alpha tables from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_partition  Partition function from forward [B] (device)
 * @param d_beta       Workspace: beta tables [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_posteriors Output: soft alignment [B, L1, L2] (device)
 * @param d_grad_open  Output: dS/dgap_open [B] (device)
 * @param d_grad_ext   Output: dS/dgap_ext [B] (device)
 * @param d_grad_T     Output: dS/dT [B] (device)
 * @param d_lengths    Sequence lengths [B, 2] or nullptr (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap_open     Gap opening penalty
 * @param gap_ext      Gap extension penalty
 * @param T            Temperature
 */
void sw_affine_backward(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    float* d_beta, float* d_posteriors,
    float* d_grad_open, float* d_grad_ext, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap_open, float gap_ext, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * Computes the product of the Hessian with a vector v without explicitly
 * forming the O(L1^2 * L2^2) Hessian matrix. Uses forward-mode autodiff
 * through the 3-state backward pass.
 *
 * @param d_alpha        Alpha from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores       Input scores [B, L1, L2] (device)
 * @param d_partition    Partition function [B] (device)
 * @param d_V            Input vector [B, L1, L2] (device)
 * @param d_d_alpha      Workspace: dalpha [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_d_partition  Workspace: dpartition [B] (device)
 * @param d_beta         Workspace: beta accumulation [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_d_beta       Workspace: dbeta [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_H_scores     Output: H * v [B, L1, L2] (device)
 * @param d_lengths      Sequence lengths [B, 2] or nullptr (device)
 * @param B              Batch size
 * @param max_L1         Maximum sequence 1 length
 * @param max_L2         Maximum sequence 2 length
 * @param gap_open       Gap opening penalty
 * @param gap_ext        Gap extension penalty
 * @param T              Temperature
 */
void sw_affine_hvp(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    const float* d_V, float* d_d_alpha, float* d_d_partition,
    float* d_beta, float* d_d_beta, float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float gap_open, float gap_ext, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * Computes how the soft alignment matrix changes with respect to a parameter
 * (gap_open, gap_ext, or temperature). Returns a [B, L1, L2] tensor.
 *
 * @param d_alpha      Alpha from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_partition  Partition function [B] (device)
 * @param d_dS_dtheta  Pre-computed dS/dtheta from backward [B] (device)
 * @param d_U          Workspace: dalpha/dtheta [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_beta       Workspace: beta [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_W          Workspace: dbeta/dtheta [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_dP_dtheta  Output: dP/dtheta [B, L1, L2] (device)
 * @param d_lengths    Sequence lengths [B, 2] or nullptr (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap_open     Gap opening penalty
 * @param gap_ext      Gap extension penalty
 * @param T            Temperature
 * @param param_type   0 = gap_open, 1 = gap_ext, 2 = temperature
 */
void sw_affine_param_grad(
    const float* d_alpha, const float* d_scores, const float* d_partition,
    const float* d_dS_dtheta,
    float* d_U, float* d_beta, float* d_W,
    float* d_dP_dtheta,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
