// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Needleman-Wunsch Affine Gap CUDA Kernels
 *
 * Differentiable global sequence alignment with affine gap penalty.
 * Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Needleman-Wunsch finds optimal GLOBAL alignments between two sequences.
 * Affine gap penalties distinguish gap opening from gap extension.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Affine gap model: gap_open + (k-1)*gap_ext for k consecutive gaps
 *   - Three states: M (match), I (insert in seq2), D (delete in seq1)
 *   - Soft version: uses temperature-scaled logsumexp instead of max
 *
 * Key differences from Smith-Waterman affine:
 *   - No "sky" restart transition in M state (global alignment)
 *   - Base cases: M(0,0)=0, I(i,0)=g_o+(i-1)*g_e, D(0,j)=g_o+(j-1)*g_e
 *   - Score = LSE(M(L1,L2), I(L1,L2), D(L1,L2)) at terminal
 *   - Beta initialized at terminal only
 *
 * ============================================================================
 * RECURRENCE RELATIONS
 * ============================================================================
 *
 * M[i,j] = score[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1])
 * I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext, D[i-1,j] + gap_open)
 * D[i,j] = LSE_T(M[i,j-1] + gap_open, I[i,j-1] + gap_open, D[i,j-1] + gap_ext)
 *
 * Base cases:
 *   M(0,0) = 0, M(i,0) = -inf for i>0, M(0,j) = -inf for j>0
 *   I(i,0) = gap_open + (i-1)*gap_ext for i>0, I(0,j) = -inf
 *   D(0,j) = gap_open + (j-1)*gap_ext for j>0, D(i,0) = -inf
 *
 * Score:
 *   S = LSE_T(M[L1,L2], I[L1,L2], D[L1,L2])
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, 3*(L1+1)*(L2+1)] flattened row-major, 3 states stacked
 *   - State 0: M, State 1: I, State 2: D
 *   - Index: alpha[b, state, i, j] = alpha[b*stride_all + state*stride_state + i*(L2+1) + j]
 *
 * Scores: [B, L1, L2] standard row-major
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
 * @param d_alpha     Output DP table [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_score     Output alignment score [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] or nullptr for full (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param gap_open    Gap opening penalty (typically negative)
 * @param gap_ext     Gap extension penalty (typically negative)
 * @param T           Temperature (T->0: hard max, T->inf: uniform)
 */
void nw_affine_forward(
    const float* d_scores, float* d_alpha, float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * @param d_alpha         Alpha table from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores        Input scores [B, L1, L2] (device)
 * @param d_score         Alignment score from forward [B] (device)
 * @param d_beta          Workspace: beta table [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_posteriors    Output: soft alignment [B, L1, L2] (device)
 * @param d_grad_gap_open Output: dS/dgap_open [B] (device)
 * @param d_grad_gap_ext  Output: dS/dgap_ext [B] (device)
 * @param d_grad_T        Output: dS/dT [B] (device)
 * @param d_lengths       Sequence lengths [B, 2] or nullptr (device)
 * @param B               Batch size
 * @param max_L1          Maximum sequence 1 length
 * @param max_L2          Maximum sequence 2 length
 * @param gap_open        Gap opening penalty
 * @param gap_ext         Gap extension penalty
 * @param T               Temperature
 */
void nw_affine_backward(
    const float* d_alpha, const float* d_scores, const float* d_score,
    float* d_beta, float* d_posteriors,
    float* d_grad_gap_open, float* d_grad_gap_ext, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * @param d_alpha        Alpha from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores       Input scores [B, L1, L2] (device)
 * @param d_score        Alignment score [B] (device)
 * @param d_V            Input vector [B, L1, L2] (device)
 * @param d_d_alpha      Workspace: dalpha [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_d_score      Workspace: dscore [B] (device)
 * @param d_beta         Workspace: beta [B, 3*(L1+1)*(L2+1)] (device)
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
void nw_affine_hvp(
    const float* d_alpha, const float* d_scores, const float* d_score,
    const float* d_V, float* d_d_alpha, float* d_d_score,
    float* d_beta, float* d_d_beta, float* d_H_scores,
    const int* d_lengths,
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * @param d_alpha      Alpha from forward [B, 3*(L1+1)*(L2+1)] (device)
 * @param d_scores     Input scores [B, L1, L2] (device)
 * @param d_score      Alignment score [B] (device)
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
void nw_affine_param_grad(
    const float* d_alpha, const float* d_scores, const float* d_score,
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
