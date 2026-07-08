// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Needleman-Wunsch Affine Gap CPU Kernels
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
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
 * CPU-SPECIFIC OPTIMIZATIONS
 * ============================================================================
 *
 * - Kahan compensated summation for numerical precision in logsumexp
 * - Sequential batch processing (parallelism via PyTorch's threading)
 * - Cache-friendly row-major traversal within each batch
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
 * @param scores     Input similarity scores [B, L1, L2] (host)
 * @param alpha      Output DP table [B, 3*(L1+1)*(L2+1)] (host)
 * @param score      Output alignment score [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr for full (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length (padded dimension)
 * @param max_L2     Maximum sequence 2 length (padded dimension)
 * @param gap_open   Gap opening penalty (typically negative)
 * @param gap_ext    Gap extension penalty (typically negative)
 * @param T          Temperature (T->0: hard max, T->inf: uniform)
 */
void nw_affine_forward_cpu(
    const float* scores, float* alpha, float* score,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * @param alpha         Alpha table from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores        Input scores [B, L1, L2] (host)
 * @param score         Alignment score from forward [B] (host)
 * @param beta          Workspace: beta table [B, 3*(L1+1)*(L2+1)] (host)
 * @param posteriors    Output: soft alignment [B, L1, L2] (host)
 * @param grad_gap_open Output: dS/dgap_open [B] (host)
 * @param grad_gap_ext  Output: dS/dgap_ext [B] (host)
 * @param grad_T        Output: dS/dT [B] (host)
 * @param lengths       Sequence lengths [B, 2] or nullptr (host)
 * @param B             Batch size
 * @param max_L1        Maximum sequence 1 length
 * @param max_L2        Maximum sequence 2 length
 * @param gap_open      Gap opening penalty
 * @param gap_ext       Gap extension penalty
 * @param T             Temperature
 */
void nw_affine_backward_cpu(
    const float* alpha, const float* scores, const float* score,
    float* beta, float* posteriors,
    float* grad_gap_open, float* grad_gap_ext, float* grad_T,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * @param alpha        Alpha from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores       Input scores [B, L1, L2] (host)
 * @param score        Alignment score [B] (host)
 * @param V            Input vector [B, L1, L2] (host)
 * @param d_alpha      Workspace: dalpha [B, 3*(L1+1)*(L2+1)] (host)
 * @param d_score      Workspace: dscore [B] (host)
 * @param beta         Workspace: beta [B, 3*(L1+1)*(L2+1)] (host)
 * @param d_beta       Workspace: dbeta [B, 3*(L1+1)*(L2+1)] (host)
 * @param H_scores     Output: H * v [B, L1, L2] (host)
 * @param lengths      Sequence lengths [B, 2] or nullptr (host)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap_open     Gap opening penalty
 * @param gap_ext      Gap extension penalty
 * @param T            Temperature
 */
void nw_affine_hvp_cpu(
    const float* alpha, const float* scores, const float* score,
    const float* V, float* d_alpha, float* d_score,
    float* beta, float* d_beta, float* H_scores,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * @param alpha      Alpha from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param score      Alignment score [B] (host)
 * @param dS_dtheta  Pre-computed dS/dtheta from backward [B] (host)
 * @param U          Workspace: dalpha/dtheta [B, 3*(L1+1)*(L2+1)] (host)
 * @param beta       Workspace: beta [B, 3*(L1+1)*(L2+1)] (host)
 * @param W          Workspace: dbeta/dtheta [B, 3*(L1+1)*(L2+1)] (host)
 * @param dP_dtheta  Output: dP/dtheta [B, L1, L2] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param gap_open   Gap opening penalty
 * @param gap_ext    Gap extension penalty
 * @param T          Temperature
 * @param param_type 0 = gap_open, 1 = gap_ext, 2 = temperature
 */
void nw_affine_param_grad_cpu(
    const float* alpha, const float* scores, const float* score,
    const float* dS_dtheta, float* U, float* beta, float* W, float* dP_dtheta,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
