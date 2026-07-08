// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Needleman-Wunsch CPU Kernels (Linear Gap Penalty)
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Needleman-Wunsch finds optimal GLOBAL alignments between two sequences.
 * This CPU implementation is functionally identical to the CUDA version
 * but uses sequential processing with enhanced numerical precision.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Linear gap model: each gap costs a fixed penalty
 *   - Soft version: uses temperature-scaled logsumexp instead of max
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // diagonal: align positions
 *       alpha[i-1,j] + gap,              // up: gap in sequence 2
 *       alpha[i,j-1] + gap               // left: gap in sequence 1
 *   )
 *
 * Base cases:
 *   alpha[0,0] = 0
 *   alpha[i,0] = i * gap for i > 0
 *   alpha[0,j] = j * gap for j > 0
 *
 * Score:
 *   S = alpha[L1, L2]
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
 * @param alpha      Output DP table [B, (L1+1)*(L2+1)] (host)
 * @param score      Output alignment score [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr for full (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length (padded dimension)
 * @param max_L2     Maximum sequence 2 length (padded dimension)
 * @param gap        Gap penalty (typically negative, e.g., -1.0)
 * @param T          Temperature (T->0: hard max, T->inf: uniform)
 */
void nw_forward_cpu(
    const float* scores, float* alpha, float* score,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * @param alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param score      Alignment score from forward [B] (host)
 * @param beta       Workspace: beta table [B, (L1+1)*(L2+1)] (host)
 * @param posteriors Output: soft alignment [B, L1, L2] (host)
 * @param grad_gap   Output: dS/dgap [B] (host)
 * @param grad_T     Output: dS/dT [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param gap        Gap penalty
 * @param T          Temperature
 */
void nw_backward_cpu(
    const float* alpha, const float* scores, const float* score,
    float* beta, float* posteriors, float* grad_gap, float* grad_T,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * @param alpha        Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores       Input scores [B, L1, L2] (host)
 * @param score        Alignment score [B] (host)
 * @param V            Input vector [B, L1, L2] (host)
 * @param d_alpha      Workspace: dalpha [B, (L1+1)*(L2+1)] (host)
 * @param d_score      Workspace: dscore [B] (host)
 * @param beta         Workspace: beta [B, (L1+1)*(L2+1)] (host)
 * @param d_beta       Workspace: dbeta [B, (L1+1)*(L2+1)] (host)
 * @param H_scores     Output: H * v [B, L1, L2] (host)
 * @param lengths      Sequence lengths [B, 2] or nullptr (host)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param gap          Gap penalty
 * @param T            Temperature
 */
void nw_hvp_cpu(
    const float* alpha, const float* scores, const float* score,
    const float* V, float* d_alpha, float* d_score,
    float* beta, float* d_beta, float* H_scores,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * @param alpha      Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param score      Alignment score [B] (host)
 * @param dS_dtheta  Pre-computed dS/dtheta from backward [B] (host)
 * @param U          Workspace: dalpha/dtheta [B, (L1+1)*(L2+1)] (host)
 * @param beta       Workspace: beta [B, (L1+1)*(L2+1)] (host)
 * @param W          Workspace: dbeta/dtheta [B, (L1+1)*(L2+1)] (host)
 * @param dP_dtheta  Output: dP/dtheta [B, L1, L2] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param gap        Gap penalty
 * @param T          Temperature
 * @param param_type 0 = gap, 1 = temperature
 */
void nw_param_grad_cpu(
    const float* alpha, const float* scores, const float* score,
    const float* dS_dtheta, float* U, float* beta, float* W, float* dP_dtheta,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
