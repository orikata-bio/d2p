// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Smith-Waterman CPU Kernels (Affine Gap Penalty)
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses 3-state DP (Match, Insert, Delete) with Kahan summation for precision.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Affine gap Smith-Waterman extends linear-gap SW with separate costs for
 * opening and extending gaps. This CPU implementation is functionally
 * identical to the CUDA version.
 *
 * Key properties:
 *   - Local alignment: best matching subsequence
 *   - Affine gap model: cost = gap_open + gap_ext * (length - 1)
 *   - 3-state DP: Match (M), Insert (I), Delete (D)
 *   - Soft version: temperature-scaled logsumexp
 *
 * ============================================================================
 * RECURRENCE RELATIONS
 * ============================================================================
 *
 * Match state:
 *   M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
 *
 * Insert state (gap in seq2):
 *   I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)
 *
 * Delete state (gap in seq1):
 *   D[i,j] = LSE_T(M[i,j-1] + gap_open, D[i,j-1] + gap_ext)
 *
 * Partition function:
 *   S = LSE_T(M[i,j] for all i,j)
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, 3*(L1+1)*(L2+1)] with states interleaved:
 *   cell_stride = (L1+1) * (L2+1)
 *   M[b,i,j] = alpha[b * 3 * cell_stride + 0 * cell_stride + i*(L2+1) + j]
 *   I[b,i,j] = alpha[b * 3 * cell_stride + 1 * cell_stride + i*(L2+1) + j]
 *   D[b,i,j] = alpha[b * 3 * cell_stride + 2 * cell_stride + i*(L2+1) + j]
 *
 * ============================================================================
 * CPU-SPECIFIC OPTIMIZATIONS
 * ============================================================================
 *
 * - Kahan compensated summation for numerical precision
 * - Batch-parallel processing via PyTorch intra-op threading
 * - Cache-friendly row-major traversal
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
 * @param scores     Input similarity scores [B, L1, L2] (host)
 * @param alpha      Output DP tables [B, 3*(L1+1)*(L2+1)] (host)
 * @param partition  Output soft alignment score [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr for full (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param gap_open   Gap opening penalty (typically negative)
 * @param gap_ext    Gap extension penalty (typically negative)
 * @param T          Temperature
 */
void sw_affine_forward_cpu(
    const float* scores, float* alpha, float* partition,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * @param alpha      Alpha tables from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param partition  Partition function [B] (host)
 * @param beta       Workspace: beta tables [B, 3*(L1+1)*(L2+1)] (host)
 * @param posteriors Output: soft alignment [B, L1, L2] (host)
 * @param grad_open  Output: dS/dgap_open [B] (host)
 * @param grad_ext   Output: dS/dgap_ext [B] (host)
 * @param grad_T     Output: dS/dT [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param gap_open   Gap opening penalty
 * @param gap_ext    Gap extension penalty
 * @param T          Temperature
 */
void sw_affine_backward_cpu(
    const float* alpha, const float* scores, const float* partition,
    float* beta, float* posteriors, float* grad_open, float* grad_ext, float* grad_T,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * @param alpha        Alpha from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores       Input scores [B, L1, L2] (host)
 * @param partition    Partition function [B] (host)
 * @param V            Input vector [B, L1, L2] (host)
 * @param d_alpha      Workspace: dalpha [B, 3*(L1+1)*(L2+1)] (host)
 * @param d_partition  Workspace: dpartition [B] (host)
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
void sw_affine_hvp_cpu(
    const float* alpha, const float* scores, const float* partition,
    const float* V, float* d_alpha, float* d_partition, float* d_beta, float* H_scores,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * @param alpha      Alpha from forward [B, 3*(L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param partition  Partition function [B] (host)
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
void sw_affine_param_grad_cpu(
    const float* alpha, const float* scores, const float* partition,
    const float* dS_dtheta, float* U, float* beta, float* W, float* dP_dtheta,
    const int* lengths, int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T, int param_type
);

#ifdef __cplusplus
}
#endif
