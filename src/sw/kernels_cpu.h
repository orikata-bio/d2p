// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Smith-Waterman CPU Kernels (Linear Gap Penalty)
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Smith-Waterman finds optimal LOCAL alignments between two sequences.
 * This CPU implementation is functionally identical to the CUDA version
 * but uses sequential processing with enhanced numerical precision.
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
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // diagonal: align positions
 *       alpha[i-1,j] + gap,              // up: gap in sequence 2
 *       alpha[i,j-1] + gap,              // left: gap in sequence 1
 *       0                                 // sky: start new local alignment
 *   )
 *
 * Base case:
 *   alpha[0,0] = 0
 *   alpha[i,0] = -inf for i > 0
 *   alpha[0,j] = -inf for j > 0
 *
 * Partition function:
 *   S = LSE_T(alpha[i,j] for all i,j)
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * Alpha table: [B, (L1+1) * (L2+1)] flattened row-major
 *   - Index: alpha[b, i, j] = alpha[b * stride + i * (L2+1) + j]
 *   - Size: B * (L1+1) * (L2+1) floats
 *   - CPU wrappers reject unsupported flattened table sizes before entry;
 *     kernels compute flattened offsets with size_t arithmetic.
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
 * NUMERICAL STABILITY
 * ============================================================================
 *
 * - NINF = -1e30f (not -inf to avoid NaN in softmax)
 * - safe_exp clamps input to [-88, 88] (float32 range)
 * - Kahan summation reduces floating-point accumulation errors
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
 * Processes the DP table sequentially using row-major order.
 * Each cell computes the logsumexp of four options.
 *
 * @param scores     Input similarity scores [B, L1, L2] (host)
 * @param alpha      Output DP table [B, (L1+1)*(L2+1)] (host)
 * @param partition  Output soft alignment score [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr for full (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length (padded dimension)
 * @param max_L2     Maximum sequence 2 length (padded dimension)
 * @param gap        Gap penalty (typically negative, e.g., -1.0)
 * @param T          Temperature (T->0: hard max, T->inf: uniform)
 */
void sw_regular_forward_cpu(
    const float* scores, float* alpha, float* partition,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Backward pass: compute posteriors and parameter gradients.
 *
 * Computes the soft alignment matrix (posteriors) and gradients with respect
 * to gap penalty and temperature. Uses backward message passing through beta.
 *
 * @param alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param partition  Partition function from forward [B] (host)
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
void sw_regular_backward_cpu(
    const float* alpha, const float* scores, const float* partition,
    float* beta, float* posteriors, float* grad_gap, float* grad_T,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Hessian-vector product: H * v where H = d^2S/dscores^2.
 *
 * Computes the product of the Hessian with a vector v without explicitly
 * forming the O(L1^2 * L2^2) Hessian matrix. Uses forward-mode autodiff
 * through the backward pass.
 *
 * @param alpha        Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores       Input scores [B, L1, L2] (host)
 * @param partition    Partition function [B] (host)
 * @param V            Input vector [B, L1, L2] (host)
 * @param d_alpha      Workspace: dalpha [B, (L1+1)*(L2+1)] (host)
 * @param d_partition  Workspace: dpartition [B] (host)
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
void sw_regular_hvp_cpu(
    const float* alpha, const float* scores, const float* partition,
    const float* V, float* d_alpha, float* d_partition,
    float* beta, float* d_beta, float* H_scores,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T
);

/**
 * Parameter Jacobian: dP/dtheta where P = posteriors.
 *
 * Computes how the soft alignment matrix changes with respect to a parameter
 * (gap or temperature). Returns a [B, L1, L2] tensor.
 *
 * @param alpha      Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param scores     Input scores [B, L1, L2] (host)
 * @param partition  Partition function [B] (host)
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
void sw_regular_param_grad_cpu(
    const float* alpha, const float* scores, const float* partition,
    const float* dS_dtheta, float* U, float* beta, float* W, float* dP_dtheta,
    const int* lengths, int B, int max_L1, int max_L2, float gap, float T,
    int param_type
);

#ifdef __cplusplus
}
#endif
