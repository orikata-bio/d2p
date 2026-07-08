// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft DTW CPU Kernels
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Dynamic Time Warping (DTW) finds optimal alignment between two time series.
 * This CPU implementation is functionally identical to the CUDA version
 * but uses sequential processing with enhanced numerical precision.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Minimization: finds minimum cost warping path
 *   - No gap penalty: cost comes from pairwise distances
 *   - Soft version: uses temperature-scaled softmin instead of hard min
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 *   alpha[i,j] = costs[i,j] + softmin_T(
 *       alpha[i-1,j-1],  // diagonal
 *       alpha[i-1,j],    // up
 *       alpha[i,j-1]     // left
 *   )
 *
 * Base case:
 *   alpha[0,0] = 0
 *   alpha[i,0] = +inf for i > 0
 *   alpha[0,j] = +inf for j > 0
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
 *
 * Costs: [B, L1, L2] standard row-major
 *   - Index: costs[b, i, j] = costs[b * L1 * L2 + i * L2 + j]
 *
 * ============================================================================
 * CPU-SPECIFIC OPTIMIZATIONS
 * ============================================================================
 *
 * - Kahan compensated summation for numerical precision in softmin
 * - Sequential batch processing (parallelism via PyTorch's threading)
 * - Cache-friendly row-major traversal within each batch
 *
 * ============================================================================
 */

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward pass: compute alpha table and DTW distance.
 *
 * @param costs     Input cost matrix [B, L1, L2] (host)
 * @param alpha     Output DP table [B, (L1+1)*(L2+1)] (host)
 * @param score     Output DTW distance [B] (host)
 * @param lengths   Sequence lengths [B, 2] or nullptr for full (host)
 * @param B         Batch size
 * @param max_L1    Maximum sequence 1 length (padded dimension)
 * @param max_L2    Maximum sequence 2 length (padded dimension)
 * @param T         Temperature (T->0: hard min, T->inf: uniform)
 * @param bandwidth Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_forward_cpu(
    const float* costs, float* alpha, float* score,
    const int* lengths, int B, int max_L1, int max_L2, float T, int64_t bandwidth
);

/**
 * Backward pass: compute posteriors and temperature gradient.
 *
 * @param alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (host)
 * @param costs      Input costs [B, L1, L2] (host)
 * @param score      DTW distance from forward [B] (host)
 * @param beta       Workspace: beta table [B, (L1+1)*(L2+1)] (host)
 * @param posteriors Output: soft alignment [B, L1, L2] (host)
 * @param grad_T     Output: dS/dT [B] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param T          Temperature
 * @param bandwidth  Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_backward_cpu(
    const float* alpha, const float* costs, const float* score,
    float* beta, float* posteriors, float* grad_T,
    const int* lengths, int B, int max_L1, int max_L2, float T, int64_t bandwidth
);

/**
 * Hessian-vector product: H * v where H = d^2S/dcosts^2.
 *
 * @param alpha      Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param costs      Input costs [B, L1, L2] (host)
 * @param score      DTW distance [B] (host)
 * @param V          Input vector [B, L1, L2] (host)
 * @param d_alpha    Workspace: dalpha [B, (L1+1)*(L2+1)] (host)
 * @param d_score    Workspace: dscore [B] (host)
 * @param beta       Workspace: beta [B, (L1+1)*(L2+1)] (host)
 * @param d_beta     Workspace: dbeta [B, (L1+1)*(L2+1)] (host)
 * @param H_costs    Output: H * v [B, L1, L2] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param T          Temperature
 * @param bandwidth  Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_hvp_cpu(
    const float* alpha, const float* costs, const float* score,
    const float* V, float* d_alpha, float* d_score,
    float* beta, float* d_beta, float* H_costs,
    const int* lengths, int B, int max_L1, int max_L2, float T, int64_t bandwidth
);

/**
 * Temperature Jacobian: dP/dT where P = posteriors.
 *
 * @param alpha      Alpha from forward [B, (L1+1)*(L2+1)] (host)
 * @param costs      Input costs [B, L1, L2] (host)
 * @param score      DTW distance [B] (host)
 * @param U          Workspace: dalpha/dT [B, (L1+1)*(L2+1)] (host)
 * @param beta       Workspace: beta [B, (L1+1)*(L2+1)] (host)
 * @param W          Workspace: dbeta/dT [B, (L1+1)*(L2+1)] (host)
 * @param dP_dT      Output: dP/dT [B, L1, L2] (host)
 * @param lengths    Sequence lengths [B, 2] or nullptr (host)
 * @param B          Batch size
 * @param max_L1     Maximum sequence 1 length
 * @param max_L2     Maximum sequence 2 length
 * @param T          Temperature
 * @param bandwidth  Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_param_grad_cpu(
    const float* alpha, const float* costs, const float* score,
    float* U, float* beta, float* W, float* dP_dT,
    const int* lengths, int B, int max_L1, int max_L2, float T, int64_t bandwidth
);

#ifdef __cplusplus
}
#endif
