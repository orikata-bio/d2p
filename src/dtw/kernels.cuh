// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft DTW CUDA Kernel Declarations
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Dynamic Time Warping (DTW) finds optimal alignment between two time series
 * that may vary in speed. Unlike sequence alignment (NW/SW), DTW uses a
 * COST matrix (distances) and MINIMIZES the total alignment cost.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Minimization: finds minimum cost warping path
 *   - No gap penalty: cost comes from pairwise distances in the cost matrix
 *   - Soft version: uses temperature-scaled softmin instead of hard min
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 *   alpha[i,j] = costs[i,j] + softmin_T(
 *       alpha[i-1,j-1],  // diagonal: align both positions
 *       alpha[i-1,j],    // up: repeat element from seq1
 *       alpha[i,j-1]     // left: repeat element from seq2
 *   )
 *
 * Where softmin_T(a,b,c) = -T * log(exp(-a/T) + exp(-b/T) + exp(-c/T))
 *                        = min(a,b,c) - T * log(sum exp(-(x-min)/T))
 *
 * Base case:
 *   alpha[0,0] = 0
 *   alpha[i,0] = +inf for i > 0  (must start at origin)
 *   alpha[0,j] = +inf for j > 0  (must start at origin)
 *
 * Score:
 *   S = alpha[L1, L2]  (must end at terminal)
 *
 * ============================================================================
 * COMPARISON WITH NW/SW
 * ============================================================================
 *
 * | Property      | SW (local)    | NW (global)   | DTW           |
 * |---------------|---------------|---------------|---------------|
 * | Objective     | maximize      | maximize      | minimize      |
 * | Soft operator | logsumexp     | logsumexp     | softmin       |
 * | Input matrix  | similarity    | similarity    | distance/cost |
 * | Gap penalty   | explicit      | explicit      | none          |
 * | Transitions   | 4 (+ sky)     | 3             | 3             |
 * | Base case     | edges = -inf  | edges = i*gap | edges = +inf  |
 *
 * ============================================================================
 * SAKOE-CHIBA BANDWIDTH
 * ============================================================================
 *
 * Optional constraint limiting how far the warping path can deviate from
 * the diagonal. Useful for long sequences to reduce computation and prevent
 * pathological alignments.
 *
 * For cell (i,j), the expected diagonal position is j_expected = i * L2 / L1.
 * Cell is valid if |j - j_expected| <= bandwidth.
 *
 * Set bandwidth = -1 to disable the constraint.
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
 */

#pragma once

/**
 * Forward pass: compute alpha table and DTW distance.
 *
 * @param d_costs     Input cost matrix [B, L1, L2] (device)
 * @param d_alpha     Output DP table [B, (L1+1)*(L2+1)] (device)
 * @param d_score     Output DTW distance [B] (device)
 * @param d_lengths   Sequence lengths [B, 2] (device)
 * @param B           Batch size
 * @param max_L1      Maximum sequence 1 length (padded dimension)
 * @param max_L2      Maximum sequence 2 length (padded dimension)
 * @param T           Temperature (T->0: hard min, T->inf: uniform)
 * @param bandwidth   Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_forward(
    const float* d_costs, float* d_alpha, float* d_score,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float T, int bandwidth
);

/**
 * Backward pass: compute posteriors and temperature gradient.
 *
 * For DTW, posteriors = beta directly (cost is node-additive).
 *
 * @param d_alpha      Alpha table from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_costs      Input costs [B, L1, L2] (device)
 * @param d_score      DTW distance from forward [B] (device)
 * @param d_beta       Workspace: beta table [B, (L1+1)*(L2+1)] (device)
 * @param d_posteriors Output: soft alignment [B, L1, L2] (device)
 * @param d_grad_T     Output: dS/dT [B] (device)
 * @param d_lengths    Sequence lengths [B, 2] (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param T            Temperature
 * @param bandwidth    Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_backward(
    const float* d_alpha, const float* d_costs, const float* d_score,
    float* d_beta, float* d_posteriors, float* d_grad_T,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float T, int bandwidth
);

/**
 * Hessian-vector product: H * v where H = d^2S/dcosts^2.
 *
 * @param d_alpha      Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_costs      Input costs [B, L1, L2] (device)
 * @param d_score      DTW distance [B] (device)
 * @param d_V          Input vector [B, L1, L2] (device)
 * @param d_d_alpha    Workspace: dalpha [B, (L1+1)*(L2+1)] (device)
 * @param d_d_score    Workspace: dscore [B] (device)
 * @param d_beta       Workspace: beta [B, (L1+1)*(L2+1)] (device)
 * @param d_d_beta     Workspace: dbeta [B, (L1+1)*(L2+1)] (device)
 * @param d_H_costs    Output: H * v [B, L1, L2] (device)
 * @param d_lengths    Sequence lengths [B, 2] (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param T            Temperature
 * @param bandwidth    Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_hvp(
    const float* d_alpha, const float* d_costs, const float* d_score,
    const float* d_V, float* d_d_alpha, float* d_d_score,
    float* d_beta, float* d_d_beta, float* d_H_costs,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float T, int bandwidth
);

/**
 * Temperature Jacobian: dP/dT where P = posteriors.
 *
 * @param d_alpha      Alpha from forward [B, (L1+1)*(L2+1)] (device)
 * @param d_costs      Input costs [B, L1, L2] (device)
 * @param d_score      DTW distance [B] (device)
 * @param d_U          Workspace: dalpha/dT [B, (L1+1)*(L2+1)] (device)
 * @param d_beta       Workspace: beta [B, (L1+1)*(L2+1)] (device)
 * @param d_W          Workspace: dbeta/dT [B, (L1+1)*(L2+1)] (device)
 * @param d_dP_dT      Output: dP/dT [B, L1, L2] (device)
 * @param d_lengths    Sequence lengths [B, 2] (device)
 * @param B            Batch size
 * @param max_L1       Maximum sequence 1 length
 * @param max_L2       Maximum sequence 2 length
 * @param T            Temperature
 * @param bandwidth    Sakoe-Chiba bandwidth (-1 = no constraint)
 */
void dtw_param_grad(
    const float* d_alpha, const float* d_costs, const float* d_score,
    float* d_U, float* d_beta, float* d_W, float* d_dP_dT,
    const int* d_lengths,
    int B, int max_L1, int max_L2, float T, int bandwidth
);
