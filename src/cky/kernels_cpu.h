// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft CKY CPU Kernel Declarations
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * CKY (Cocke-Kasami-Younger) parsing computes the partition function over
 * all binary parse trees for a sequence. The soft version uses temperature-
 * scaled logsumexp to make the computation differentiable.
 *
 * Key properties:
 *   - Constituency parsing: builds binary trees over sequences
 *   - Maximization: finds highest-scoring parse (like SW/NW)
 *   - Soft version: uses logsumexp for differentiability
 *   - Span-based DP: fills chart by increasing span width
 *
 * ============================================================================
 * RECURRENCE RELATION
 * ============================================================================
 *
 * Inside algorithm (forward):
 *   Z[i,j] = logsumexp_T over k in [i, j-1]:
 *       Z[i,k] + Z[k+1,j] + merge_scores[i,k,j]
 *
 * Base case:
 *   Z[i,i] = leaf_scores[i]  (single element spans)
 *
 * Partition function:
 *   logZ = Z[0, n-1]  (full sentence span)
 *
 * ============================================================================
 * MEMORY LAYOUT
 * ============================================================================
 *
 * merge_scores: [B, n, n, n] - merge_scores[b, i, k, j] for combining [i,k] and [k+1,j]
 * leaf_scores:  [B, n]       - leaf_scores[b, i] for leaf span [i,i]
 * Z (chart):    [B, n, n]    - inside values (upper triangular)
 * beta:         [B, n, n]    - outside values / span marginals
 * Pcond:        [B, n, n, n] - conditional split posteriors P(k|i,j)
 * Pjoint:       [B, n, n, n] - joint split posteriors beta[i,j] * P(k|i,j)
 *
 * ============================================================================
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Forward pass: compute inside values (chart) and partition function.
 *
 * @param merge_scores  Merge scores [B, n, n, n]
 * @param leaf_scores   Leaf scores [B, n]
 * @param Z             Output: inside chart [B, n, n]
 * @param partition     Output: log partition function [B]
 * @param T             Temperature tensor [1] or [B, n, n]
 * @param B             Batch size
 * @param n             Sequence length
 * @param T_is_scalar   Whether T stores a single scalar value
 */
void cky_forward_cpu(
    const float* merge_scores,
    const float* leaf_scores,
    float* Z,
    float* partition,
    const float* T,
    int B, int n,
    bool T_is_scalar
);

/**
 * Backward pass: compute outside values, posteriors, and gradients.
 *
 * @param Z             Inside chart from forward [B, n, n]
 * @param merge_scores  Merge scores [B, n, n, n]
 * @param leaf_scores   Leaf scores [B, n] (unused)
 * @param partition     Partition function [B] (unused)
 * @param beta          Output: outside values [B, n, n]
 * @param Pcond         Output: conditional split probs [B, n, n, n]
 * @param Pjoint        Output: joint split probs [B, n, n, n]
 * @param grad_merge    Output: grad w.r.t. merge [B, n, n, n]
 * @param grad_leaf     Output: grad w.r.t. leaf [B, n]
 * @param grad_T        Output: grad w.r.t. temperature [B] or [B, n, n]
 * @param T             Temperature tensor [1] or [B, n, n]
 * @param B             Batch size
 * @param n             Sequence length
 * @param T_is_scalar   Whether T stores a single scalar value
 */
void cky_backward_cpu(
    const float* Z,
    const float* merge_scores,
    const float* leaf_scores,
    const float* partition,
    float* beta,
    float* Pcond,
    float* Pjoint,
    float* grad_merge,
    float* grad_leaf,
    float* grad_T,
    const float* T,
    int B, int n,
    bool T_is_scalar
);

/**
 * Hessian-vector product: H * v where H = d^2 logZ/d(scores)^2.
 *
 * @param Z             Inside chart [B, n, n]
 * @param merge_scores  Merge scores [B, n, n, n]
 * @param leaf_scores   Leaf scores [B, n] (unused)
 * @param partition     Partition function [B] (unused)
 * @param V_merge       Tangent vector for merge [B, n, n, n]
 * @param V_leaf        Tangent vector for leaf [B, n]
 * @param d_Z           Workspace: tangent inside [B, n, n]
 * @param d_partition   Workspace: tangent partition [B]
 * @param beta          Workspace: outside values [B, n, n]
 * @param d_beta        Workspace: tangent outside [B, n, n]
 * @param HVP_merge     Output: HVP for merge [B, n, n, n]
 * @param HVP_leaf      Output: HVP for leaf [B, n]
 * @param B             Batch size
 * @param n             Sequence length
 * @param T             Temperature
 */
void cky_hvp_cpu(
    const float* Z,
    const float* merge_scores,
    const float* leaf_scores,
    const float* partition,
    const float* V_merge,
    const float* V_leaf,
    float* d_Z,
    float* d_partition,
    float* beta,
    float* d_beta,
    float* HVP_merge,
    float* HVP_leaf,
    int B, int n, float T
);

/**
 * Temperature Jacobian: dP/dT where P = posteriors.
 *
 * @param Z             Inside chart [B, n, n]
 * @param merge_scores  Merge scores [B, n, n, n]
 * @param leaf_scores   Leaf scores [B, n] (unused)
 * @param partition     Partition function [B] (unused)
 * @param dP_dT_merge   Output: dP/dT for merge [B, n, n, n]
 * @param dP_dT_leaf    Output: dP/dT for leaf [B, n]
 * @param B             Batch size
 * @param n             Sequence length
 * @param T             Temperature
 */
void cky_param_grad_cpu(
    const float* Z,
    const float* merge_scores,
    const float* leaf_scores,
    const float* partition,
    float* dP_dT_merge,
    float* dP_dT_leaf,
    int B, int n, float T
);

#ifdef __cplusplus
}
#endif
