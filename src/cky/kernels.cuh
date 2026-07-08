// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft CKY CUDA Kernel Declarations
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
 * Outside algorithm (backward):
 *   beta[i,j] = marginal probability of span [i,j] being in the parse
 *   Pcond[i,k,j] = P(split at k | span [i,j])
 *   Pjoint[i,k,j] = beta[i,j] * Pcond[i,k,j]
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

/**
 * Forward pass: compute inside values (chart) and partition function.
 *
 * @param d_merge_scores  Merge scores [B, n, n, n] (device)
 * @param d_leaf_scores   Leaf scores [B, n] (device)
 * @param d_Z             Output: inside chart [B, n, n] (device)
 * @param d_partition     Output: log partition function [B] (device)
 * @param d_T             Temperature tensor [1] or [B, n, n] (device)
 * @param B               Batch size
 * @param n               Sequence length
 * @param T_is_scalar     Whether d_T stores a single scalar value
 */
void cky_forward(
    const float* d_merge_scores,
    const float* d_leaf_scores,
    float* d_Z,
    float* d_partition,
    const float* d_T,
    int B, int n,
    bool T_is_scalar
);

/**
 * Backward pass: compute outside values, posteriors, and gradients.
 *
 * @param d_Z             Inside chart from forward [B, n, n] (device)
 * @param d_merge_scores  Merge scores [B, n, n, n] (device)
 * @param d_leaf_scores   Leaf scores [B, n] (device, unused)
 * @param d_partition     Partition function [B] (device, unused)
 * @param d_beta          Output: outside values [B, n, n] (device)
 * @param d_Pcond         Output: conditional split probs [B, n, n, n] (device)
 * @param d_Pjoint        Output: joint split probs [B, n, n, n] (device)
 * @param d_grad_merge    Output: grad w.r.t. merge (same as Pjoint) [B, n, n, n] (device)
 * @param d_grad_leaf     Output: grad w.r.t. leaf [B, n] (device)
 * @param d_grad_T        Output: grad w.r.t. temperature [B] or [B, n, n] (device)
 * @param d_T             Temperature tensor [1] or [B, n, n] (device)
 * @param B               Batch size
 * @param n               Sequence length
 * @param T_is_scalar     Whether d_T stores a single scalar value
 */
void cky_backward(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    float* d_beta,
    float* d_Pcond,
    float* d_Pjoint,
    float* d_grad_merge,
    float* d_grad_leaf,
    float* d_grad_T,
    const float* d_T,
    int B, int n,
    bool T_is_scalar
);

/**
 * Hessian-vector product: H * v where H = d^2logZ/d(scores)^2.
 *
 * @param d_Z             Inside chart [B, n, n] (device)
 * @param d_merge_scores  Merge scores [B, n, n, n] (device)
 * @param d_leaf_scores   Leaf scores [B, n] (device, unused)
 * @param d_partition     Partition function [B] (device, unused)
 * @param d_V_merge       Tangent vector for merge [B, n, n, n] (device)
 * @param d_V_leaf        Tangent vector for leaf [B, n] (device)
 * @param d_d_Z           Workspace: tangent inside [B, n, n] (device)
 * @param d_d_partition   Workspace: tangent partition [B] (device)
 * @param d_beta          Workspace: outside values [B, n, n] (device)
 * @param d_d_beta        Workspace: tangent outside [B, n, n] (device)
 * @param d_HVP_merge     Output: HVP for merge [B, n, n, n] (device)
 * @param d_HVP_leaf      Output: HVP for leaf [B, n] (device)
 * @param B               Batch size
 * @param n               Sequence length
 * @param T               Temperature
 */
void cky_hvp(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    const float* d_V_merge,
    const float* d_V_leaf,
    float* d_d_Z,
    float* d_d_partition,
    float* d_beta,
    float* d_d_beta,
    float* d_HVP_merge,
    float* d_HVP_leaf,
    int B, int n, float T
);

/**
 * Temperature Jacobian: dP/dT where P = posteriors.
 *
 * @param d_Z             Inside chart [B, n, n] (device)
 * @param d_merge_scores  Merge scores [B, n, n, n] (device)
 * @param d_leaf_scores   Leaf scores [B, n] (device, unused)
 * @param d_partition     Partition function [B] (device, unused)
 * @param d_U             Workspace: dZ/dT [B, n, n] (device)
 * @param d_beta          Workspace: outside values [B, n, n] (device)
 * @param d_W             Workspace: d_beta/dT [B, n, n] (device)
 * @param d_dP_dT_merge   Output: dP/dT for merge [B, n, n, n] (device)
 * @param d_dP_dT_leaf    Output: dP/dT for leaf [B, n] (device)
 * @param B               Batch size
 * @param n               Sequence length
 * @param T               Temperature
 */
void cky_param_grad(
    const float* d_Z,
    const float* d_merge_scores,
    const float* d_leaf_scores,
    const float* d_partition,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT_merge,
    float* d_dP_dT_leaf,
    int B, int n, float T
);
