// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Eisner CPU Kernel Declarations
 *
 * Differentiable Eisner algorithm for projective dependency parsing.
 * Uses 4-state DP with complete/incomplete span decomposition.
 *
 * Table structure:
 *   C_R[i,j]: Complete span [i,j] with head at i (pointing right)
 *   C_L[i,j]: Complete span [i,j] with head at j (pointing left)
 *   I_R[i,j]: Incomplete span with arc i->j
 *   I_L[i,j]: Incomplete span with arc j->i
 */

#pragma once

namespace d2p {
namespace eisner {
namespace cpu {

constexpr float NINF = -1e30f;
constexpr int MAX_EISNER_N_FOR_INT_CELL_INDEX = 46340;

/**
 * Forward pass - compute DP tables and partition function
 *
 * @param arc_scores  Arc scores [B, n, n] where arc[i,j] = score of i->j
 * @param C_R         Complete right table [B, n, n]
 * @param C_L         Complete left table [B, n, n]
 * @param I_R         Incomplete right table [B, n, n]
 * @param I_L         Incomplete left table [B, n, n]
 * @param partition   Partition function output [B]
 * @param lengths     Sequence lengths [B] or nullptr
 * @param B           Batch size
 * @param n           Max sequence length
 * @param temperature Softmax temperature
 */
void forward(
    const float* arc_scores,
    float* C_R,
    float* C_L,
    float* I_R,
    float* I_L,
    float* partition,
    const int* lengths,
    int B, int n, float temperature
);

/**
 * Backward pass - compute marginals and temperature gradient
 *
 * @param arc_scores  Arc scores [B, n, n]
 * @param C_R         Complete right table [B, n, n]
 * @param C_L         Complete left table [B, n, n]
 * @param I_R         Incomplete right table [B, n, n]
 * @param I_L         Incomplete left table [B, n, n]
 * @param beta_C_R    Beta for C_R [B, n, n]
 * @param beta_C_L    Beta for C_L [B, n, n]
 * @param beta_I_R    Beta for I_R [B, n, n]
 * @param beta_I_L    Beta for I_L [B, n, n]
 * @param marginals   Arc marginals output [B, n, n]
 * @param grad_T      Temperature gradient output [B]
 * @param lengths     Sequence lengths [B] or nullptr
 * @param B           Batch size
 * @param n           Max sequence length
 * @param temperature Softmax temperature
 */
void backward(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* marginals,
    float* grad_T,
    const int* lengths,
    int B, int n, float temperature
);

/**
 * Hessian-vector product for second-order optimization
 *
 * @param arc_scores  Arc scores [B, n, n]
 * @param V           Tangent vector [B, n, n]
 * @param C_R         Complete right table [B, n, n]
 * @param C_L         Complete left table [B, n, n]
 * @param I_R         Incomplete right table [B, n, n]
 * @param I_L         Incomplete left table [B, n, n]
 * @param d_C_R       Tangent for C_R [B, n, n]
 * @param d_C_L       Tangent for C_L [B, n, n]
 * @param d_I_R       Tangent for I_R [B, n, n]
 * @param d_I_L       Tangent for I_L [B, n, n]
 * @param beta_C_R    Beta for C_R [B, n, n]
 * @param beta_C_L    Beta for C_L [B, n, n]
 * @param beta_I_R    Beta for I_R [B, n, n]
 * @param beta_I_L    Beta for I_L [B, n, n]
 * @param d_beta_C_R  Tangent beta for C_R [B, n, n]
 * @param d_beta_C_L  Tangent beta for C_L [B, n, n]
 * @param d_beta_I_R  Tangent beta for I_R [B, n, n]
 * @param d_beta_I_L  Tangent beta for I_L [B, n, n]
 * @param HVP         Hessian-vector product output [B, n, n]
 * @param lengths     Sequence lengths [B] or nullptr
 * @param B           Batch size
 * @param n           Max sequence length
 * @param temperature Softmax temperature
 */
void hvp(
    const float* arc_scores,
    const float* V,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* d_C_R,
    float* d_C_L,
    float* d_I_R,
    float* d_I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* d_beta_C_R,
    float* d_beta_C_L,
    float* d_beta_I_R,
    float* d_beta_I_L,
    float* HVP,
    const int* lengths,
    int B, int n, float temperature
);

/**
 * Parameter gradient - compute dP/dT (derivative of arc marginals w.r.t. temperature)
 *
 * @param arc_scores  Arc scores [B, n, n]
 * @param C_R         Complete right table [B, n, n] (from forward)
 * @param C_L         Complete left table [B, n, n] (from forward)
 * @param I_R         Incomplete right table [B, n, n] (from forward)
 * @param I_L         Incomplete left table [B, n, n] (from forward)
 * @param dP_dT       Parameter Jacobian output [B, n, n]
 * @param lengths     Sequence lengths [B] or nullptr
 * @param B           Batch size
 * @param n           Max sequence length
 * @param temperature Softmax temperature
 */
void param_grad(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* dP_dT,
    const int* lengths,
    int B, int n, float temperature
);

} // namespace cpu
} // namespace eisner
} // namespace d2p
