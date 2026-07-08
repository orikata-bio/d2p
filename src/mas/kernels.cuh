// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels.cuh
 * @brief Soft Monotonic Alignment Search (MAS) CUDA Kernel Declarations
 *
 * MAS aligns frames (T) to text positions (S) with monotonic constraint.
 * Used in TTS/ASR systems.
 *
 * Key properties:
 *   - Each frame aligns to exactly one text position
 *   - Alignment must be monotonic (can only move forward in text)
 *   - All text must be covered (must end at S-1)
 *
 * Two transitions only:
 *   - Stay: alpha(t-1, s) - same text token
 *   - Diag: alpha(t-1, s-1) - next text token
 *
 * Recurrence:
 *   alpha(t, s) = score(t, s) + LSE_T(alpha(t-1, s), alpha(t-1, s-1))
 *
 * Shapes:
 *   scores:     [B, T, S]  - frame-to-text similarity
 *   alpha:      [B, T, S]  - DP table
 *   partition:  [B]        - alignment score
 *   posteriors: [B, T, S]  - P(frame t aligns to text s)
 *   lengths:    [B, 2]     - (T, S) per batch element
 */

#pragma once

namespace d2p {
namespace mas {

/// Negative infinity for log-domain computations
constexpr float NINF = -1e30f;

/**
 * Forward pass: compute alpha table and partition function.
 *
 * @param d_scores    [B, T, S] input similarity scores
 * @param d_alpha     [B, T, S] output DP table
 * @param d_partition [B] output partition function (final score)
 * @param d_lengths   [B, 2] sequence lengths (T, S)
 * @param B           batch size
 * @param max_T       max frames
 * @param max_S       max text length
 * @param temperature softmax temperature
 */
void forward(
    const float* d_scores,
    float* d_alpha,
    float* d_partition,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Backward pass: compute posteriors and temperature gradient.
 *
 * @param d_alpha      [B, T, S] alpha table from forward
 * @param d_scores     [B, T, S] input scores
 * @param d_partition  [B] partition function
 * @param d_beta       [B, T, S] workspace for beta values
 * @param d_posteriors [B, T, S] output posteriors
 * @param d_grad_T     [B] output temperature gradient
 * @param d_lengths    [B, 2] sequence lengths
 * @param B            batch size
 * @param max_T        max frames
 * @param max_S        max text length
 * @param temperature  softmax temperature
 */
void backward(
    const float* d_alpha,
    const float* d_scores,
    const float* d_partition,
    float* d_beta,
    float* d_posteriors,
    float* d_grad_T,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Hessian-vector product for second-order gradients.
 *
 * @param d_alpha     [B, T, S] alpha table from forward
 * @param d_scores    [B, T, S] input scores
 * @param d_V         [B, T, S] tangent vector
 * @param d_d_alpha   [B, T, S] workspace for tangent of alpha
 * @param d_d_score   [B] tangent of partition function
 * @param d_beta      [B, T, S] workspace for beta
 * @param d_d_beta    [B, T, S] workspace for tangent of beta
 * @param d_H_scores  [B, T, S] output HVP
 * @param d_lengths   [B, 2] sequence lengths
 * @param B           batch size
 * @param max_T       max frames
 * @param max_S       max text length
 * @param temperature softmax temperature
 */
void hvp(
    const float* d_alpha,
    const float* d_scores,
    const float* d_V,
    float* d_d_alpha,
    float* d_d_score,
    float* d_beta,
    float* d_d_beta,
    float* d_H_scores,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Parameter gradient: dP/dT (gradient of posteriors w.r.t. temperature).
 *
 * @param d_alpha    [B, T, S] alpha table from forward
 * @param d_scores   [B, T, S] input scores
 * @param d_U        [B, T, S] workspace
 * @param d_beta     [B, T, S] workspace for beta
 * @param d_W        [B, T, S] workspace
 * @param d_dP_dT    [B, T, S] output parameter gradient
 * @param d_lengths  [B, 2] sequence lengths
 * @param B          batch size
 * @param max_T      max frames
 * @param max_S      max text length
 * @param temperature softmax temperature
 */
void param_grad(
    const float* d_alpha,
    const float* d_scores,
    float* d_U,
    float* d_beta,
    float* d_W,
    float* d_dP_dT,
    const int* d_lengths,
    int B, int max_T, int max_S,
    float temperature
);

} // namespace mas
} // namespace d2p
