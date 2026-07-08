// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.h
 * @brief Soft Monotonic Alignment Search (MAS) CPU Kernel Declarations
 *
 * CPU implementation of MAS for TTS/ASR alignment.
 */

#pragma once

namespace d2p {
namespace mas {
namespace cpu {

/// Negative infinity for log-domain computations
constexpr float NINF = -1e30f;

/**
 * Forward pass: compute alpha table and partition function.
 */
void forward(
    const float* scores,
    float* alpha,
    float* partition,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Backward pass: compute posteriors and temperature gradient.
 */
void backward(
    const float* alpha,
    const float* scores,
    const float* partition,
    float* beta,
    float* posteriors,
    float* grad_T,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Hessian-vector product for second-order gradients.
 */
void hvp(
    const float* alpha,
    const float* scores,
    const float* V,
    float* d_alpha,
    float* d_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
);

/**
 * Parameter gradient: dP/dT.
 */
void param_grad(
    const float* alpha,
    const float* scores,
    float* U,
    float* beta,
    float* W,
    float* dP_dT,
    const int* lengths,
    int B, int max_T, int max_S,
    float temperature
);

} // namespace cpu
} // namespace mas
} // namespace d2p
