// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for Soft DTW operators
 *
 * Defines the operator schemas (m.def calls) for soft dynamic time warping.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * Using TORCH_LIBRARY_FRAGMENT allows splitting registrations across files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // =========================================================================
    // SOFT DTW (Dynamic Time Warping)
    // =========================================================================
    //
    // Minimization DP using softmin with recurrence:
    //   alpha[i,j] = costs[i,j] + softmin_T(
    //       alpha[i-1,j-1],  // diagonal
    //       alpha[i-1,j],    // up
    //       alpha[i,j-1]     // left
    //   )
    //
    // Unlike SW/NW:
    //   - Input is a COST matrix (lower = better), not similarity
    //   - Uses softmin instead of logsumexp (minimization)
    //   - No gap penalty - cost comes from the matrix
    //   - Optional Sakoe-Chiba bandwidth constraint

    // Core operators (tensor params for full differentiability)
    m.def("soft_dtw(Tensor costs, Tensor temperature, Tensor lengths, int bandwidth) -> Tensor[]");
    m.def("soft_dtw_float(Tensor costs, float temperature, Tensor? lengths, int? bandwidth) -> Tensor[]");
    m.def("soft_dtw_with_grads(Tensor costs, float temperature, Tensor? lengths, int? bandwidth) -> (Tensor, Tensor, Tensor)");
    m.def("soft_dtw_hvp(Tensor costs, Tensor tangent, float temperature, Tensor? lengths, int? bandwidth) -> Tensor");
    m.def("soft_dtw_param_jacobian(Tensor costs, float temperature, Tensor? lengths, int? bandwidth) -> Tensor");
    m.def("soft_dtw_backward_full(Tensor costs, Tensor grad_alignment, float temperature, Tensor? lengths, int? bandwidth) -> (Tensor, Tensor)");
}

#endif // USE_TORCH_LIBRARY
