// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for Regular Smith-Waterman operators
 *
 * Defines the operator schemas (m.def calls) for linear gap penalty SW.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * Using TORCH_LIBRARY_FRAGMENT allows splitting registrations across files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // =========================================================================
    // REGULAR SMITH-WATERMAN (Linear Gap Penalty)
    // =========================================================================
    //
    // Single-state DP with recurrence:
    //   alpha[i,j] = LSE_T(
    //       alpha[i-1,j-1] + scores[i,j],   // align
    //       alpha[i-1,j] + gap,              // gap in seq2
    //       alpha[i,j-1] + gap,              // gap in seq1
    //       0                                 // start new alignment
    //   )

    // Core operators (tensor params for full differentiability)
    m.def("soft_sw(Tensor scores, Tensor gap, Tensor temperature, Tensor lengths) -> Tensor[]");
    m.def("soft_sw_float(Tensor scores, float gap, float temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_sw_with_grads(Tensor scores, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("soft_sw_hvp(Tensor scores, Tensor tangent, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sw_param_jacobian(Tensor scores, int param_type, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sw_backward_full(Tensor scores, Tensor grad_alignment, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");

    // Namespaced API (sw_*)
    m.def("sw_forward(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor[]");
    m.def("sw_forward_t(Tensor scores, Tensor gap, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("sw_value_grad_params(Tensor scores, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("sw_marginals_backward(Tensor scores, Tensor grad_marginals, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("sw_marginals_hvp(Tensor scores, Tensor v, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("sw_marginals_grad_gap(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("sw_marginals_grad_temp(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
