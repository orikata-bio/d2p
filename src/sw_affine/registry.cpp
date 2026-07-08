// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for Affine Gap Smith-Waterman operators
 *
 * Defines the operator schemas (m.def calls) for affine gap penalty SW.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * Using TORCH_LIBRARY_FRAGMENT allows splitting registrations across files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // =========================================================================
    // AFFINE SMITH-WATERMAN (Affine Gap Penalty)
    // =========================================================================
    //
    // Three-state DP (Match, Insert, Delete) with:
    //   M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
    //   I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)
    //   D[i,j] = LSE_T(M[i,j-1] + gap_open, D[i,j-1] + gap_ext)

    // Core operators (tensor params for full differentiability)
    m.def("soft_sw_affine(Tensor scores, Tensor gap_open, Tensor gap_ext, Tensor temperature, Tensor lengths) -> Tensor[]");
    m.def("soft_sw_affine_float(Tensor scores, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_sw_affine_with_grads(Tensor scores, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("soft_sw_affine_hvp(Tensor scores, Tensor tangent, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sw_affine_param_jacobian(Tensor scores, int param_type, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_sw_affine_backward_full(Tensor scores, Tensor grad_alignment, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");

    // Namespaced API (sw_affine_*)
    m.def("sw_affine_forward(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor[]");
    m.def("sw_affine_forward_t(Tensor scores, Tensor gap_open, Tensor gap_ext, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("sw_affine_value_grad_params(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("sw_affine_marginals_backward(Tensor scores, Tensor grad_marginals, float gap_open, float gap_ext, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("sw_affine_marginals_hvp(Tensor scores, Tensor v, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("sw_affine_marginals_grad_gap_open(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("sw_affine_marginals_grad_gap_ext(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("sw_affine_marginals_grad_temp(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
