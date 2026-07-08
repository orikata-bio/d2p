// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for Needleman-Wunsch operators (linear gap)
 *
 * Defines the operator schemas (m.def calls) for linear gap penalty NW.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * Using TORCH_LIBRARY_FRAGMENT allows splitting registrations across files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // =========================================================================
    // NEEDLEMAN-WUNSCH (Linear Gap Penalty)
    // =========================================================================
    //
    // Global alignment: aligns full sequences end-to-end.
    // Single-state DP with recurrence:
    //   alpha[i,j] = LSE_T(
    //       alpha[i-1,j-1] + scores[i,j],   // align
    //       alpha[i-1,j] + gap,              // gap in seq2
    //       alpha[i,j-1] + gap               // gap in seq1
    //   )
    //
    // Key differences from Smith-Waterman:
    //   - No "start new alignment" option (global, not local)
    //   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
    //   - Score = alpha[L1,L2] at terminal (not logsumexp over all cells)

    // Core operators (tensor params for full differentiability)
    m.def("soft_nw(Tensor scores, Tensor gap, Tensor temperature, Tensor lengths) -> Tensor[]");
    m.def("soft_nw_float(Tensor scores, float gap, float temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_nw_with_grads(Tensor scores, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("soft_nw_hvp(Tensor scores, Tensor tangent, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_nw_param_jacobian(Tensor scores, int param_type, float gap, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_nw_backward_full(Tensor scores, Tensor grad_alignment, float gap, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");

    // Namespaced API (nw_*)
    m.def("nw_forward(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor[]");
    m.def("nw_forward_t(Tensor scores, Tensor gap, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("nw_value_grad_params(Tensor scores, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("nw_marginals_backward(Tensor scores, Tensor grad_marginals, float gap, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("nw_marginals_hvp(Tensor scores, Tensor v, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("nw_marginals_grad_gap(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
    m.def("nw_marginals_grad_temp(Tensor scores, float gap, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
