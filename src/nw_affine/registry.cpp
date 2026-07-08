// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Schema definitions for Needleman-Wunsch Affine operators
 *
 * Defines the operator schemas (m.def calls) for affine gap penalty NW.
 * Implementations are registered in torch_cuda.cpp and torch_cpu.cpp.
 *
 * Using TORCH_LIBRARY_FRAGMENT allows splitting registrations across files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // =========================================================================
    // NEEDLEMAN-WUNSCH AFFINE (Three-State DP)
    // =========================================================================
    //
    // Global alignment with affine gap penalty: gap_open + (k-1)*gap_ext for k gaps.
    // Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
    //
    // Recurrences:
    //   M[i,j] = score[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1])
    //   I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext, D[i-1,j] + gap_open)
    //   D[i,j] = LSE_T(M[i,j-1] + gap_open, I[i,j-1] + gap_open, D[i,j-1] + gap_ext)
    //
    // Key differences from Smith-Waterman affine:
    //   - No "sky" restart (global, not local alignment)
    //   - Base cases: M(0,0)=0, I(i,0)=g_o+(i-1)*g_e, D(0,j)=g_o+(j-1)*g_e
    //   - Score = LSE(M[L1,L2], I[L1,L2], D[L1,L2]) at terminal

    // Core operators (tensor params for full differentiability)
    m.def("soft_nw_affine(Tensor scores, Tensor gap_open, Tensor gap_ext, Tensor temperature, Tensor lengths) -> Tensor[]");
    m.def("soft_nw_affine_float(Tensor scores, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_nw_affine_with_grads(Tensor scores, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("soft_nw_affine_hvp(Tensor scores, Tensor tangent, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_nw_affine_param_jacobian(Tensor scores, int param_type, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_nw_affine_backward_full(Tensor scores, Tensor grad_alignment, float gap_open, float gap_ext, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");

    // Namespaced API (nw_affine_*)
    m.def("nw_affine_forward(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor[]");
    m.def("nw_affine_forward_t(Tensor scores, Tensor gap_open, Tensor gap_ext, Tensor temp, Tensor lengths) -> Tensor[]");
    m.def("nw_affine_value_grad_params(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("nw_affine_marginals_backward(Tensor scores, Tensor grad_marginals, float gap_open, float gap_ext, float temp, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("nw_affine_marginals_hvp(Tensor scores, Tensor v, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("nw_affine_marginals_grad_gap_open(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("nw_affine_marginals_grad_gap_ext(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
    m.def("nw_affine_marginals_grad_temp(Tensor scores, float gap_open, float gap_ext, float temp, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
