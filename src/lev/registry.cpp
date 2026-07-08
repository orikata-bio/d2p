// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft Levenshtein Operator Schema Registration
 *
 * Registers operator schemas for the d2p::soft_levenshtein family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Main autograd function with tensor parameters
    // Returns (distance, posteriors) tuple
    m.def("soft_levenshtein(Tensor scores, Tensor ins_cost, Tensor del_cost, Tensor temperature, Tensor lengths) -> Tensor[]");

    // Convenience function with float parameters (no param gradient)
    m.def("soft_levenshtein_float(Tensor scores, float ins_cost, float del_cost, float temperature, Tensor? lengths) -> Tensor[]");

    // Forward + backward with explicit outputs for debugging
    // Returns (distance, posteriors, grad_ins, grad_del, grad_T)
    m.def("soft_levenshtein_with_grads(Tensor scores, float ins_cost, float del_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_levenshtein_hvp(Tensor scores, Tensor tangent, float ins_cost, float del_cost, float temperature, Tensor? lengths) -> Tensor");

    // Parameter Jacobian: dP/d{param_type}
    // param_type: 0=ins, 1=del, 2=temperature
    m.def("soft_levenshtein_param_jacobian(Tensor scores, int param_type, float ins_cost, float del_cost, float temperature, Tensor? lengths) -> Tensor");

    // Complete backward given grad_posteriors
    // Returns (grad_scores, grad_ins, grad_del, grad_temperature)
    m.def("soft_levenshtein_backward_full(Tensor scores, Tensor grad_posteriors, float ins_cost, float del_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor)");
}

#endif // USE_TORCH_LIBRARY
