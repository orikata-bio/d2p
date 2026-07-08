// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft Damerau Operator Schema Registration
 *
 * Registers operator schemas for the d2p::soft_damerau family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Main autograd function with tensor parameters
    // Returns (damerau_score, posteriors) tuple
    m.def("soft_damerau(Tensor sub_costs, Tensor trans_src, Tensor ins_cost, Tensor del_cost, Tensor trans_cost, Tensor temperature, Tensor lengths) -> Tensor[]");

    // Convenience function with float parameters (no param gradient)
    m.def("soft_damerau_float(Tensor sub_costs, Tensor trans_src, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor[]");

    // Forward + backward with explicit outputs for debugging
    // Returns (damerau_score, posteriors, grad_T, grad_ins, grad_del, grad_trans)
    m.def("soft_damerau_with_grads(Tensor sub_costs, Tensor trans_src, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_damerau_hvp(Tensor sub_costs, Tensor trans_src, Tensor tangent, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor");

    // Parameter Jacobian: dP/d{param_type}
    // param_type: 0=ins, 1=del, 2=trans, 3=temperature
    m.def("soft_damerau_param_jacobian(Tensor sub_costs, Tensor trans_src, int param_type, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> Tensor");

    // Complete backward given grad_posteriors
    // Returns (grad_sub_costs, grad_temperature, grad_ins, grad_del, grad_trans)
    m.def("soft_damerau_backward_full(Tensor sub_costs, Tensor trans_src, Tensor grad_output, float ins_cost, float del_cost, float trans_cost, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

#endif // USE_TORCH_LIBRARY
