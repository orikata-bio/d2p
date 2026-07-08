// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft MAS Operator Schema Registration
 *
 * Registers operator schemas for the d2p::soft_mas family of functions.
 * CPU and CUDA implementations are registered in their respective files.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Main function returning posteriors
    m.def("soft_mas(Tensor scores, float temperature, Tensor? lengths=None) -> Tensor");

    // Autograd-wrapped version returning partition function
    m.def("soft_mas_float(Tensor scores, float temperature, Tensor? lengths=None) -> Tensor[]");

    // Forward + backward returning (partition, posteriors)
    m.def("soft_mas_with_grads(Tensor scores, float temperature, Tensor? lengths=None) -> (Tensor, Tensor)");

    // Hessian-vector product
    m.def("soft_mas_hvp(Tensor scores, Tensor V, float temperature, Tensor? lengths=None) -> Tensor");

    // Parameter Jacobian (dP/dT)
    m.def("soft_mas_param_jacobian(Tensor scores, float temperature, Tensor? lengths=None) -> Tensor");

    // Full backward returning (partition, posteriors, grad_T)
    m.def("soft_mas_backward_full(Tensor scores, float temperature, Tensor? lengths=None) -> (Tensor, Tensor, Tensor)");
}

#endif // USE_TORCH_LIBRARY
