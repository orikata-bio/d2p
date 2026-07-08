// SPDX-License-Identifier: Apache-2.0
/**
 * @file registry.cpp
 * @brief Soft Eisner PyTorch Library Registration
 *
 * Defines the operator schemas for differentiable Eisner dependency parsing.
 */

#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    m.def("soft_eisner(Tensor arc_scores, Tensor temperature, Tensor? lengths) -> Tensor[]");
    m.def("soft_eisner_float(Tensor arc_scores, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_eisner_with_grads(Tensor arc_scores, float temperature, Tensor? lengths) -> (Tensor, Tensor)");
    m.def("soft_eisner_hvp(Tensor arc_scores, Tensor V, float temperature, Tensor? lengths) -> Tensor");
    m.def("soft_eisner_backward_full(Tensor arc_scores, float temperature, Tensor? lengths) -> (Tensor, Tensor, Tensor)");
    m.def("soft_eisner_param_jacobian(Tensor arc_scores, float temperature, Tensor? lengths) -> Tensor");
}

#endif // USE_TORCH_LIBRARY
