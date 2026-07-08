// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft CKY CPU Extension with PyTorch Autograd
 *
 * CPU implementations that mirror the CUDA interface.
 * Registered with TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * Operations:
 *   - soft_cky: Main function returning (logZ, posteriors) with full autograd
 *   - soft_cky_float: Same but with float temperature (no temp gradient)
 *   - soft_cky_with_grads: Forward + backward with explicit outputs for debugging
 *   - soft_cky_hvp: Hessian-vector product
 *   - soft_cky_param_jacobian: dP/dT (posterior derivative w.r.t. temperature)
 *   - soft_cky_backward_full: Complete backward given grad_posteriors
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"

// CPU kernel declarations
#include "cky/kernels_cpu.h"

using namespace d2p::common;

namespace {

constexpr int64_t kMaxCkyCpuNForIntChartIndex = 46340;  // floor(sqrt(INT_MAX))

void validate_cky_chart_shapes_cpu(
    const torch::Tensor& merge_scores,
    const torch::Tensor& leaf_scores
) {
    TORCH_CHECK(merge_scores.dim() == 4, "merge_scores must be 4D [B, n, n, n]");
    TORCH_CHECK(leaf_scores.dim() == 2, "leaf_scores must be 2D [B, n]");

    const int64_t B = merge_scores.size(0);
    const int64_t n = merge_scores.size(1);

    TORCH_CHECK(
        B <= std::numeric_limits<int>::max(),
        "soft_cky CPU batch size is too large"
    );
    TORCH_CHECK(n > 0, "soft_cky requires n > 0");
    TORCH_CHECK(
        n <= kMaxCkyCpuNForIntChartIndex,
        "soft_cky CPU sequence length is too large"
    );
    TORCH_CHECK(
        merge_scores.size(2) == n && merge_scores.size(3) == n,
        "merge_scores must be [B, n, n, n]"
    );
    TORCH_CHECK(
        leaf_scores.size(0) == B && leaf_scores.size(1) == n,
        "leaf_scores must be [B, n]"
    );
}

void validate_cky_temperature_tensor_cpu(
    const torch::Tensor& temperature,
    int64_t B,
    int64_t n
) {
    const bool T_is_scalar = temperature.numel() == 1;
    if (!T_is_scalar) {
        TORCH_CHECK(
            temperature.dim() == 3 &&
            temperature.size(0) == B &&
            temperature.size(1) == n &&
            temperature.size(2) == n,
            "temperature must be scalar or [B, n, n]"
        );
    }
}

}  // namespace

// =============================================================================
// CKY CPU Autograd Function
//
// Forward: merge_scores, leaf_scores, temperature -> (logZ, posteriors)
// Backward: uses HVP for grad w.r.t. merge_scores and leaf_scores
//           uses param_grad for grad w.r.t. temperature
// =============================================================================

class SoftCKYCPUFunction : public torch::autograd::Function<SoftCKYCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor merge_scores,
        torch::Tensor leaf_scores,
        torch::Tensor temperature
    ) {
        D2P_CHECK_INPUT_CPU(merge_scores);
        D2P_CHECK_INPUT_CPU(leaf_scores);
        D2P_CHECK_INPUT_CPU(temperature);
        TORCH_CHECK(merge_scores.dtype() == torch::kFloat32, "merge_scores must be float32");
        TORCH_CHECK(leaf_scores.dtype() == torch::kFloat32, "leaf_scores must be float32");
        TORCH_CHECK(temperature.dtype() == torch::kFloat32, "temperature must be float32");

        validate_cky_chart_shapes_cpu(merge_scores, leaf_scores);
        const int B = static_cast<int>(merge_scores.size(0));
        const int n = static_cast<int>(merge_scores.size(1));

        torch::Tensor temperature_contig = temperature.contiguous();
        validate_cky_temperature_tensor_cpu(temperature_contig, B, n);
        bool T_is_scalar = temperature_contig.numel() == 1;
        auto options = merge_scores.options();
        ctx->set_materialize_grads(false);

        // Allocate outputs
        torch::Tensor Z = torch::zeros({B, n, n}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, n, n}, options);
        torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
        torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_leaf = torch::zeros({B, n}, options);
        torch::Tensor grad_T = T_is_scalar
            ? torch::zeros({B}, options)
            : torch::zeros({B, n, n}, options);

        // Forward pass (inside algorithm)
        cky_forward_cpu(
            merge_scores.data_ptr<float>(),
            leaf_scores.data_ptr<float>(),
            Z.data_ptr<float>(),
            partition.data_ptr<float>(),
            temperature_contig.data_ptr<float>(),
            B, n, T_is_scalar
        );

        // Backward pass of DP (outside algorithm) - computes posteriors
        cky_backward_cpu(
            Z.data_ptr<float>(),
            merge_scores.data_ptr<float>(),
            leaf_scores.data_ptr<float>(),
            partition.data_ptr<float>(),
            beta.data_ptr<float>(),
            Pcond.data_ptr<float>(),
            Pjoint.data_ptr<float>(),
            grad_merge.data_ptr<float>(),
            grad_leaf.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            temperature_contig.data_ptr<float>(),
            B, n, T_is_scalar
        );

        // Save for backward (clone to prevent memory reuse issues)
        ctx->save_for_backward({merge_scores.clone(), leaf_scores.clone(), Z.clone(),
                               partition.clone(), grad_leaf.clone(), grad_T.clone(), temperature_contig.clone()});
        ctx->saved_data["T_is_scalar"] = T_is_scalar;

        // Return: (logZ, posteriors) - both differentiable
        return {partition, Pjoint};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor merge_scores = saved[0];
        torch::Tensor leaf_scores = saved[1];
        torch::Tensor Z = saved[2];
        torch::Tensor partition = saved[3];
        torch::Tensor grad_leaf_fwd = saved[4];
        torch::Tensor grad_T_fwd = saved[5];  // dZ/dT per batch (from forward)
        torch::Tensor temperature = saved[6];
        bool T_is_scalar = ctx->saved_data["T_is_scalar"].toBool();
        float temp_val = T_is_scalar ? temperature.item<float>() : 0.0f;

        const int B = static_cast<int>(merge_scores.size(0));
        const int n = static_cast<int>(merge_scores.size(1));

        auto options = merge_scores.options();

        // grad_outputs[0] is dL/dlogZ [B]
        // grad_outputs[1] is dL/dPjoint [B, n, n, n]
        torch::Tensor grad_logZ = grad_outputs[0];
        torch::Tensor grad_Pjoint = grad_outputs[1];
        bool has_grad_Pjoint = grad_Pjoint.defined() && grad_Pjoint.numel() > 0;
        if (has_grad_Pjoint && !T_is_scalar) {
            has_grad_Pjoint = grad_Pjoint.abs().sum().item<float>() != 0.0f;
        }

        // Initialize accumulated gradients
        torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_leaf = torch::zeros({B, n}, options);
        torch::Tensor total_grad_T = T_is_scalar
            ? torch::zeros({B}, options)
            : torch::zeros({B, n, n}, options);

        // ============ Gradient from logZ (partition function) ============
        // dL/d(merge) via logZ: dL/dZ * dZ/d(merge) = grad_logZ * Pjoint
        // dL/d(leaf) via logZ: dL/dZ * dZ/d(leaf) = grad_logZ * grad_leaf
        // dL/dT via logZ: sum(dL/dZ * dZ/dT) = sum(grad_logZ * grad_T_fwd)
        if (grad_logZ.defined() && grad_logZ.numel() > 0) {
            // Recompute posteriors for this path
            torch::Tensor beta = torch::zeros({B, n, n}, options);
            torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
            torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
            torch::Tensor gm = torch::zeros({B, n, n, n}, options);
            torch::Tensor gl = torch::zeros({B, n}, options);
            torch::Tensor gT = T_is_scalar
                ? torch::zeros({B}, options)
                : torch::zeros({B, n, n}, options);

            cky_backward_cpu(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                Pcond.data_ptr<float>(),
                Pjoint.data_ptr<float>(),
                gm.data_ptr<float>(),
                gl.data_ptr<float>(),
                gT.data_ptr<float>(),
                temperature.data_ptr<float>(),
                B, n, T_is_scalar
            );

            // dZ/d(merge) = Pjoint
            grad_merge += grad_logZ.view({B, 1, 1, 1}) * Pjoint;

            // dZ/d(leaf) = gl (diagonal posteriors)
            grad_leaf += grad_logZ.view({B, 1}) * grad_leaf_fwd;

            // dL/dT via logZ path
            if (T_is_scalar) {
                total_grad_T += grad_logZ.view({B}) * grad_T_fwd;
            } else {
                total_grad_T += grad_logZ.view({B, 1, 1}) * grad_T_fwd;
            }
        }

        // ============ Gradient from Pjoint (posteriors) ============
        if (has_grad_Pjoint) {
            TORCH_CHECK(
                T_is_scalar,
                "soft_cky backward through marginals is only supported for scalar temperature; "
                "use the score output or parent CKYTape local mode for tensor temperature"
            );
            // Validate
            TORCH_CHECK(grad_Pjoint.sizes() == merge_scores.sizes(),
                        "grad_Pjoint shape mismatch");

            if (grad_Pjoint.dtype() != torch::kFloat32) {
                grad_Pjoint = grad_Pjoint.to(torch::kFloat32);
            }
            grad_Pjoint = grad_Pjoint.contiguous();

            // HVP: d^2Z/d(merge)^2 * grad_Pjoint
            torch::Tensor d_Z = torch::zeros({B, n, n}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta = torch::zeros({B, n, n}, options);
            torch::Tensor hvp_merge = torch::zeros({B, n, n, n}, options);
            torch::Tensor hvp_leaf = torch::zeros({B, n}, options);

            cky_hvp_cpu(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_Pjoint.data_ptr<float>(),
                torch::zeros({B, n}, options).data_ptr<float>(),  // V_leaf = 0
                d_Z.data_ptr<float>(),
                d_partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_merge.data_ptr<float>(),
                hvp_leaf.data_ptr<float>(),
                B, n, temp_val
            );

            grad_merge += hvp_merge;
            grad_leaf += hvp_leaf;

            // Param grad: dP/dT (cross-derivative for temperature)
            torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
            torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

            cky_param_grad_cpu(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                dP_dT_merge.data_ptr<float>(),
                dP_dT_leaf.data_ptr<float>(),
                B, n, temp_val
            );

            // dL/dT via posteriors path
            total_grad_T[0] += (grad_Pjoint * dP_dT_merge).sum();
        }

        // Return: grad_merge, grad_leaf, grad_temperature
        torch::Tensor grad_temperature = T_is_scalar
            ? total_grad_T.sum().reshape(temperature.sizes())
            : total_grad_T;
        return {grad_merge, grad_leaf, grad_temperature};
    }
};

// =============================================================================
// Python Interface Functions (CPU)
// =============================================================================

// Main autograd function: (merge_scores, leaf_scores, temperature) -> (logZ, posteriors)
// Temperature is a tensor for gradient flow
std::vector<torch::Tensor> soft_cky_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor temperature
) {
    return SoftCKYCPUFunction::apply(merge_scores, leaf_scores, temperature);
}

// Float version: temperature is a scalar (no gradient for temperature)
std::vector<torch::Tensor> soft_cky_float_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    auto options = merge_scores.options();
    auto temp_t = torch::tensor({temperature}, options).requires_grad_(true);
    return SoftCKYCPUFunction::apply(merge_scores, leaf_scores, temp_t);
}

// With explicit gradients (for debugging/inspection)
// Returns: (logZ, Pjoint, grad_leaf, grad_T)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_cky_with_grads_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    D2P_CHECK_INPUT_CPU(merge_scores);
    D2P_CHECK_INPUT_CPU(leaf_scores);
    validate_cky_chart_shapes_cpu(merge_scores, leaf_scores);

    const int B = static_cast<int>(merge_scores.size(0));
    const int n = static_cast<int>(merge_scores.size(1));

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
    torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_leaf = torch::zeros({B, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    cky_forward_cpu(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    cky_backward_cpu(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        Pcond.data_ptr<float>(),
        Pjoint.data_ptr<float>(),
        grad_merge.data_ptr<float>(),
        grad_leaf.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    return std::make_tuple(partition, Pjoint, grad_leaf, grad_T);
}

// HVP: Hessian-vector product
torch::Tensor soft_cky_hvp_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_merge,
    torch::Tensor v_leaf,
    double temperature
) {
    D2P_CHECK_INPUT_CPU(merge_scores);
    D2P_CHECK_INPUT_CPU(leaf_scores);
    D2P_CHECK_INPUT_CPU(v_merge);
    D2P_CHECK_INPUT_CPU(v_leaf);
    validate_cky_chart_shapes_cpu(merge_scores, leaf_scores);
    TORCH_CHECK(v_merge.sizes() == merge_scores.sizes(), "v_merge must match merge_scores shape");
    TORCH_CHECK(v_leaf.sizes() == leaf_scores.sizes(), "v_leaf must match leaf_scores shape");

    const int B = static_cast<int>(merge_scores.size(0));
    const int n = static_cast<int>(merge_scores.size(1));

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    // Forward pass first
    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    cky_forward_cpu(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    // HVP
    torch::Tensor d_Z = torch::zeros({B, n, n}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta = torch::zeros({B, n, n}, options);
    torch::Tensor hvp_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor hvp_leaf = torch::zeros({B, n}, options);

    cky_hvp_cpu(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        v_merge.data_ptr<float>(),
        v_leaf.data_ptr<float>(),
        d_Z.data_ptr<float>(),
        d_partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        hvp_merge.data_ptr<float>(),
        hvp_leaf.data_ptr<float>(),
        B, n, T
    );

    return hvp_merge;
}

// Param Jacobian: dP/dT (derivative of posteriors w.r.t. temperature)
torch::Tensor soft_cky_param_jacobian_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    D2P_CHECK_INPUT_CPU(merge_scores);
    D2P_CHECK_INPUT_CPU(leaf_scores);
    validate_cky_chart_shapes_cpu(merge_scores, leaf_scores);

    const int B = static_cast<int>(merge_scores.size(0));
    const int n = static_cast<int>(merge_scores.size(1));

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    // Forward pass
    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    cky_forward_cpu(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    // Param grad
    torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

    cky_param_grad_cpu(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        dP_dT_merge.data_ptr<float>(),
        dP_dT_leaf.data_ptr<float>(),
        B, n, T
    );

    return dP_dT_merge;
}

// Backward full: Given grad_posteriors, compute full backward pass
// Returns: (grad_merge, grad_leaf, grad_temperature)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_cky_backward_full_cpu(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor grad_posteriors,
    double temperature
) {
    D2P_CHECK_INPUT_CPU(merge_scores);
    D2P_CHECK_INPUT_CPU(leaf_scores);
    D2P_CHECK_INPUT_CPU(grad_posteriors);
    validate_cky_chart_shapes_cpu(merge_scores, leaf_scores);
    TORCH_CHECK(grad_posteriors.sizes() == merge_scores.sizes(),
                "grad_posteriors must match merge_scores shape");

    const int B = static_cast<int>(merge_scores.size(0));
    const int n = static_cast<int>(merge_scores.size(1));

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    // Forward pass
    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    cky_forward_cpu(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    // HVP for score gradients
    torch::Tensor d_Z = torch::zeros({B, n, n}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta = torch::zeros({B, n, n}, options);
    torch::Tensor hvp_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor hvp_leaf = torch::zeros({B, n}, options);

    cky_hvp_cpu(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        grad_posteriors.data_ptr<float>(),
        torch::zeros({B, n}, options).data_ptr<float>(),
        d_Z.data_ptr<float>(),
        d_partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        hvp_merge.data_ptr<float>(),
        hvp_leaf.data_ptr<float>(),
        B, n, T
    );

    // Param grad for temperature gradient
    torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

    cky_param_grad_cpu(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        dP_dT_merge.data_ptr<float>(),
        dP_dT_leaf.data_ptr<float>(),
        B, n, T
    );

    // Temperature gradient: sum(grad_posteriors * dP/dT)
    torch::Tensor grad_T = (grad_posteriors * dP_dT_merge).sum().reshape({1});

    return std::make_tuple(hvp_merge, hvp_leaf, grad_T);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_cky", soft_cky_cpu);
    m.impl("soft_cky_float", soft_cky_float_cpu);
    m.impl("soft_cky_with_grads", soft_cky_with_grads_cpu);
    m.impl("soft_cky_hvp", soft_cky_hvp_cpu);
    m.impl("soft_cky_param_jacobian", soft_cky_param_jacobian_cpu);
    m.impl("soft_cky_backward_full", soft_cky_backward_full_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_cky", soft_cky_cpu);
    m.impl("soft_cky_float", soft_cky_float_cpu);
}

#endif // USE_TORCH_LIBRARY
