// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft Eisner CPU PyTorch Bindings
 *
 * CPU implementation of projective dependency parsing with full autograd support.
 */

#include <torch/extension.h>
#include <vector>

#include "common/torch_utils.h"
#include "kernels_cpu.h"

// =============================================================================
// Helper Macros
// =============================================================================

using namespace d2p::common;

namespace {

void validate_eisner_arc_scores_cpu(const torch::Tensor& arc_scores) {
    D2P_CHECK_CPU(arc_scores);
    TORCH_CHECK(arc_scores.dim() == 3, "arc_scores must be 3D [B, n, n]");
    TORCH_CHECK(arc_scores.size(1) == arc_scores.size(2), "arc_scores must be [B, n, n]");
    TORCH_CHECK(arc_scores.size(1) > 0, "soft_eisner requires n > 0");
    TORCH_CHECK(
        arc_scores.size(1) <= d2p::eisner::cpu::MAX_EISNER_N_FOR_INT_CELL_INDEX,
        "soft_eisner requires n <= ",
        d2p::eisner::cpu::MAX_EISNER_N_FOR_INT_CELL_INDEX,
        " to avoid 32-bit cell-index overflow, got ",
        arc_scores.size(1)
    );
    D2P_CHECK_CONTIGUOUS(arc_scores);
    D2P_CHECK_FLOAT(arc_scores);
}

void validate_eisner_lengths_cpu(
    const torch::Tensor& lengths,
    int B,
    int n,
    torch::Device device
) {
    D2P_CHECK_CONTIGUOUS(lengths);
    D2P_CHECK_LENGTHS_1D(lengths, B, device);

    auto lengths_acc = lengths.accessor<int32_t, 1>();
    for (int b = 0; b < B; ++b) {
        int seq_len = lengths_acc[b];
        TORCH_CHECK(
            seq_len >= 1 && seq_len <= n,
            "lengths[", b, "] must be between 1 and ", n, ", got ", seq_len
        );
    }
}

torch::Tensor resolve_eisner_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int n,
    torch::Device device
) {
    if (!lengths_opt.has_value() || !lengths_opt->defined()) {
        return {};
    }

    torch::Tensor lengths = lengths_opt.value();
    validate_eisner_lengths_cpu(lengths, B, n, device);
    return lengths;
}

void validate_eisner_tangent_cpu(
    const torch::Tensor& arc_scores,
    const torch::Tensor& tangent
) {
    D2P_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(
        tangent.sizes() == arc_scores.sizes(),
        "tangent must have same shape as arc_scores"
    );
}

}  // namespace

// =============================================================================
// Autograd Function
// =============================================================================

class SoftEisnerCPUFunction : public torch::autograd::Function<SoftEisnerCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor arc_scores,
        torch::Tensor temperature,
        c10::optional<torch::Tensor> lengths_opt
    ) {
        // r70: leave the unused marginals output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP path.
        ctx->set_materialize_grads(false);

        validate_eisner_arc_scores_cpu(arc_scores);
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        int B = arc_scores.size(0);
        int n = arc_scores.size(1);

        float temp_val = temperature.item<float>();
        auto options = arc_scores.options();

        torch::Tensor lengths = resolve_eisner_lengths_cpu(lengths_opt, B, n, arc_scores.device());
        const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

        // Allocate tables
        torch::Tensor C_R = torch::zeros({B, n, n}, options);
        torch::Tensor C_L = torch::zeros({B, n, n}, options);
        torch::Tensor I_R = torch::zeros({B, n, n}, options);
        torch::Tensor I_L = torch::zeros({B, n, n}, options);
        torch::Tensor partition = torch::zeros({B}, options);

        // Forward pass
        d2p::eisner::cpu::forward(
            arc_scores.data_ptr<float>(),
            C_R.data_ptr<float>(),
            C_L.data_ptr<float>(),
            I_R.data_ptr<float>(),
            I_L.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths_ptr,
            B, n, temp_val
        );

        // Backward pass (compute marginals)
        torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
        torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
        torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
        torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
        torch::Tensor marginals = torch::zeros({B, n, n}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        d2p::eisner::cpu::backward(
            arc_scores.data_ptr<float>(),
            C_R.data_ptr<float>(),
            C_L.data_ptr<float>(),
            I_R.data_ptr<float>(),
            I_L.data_ptr<float>(),
            beta_C_R.data_ptr<float>(),
            beta_C_L.data_ptr<float>(),
            beta_I_R.data_ptr<float>(),
            beta_I_L.data_ptr<float>(),
            marginals.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths_ptr,
            B, n, temp_val
        );

        // Save for backward
        ctx->saved_data["temperature"] = temp_val;
        if (lengths.defined()) {
            ctx->saved_data["has_lengths"] = true;
            ctx->save_for_backward({arc_scores.clone(), C_R, C_L, I_R, I_L, grad_T, lengths});
        } else {
            ctx->saved_data["has_lengths"] = false;
            ctx->save_for_backward({arc_scores.clone(), C_R, C_L, I_R, I_L, grad_T});
        }

        return {partition, marginals};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor arc_scores = saved[0];
        torch::Tensor C_R = saved[1];
        torch::Tensor C_L = saved[2];
        torch::Tensor I_R = saved[3];
        torch::Tensor I_L = saved[4];
        torch::Tensor grad_T_fwd = saved[5];

        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());
        bool has_lengths = ctx->saved_data["has_lengths"].toBool();

        const int* lengths_ptr = nullptr;
        if (has_lengths && saved.size() > 6) {
            torch::Tensor lengths = saved[6];
            lengths_ptr = lengths.data_ptr<int>();
        }

        int B = arc_scores.size(0);
        int n = arc_scores.size(1);
        auto options = arc_scores.options();

        torch::Tensor grad_partition = grad_outputs[0];
        torch::Tensor grad_marginals = grad_outputs[1];

        // Initialize gradients
        torch::Tensor grad_arc = torch::zeros({B, n, n}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from partition function
        if (grad_partition.defined() && grad_partition.numel() > 0) {
            // Recompute marginals
            torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor marginals = torch::zeros({B, n, n}, options);
            torch::Tensor grad_T = torch::zeros({B}, options);

            d2p::eisner::cpu::backward(
                arc_scores.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                beta_C_R.data_ptr<float>(),
                beta_C_L.data_ptr<float>(),
                beta_I_R.data_ptr<float>(),
                beta_I_L.data_ptr<float>(),
                marginals.data_ptr<float>(),
                grad_T.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            grad_arc += grad_partition.view({B, 1, 1}) * marginals;
            total_grad_T += (grad_partition * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from marginals via HVP
        if (grad_marginals.defined() && grad_marginals.numel() > 0) {
            grad_marginals = grad_marginals.contiguous().to(torch::kFloat32);

            torch::Tensor d_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_C_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_C_L = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_I_R = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta_I_L = torch::zeros({B, n, n}, options);
            torch::Tensor HVP = torch::zeros({B, n, n}, options);

            d2p::eisner::cpu::hvp(
                arc_scores.data_ptr<float>(),
                grad_marginals.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                d_C_R.data_ptr<float>(),
                d_C_L.data_ptr<float>(),
                d_I_R.data_ptr<float>(),
                d_I_L.data_ptr<float>(),
                beta_C_R.data_ptr<float>(),
                beta_C_L.data_ptr<float>(),
                beta_I_R.data_ptr<float>(),
                beta_I_L.data_ptr<float>(),
                d_beta_C_R.data_ptr<float>(),
                d_beta_C_L.data_ptr<float>(),
                d_beta_I_R.data_ptr<float>(),
                d_beta_I_L.data_ptr<float>(),
                HVP.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            grad_arc += HVP;

            // r71: marginals -> temperature. The marginals depend on the
            // temperature, so a loss through the marginals output contributes
            // to the temperature gradient via the dP/dT parameter Jacobian.
            // Mirrors the partition-path term above and the CPU/CUDA edit-distance
            // posteriors -> temperature terms, so CPU and CUDA return the same
            // complete temperature gradient.
            torch::Tensor dP_dT = torch::zeros({B, n, n}, options);

            d2p::eisner::cpu::param_grad(
                arc_scores.data_ptr<float>(),
                C_R.data_ptr<float>(),
                C_L.data_ptr<float>(),
                I_R.data_ptr<float>(),
                I_L.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths_ptr,
                B, n, temp_val
            );

            total_grad_T += (grad_marginals * dP_dT).sum().reshape({1});
        }

        return {grad_arc, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Wrapper Functions
// =============================================================================

std::vector<torch::Tensor> soft_eisner_cpu(
    torch::Tensor arc_scores,
    torch::Tensor temperature,
    c10::optional<torch::Tensor> lengths
) {
    return SoftEisnerCPUFunction::apply(arc_scores, temperature, lengths);
}

torch::Tensor soft_eisner_float_cpu(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    auto options = arc_scores.options();
    auto temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    auto results = SoftEisnerCPUFunction::apply(arc_scores, temp_t, lengths);
    return results[0];
}

std::tuple<torch::Tensor, torch::Tensor> soft_eisner_with_grads_cpu(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cpu(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cpu(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::eisner::cpu::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor marginals = torch::zeros({B, n, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::eisner::cpu::backward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        marginals.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return std::make_tuple(partition, marginals);
}

torch::Tensor soft_eisner_hvp_cpu(
    torch::Tensor arc_scores,
    torch::Tensor V,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cpu(arc_scores);
    validate_eisner_tangent_cpu(arc_scores, V);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cpu(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::eisner::cpu::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor d_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor HVP = torch::zeros({B, n, n}, options);

    d2p::eisner::cpu::hvp(
        arc_scores.data_ptr<float>(),
        V.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        d_C_R.data_ptr<float>(),
        d_C_L.data_ptr<float>(),
        d_I_R.data_ptr<float>(),
        d_I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        d_beta_C_R.data_ptr<float>(),
        d_beta_C_L.data_ptr<float>(),
        d_beta_I_R.data_ptr<float>(),
        d_beta_I_L.data_ptr<float>(),
        HVP.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return HVP;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> soft_eisner_backward_full_cpu(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cpu(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cpu(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::eisner::cpu::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    torch::Tensor beta_C_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_C_L = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_R = torch::zeros({B, n, n}, options);
    torch::Tensor beta_I_L = torch::zeros({B, n, n}, options);
    torch::Tensor marginals = torch::zeros({B, n, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::eisner::cpu::backward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        beta_C_R.data_ptr<float>(),
        beta_C_L.data_ptr<float>(),
        beta_I_R.data_ptr<float>(),
        beta_I_L.data_ptr<float>(),
        marginals.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return std::make_tuple(partition, marginals, grad_T);
}

torch::Tensor soft_eisner_param_jacobian_cpu(
    torch::Tensor arc_scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_eisner_arc_scores_cpu(arc_scores);

    int B = arc_scores.size(0);
    int n = arc_scores.size(1);
    float T = static_cast<float>(temperature);
    auto options = arc_scores.options();

    torch::Tensor lengths = resolve_eisner_lengths_cpu(lengths_opt, B, n, arc_scores.device());
    const int* lengths_ptr = lengths.defined() ? lengths.data_ptr<int>() : nullptr;

    // Forward pass
    torch::Tensor C_R = torch::zeros({B, n, n}, options);
    torch::Tensor C_L = torch::zeros({B, n, n}, options);
    torch::Tensor I_R = torch::zeros({B, n, n}, options);
    torch::Tensor I_L = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::eisner::cpu::forward(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    // Parameter gradient
    torch::Tensor dP_dT = torch::zeros({B, n, n}, options);

    d2p::eisner::cpu::param_grad(
        arc_scores.data_ptr<float>(),
        C_R.data_ptr<float>(),
        C_L.data_ptr<float>(),
        I_R.data_ptr<float>(),
        I_L.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths_ptr,
        B, n, T
    );

    return dP_dT;
}

// =============================================================================
// Library Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_eisner", soft_eisner_cpu);
    m.impl("soft_eisner_float", soft_eisner_float_cpu);
    m.impl("soft_eisner_with_grads", soft_eisner_with_grads_cpu);
    m.impl("soft_eisner_hvp", soft_eisner_hvp_cpu);
    m.impl("soft_eisner_backward_full", soft_eisner_backward_full_cpu);
    m.impl("soft_eisner_param_jacobian", soft_eisner_param_jacobian_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_eisner", soft_eisner_cpu);
    m.impl("soft_eisner_float", soft_eisner_float_cpu);
}

#endif // USE_TORCH_LIBRARY
