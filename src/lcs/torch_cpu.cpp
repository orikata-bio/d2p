// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft LCS CPU PyTorch Bindings
 *
 * CPU implementations mirroring CUDA interface for automatic dispatch.
 */

#include <torch/extension.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels_cpu.h"

// ============================================================================
// Helper Macros
// ============================================================================

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

void validate_lcs_scores_cpu(const torch::Tensor& scores) {
    CHECK_INPUT_CPU(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");

    constexpr int64_t max_int = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(
        scores.size(0) <= max_int && scores.size(1) <= max_int && scores.size(2) <= max_int,
        "scores dimensions must fit int32 CPU kernel parameters"
    );
}

int64_t checked_lcs_alpha_size_cpu(int64_t B, int64_t max_L1, int64_t max_L2) {
    constexpr int64_t max_int = static_cast<int64_t>(std::numeric_limits<int>::max());
    const int64_t rows = max_L1 + 1;
    const int64_t cols = max_L2 + 1;
    const int64_t alpha_size = rows * cols;

    TORCH_CHECK(
        alpha_size <= max_int,
        "LCS workspace too large: (L1 + 1) * (L2 + 1) must be <= ",
        max_int,
        ", got ",
        alpha_size,
        " for L1=",
        max_L1,
        ", L2=",
        max_L2
    );
    TORCH_CHECK(
        B == 0 || alpha_size <= std::numeric_limits<int64_t>::max() / B,
        "LCS workspace too large: B * (L1 + 1) * (L2 + 1) overflows int64"
    );
    return alpha_size;
}

void validate_lcs_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
    CHECK_INPUT_CPU(lengths);
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2);
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");

    auto lengths_acc = lengths.accessor<int32_t, 2>();
    for (int b = 0; b < B; ++b) {
        int l1 = lengths_acc[b][0];
        int l2 = lengths_acc[b][1];
        TORCH_CHECK(
            l1 >= 0 && l1 <= max_L1,
            "lengths[", b, ",0] must be between 0 and ", max_L1, ", got ", l1
        );
        TORCH_CHECK(
            l2 >= 0 && l2 <= max_L2,
            "lengths[", b, ",1] must be between 0 and ", max_L2, ", got ", l2
        );
    }
}

void validate_lcs_backward_input_cpu(const torch::Tensor& scores, const torch::Tensor& grad_posteriors) {
    TORCH_CHECK(!grad_posteriors.is_cuda(), "grad_posteriors must be a CPU tensor");
    TORCH_CHECK(
        grad_posteriors.sizes() == scores.sizes(),
        "grad_posteriors must have same shape as scores"
    );
}

torch::Tensor make_default_lengths_lcs_cpu(int B, int L1, int L2) {
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto lengths = torch::empty({B, 2}, options);
    auto acc = lengths.accessor<int32_t, 2>();
    for (int b = 0; b < B; b++) {
        acc[b][0] = L1;
        acc[b][1] = L2;
    }
    return lengths;
}

torch::Tensor resolve_lcs_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_lcs_cpu(B, max_L1, max_L2);
    validate_lcs_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// ============================================================================
// Autograd Function
// ============================================================================

class SoftLCSCPUFunction : public torch::autograd::Function<SoftLCSCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP/param-grad path.
        ctx->set_materialize_grads(false);

        validate_lcs_scores_cpu(scores);
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        int B = scores.size(0);
        int max_L1 = scores.size(1);
        int max_L2 = scores.size(2);
        int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

        validate_lcs_lengths_cpu(lengths, B, max_L1, max_L2);

        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor lcs_score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        d2p::lcs::cpu::lcs_forward_cpu(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            lcs_score.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2, temp_val
        );

        d2p::lcs::cpu::lcs_backward_cpu(
            alpha.data_ptr<float>(),
            scores.data_ptr<float>(),
            lcs_score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), lcs_score.clone(), lengths.clone(), grad_T.clone()});
        ctx->saved_data["temperature"] = temp_val;

        return {lcs_score, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor scores = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor lcs_score = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_T_fwd = saved[4];

        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        int B = scores.size(0);
        int max_L1 = scores.size(1);
        int max_L2 = scores.size(2);
        int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

        auto options = scores.options();

        torch::Tensor grad_lcs_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from lcs_score path
        if (grad_lcs_score.defined() && grad_lcs_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            d2p::lcs::cpu::lcs_backward_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2, temp_val
            );

            grad_scores += grad_lcs_score.view({B, 1, 1}) * posteriors;
            total_grad_T += (grad_lcs_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from posteriors path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            validate_lcs_backward_input_cpu(scores, grad_posteriors);
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_lcs_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::lcs::cpu::lcs_hvp_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_lcs_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grad for temperature
            torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

            d2p::lcs::cpu::lcs_param_grad_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                U_T.data_ptr<float>(),
                beta_T.data_ptr<float>(),
                W_T.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2, temp_val
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        return {grad_scores, total_grad_T, torch::Tensor()};
    }
};

// ============================================================================
// Operator Implementations
// ============================================================================

std::vector<torch::Tensor> soft_lcs_cpu(
    torch::Tensor scores,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftLCSCPUFunction::apply(scores, temperature, lengths);
}

std::vector<torch::Tensor> soft_lcs_cpu_float(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lcs_scores_cpu(scores);

    int B = scores.size(0);
    int L1 = scores.size(1);
    int L2 = scores.size(2);

    auto options = scores.options();
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    torch::Tensor lengths = resolve_lcs_lengths_cpu(lengths_opt, B, L1, L2);

    return SoftLCSCPUFunction::apply(scores, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_lcs_cpu_with_grads(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lcs_scores_cpu(scores);

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lcs_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor lcs_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::lcs::cpu::lcs_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    d2p::lcs::cpu::lcs_backward_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    return std::make_tuple(lcs_score, posteriors, grad_T);
}

torch::Tensor soft_lcs_hvp_cpu_impl(
    torch::Tensor scores,
    torch::Tensor tangent,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lcs_scores_cpu(scores);
    CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(tangent.dtype() == torch::kFloat32, "tangent must be float32");
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lcs_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor lcs_score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_lcs_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::lcs::cpu::lcs_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    d2p::lcs::cpu::lcs_hvp_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    return H_scores;
}

torch::Tensor soft_lcs_param_jacobian_cpu_impl(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lcs_scores_cpu(scores);

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lcs_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor lcs_score = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    d2p::lcs::cpu::lcs_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    d2p::lcs::cpu::lcs_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    return dP_dT;
}

std::tuple<torch::Tensor, torch::Tensor>
soft_lcs_backward_full_cpu_impl(
    torch::Tensor scores,
    torch::Tensor grad_output,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lcs_scores_cpu(scores);
    CHECK_INPUT_CPU(grad_output);
    TORCH_CHECK(grad_output.dim() == 3, "grad_posteriors must be 3D");
    validate_lcs_backward_input_cpu(scores, grad_output);

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = checked_lcs_alpha_size_cpu(B, max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lcs_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    grad_output = grad_output.contiguous().to(torch::kFloat32);

    float temp_val = static_cast<float>(temperature);

    // Forward pass
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor lcs_score = torch::zeros({B}, options);

    d2p::lcs::cpu::lcs_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    // HVP for grad_scores
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_lcs_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::lcs::cpu::lcs_hvp_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );

    // Param grad for temperature
    torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    d2p::lcs::cpu::lcs_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        U_T.data_ptr<float>(),
        beta_T.data_ptr<float>(),
        W_T.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, temp_val
    );
    torch::Tensor total_grad_T = (grad_output * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_T);
}

// ============================================================================
// Registration
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_lcs", soft_lcs_cpu);
    m.impl("soft_lcs_float", soft_lcs_cpu_float);
    m.impl("soft_lcs_with_grads", soft_lcs_cpu_with_grads);
    m.impl("soft_lcs_hvp", soft_lcs_hvp_cpu_impl);
    m.impl("soft_lcs_param_jacobian", soft_lcs_param_jacobian_cpu_impl);
    m.impl("soft_lcs_backward_full", soft_lcs_backward_full_cpu_impl);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_lcs", soft_lcs_cpu);
    m.impl("soft_lcs_float", soft_lcs_cpu_float);
}

#endif // USE_TORCH_LIBRARY
