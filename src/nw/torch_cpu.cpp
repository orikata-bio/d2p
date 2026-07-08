// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Needleman-Wunsch CPU Extension with PyTorch Autograd
 *
 * Global alignment with linear gap penalty. CPU implementations registered
 * via TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * Recurrence:
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // align
 *       alpha[i-1,j] + gap,              // gap in seq2
 *       alpha[i,j-1] + gap               // gap in seq1
 *   )
 *
 * Key differences from Smith-Waterman:
 *   - 3 transitions (no "start new alignment" option)
 *   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
 *   - Score = alpha[L1,L2] at terminal (global alignment)
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"

// CPU kernel declarations
#include "nw/kernels_cpu.h"

using namespace d2p::common;

namespace {

struct NWCPUShape {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

int checked_nw_int_dim(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<int>::max(),
        name, " must fit in int32, got ", value
    );
    return static_cast<int>(value);
}

int64_t checked_nw_alpha_size(int max_L1, int max_L2) {
    const int64_t rows = static_cast<int64_t>(max_L1) + 1;
    const int64_t cols = static_cast<int64_t>(max_L2) + 1;
    const int64_t max_supported = std::numeric_limits<int>::max();
    TORCH_CHECK(
        rows <= max_supported / cols,
        "NW alpha workspace size is too large: (", max_L1, " + 1) * (", max_L2,
        " + 1) exceeds ", max_supported
    );
    return rows * cols;
}

NWCPUShape checked_nw_scores_cpu(const torch::Tensor& scores) {
    D2P_CHECK_INPUT_CPU(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    int B = checked_nw_int_dim(scores.size(0), "scores.size(0)");
    int max_L1 = checked_nw_int_dim(scores.size(1), "scores.size(1)");
    int max_L2 = checked_nw_int_dim(scores.size(2), "scores.size(2)");
    int64_t alpha_size = checked_nw_alpha_size(max_L1, max_L2);
    return {B, max_L1, max_L2, alpha_size};
}

void validate_nw_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
    D2P_CHECK_CPU(lengths);
    D2P_CHECK_CONTIGUOUS(lengths);
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

torch::Tensor resolve_nw_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_nw_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// =============================================================================
// NW CPU Autograd Function
// =============================================================================

class SoftNWCPUFunction : public torch::autograd::Function<SoftNWCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP/param-grad path.
        ctx->set_materialize_grads(false);

        NWCPUShape shape = checked_nw_scores_cpu(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        validate_nw_lengths_cpu(lengths, B, max_L1, max_L2);

        float gap_val = gap.item<float>();
        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        nw_forward_cpu(
            scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
        );

        nw_backward_cpu(
            alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
            beta.data_ptr<float>(), posteriors.data_ptr<float>(),
            grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), score.clone(), lengths.clone(),
                                grad_gap.clone(), grad_T.clone()});
        ctx->saved_data["gap"] = gap_val;
        ctx->saved_data["temperature"] = temp_val;

        return {score, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor scores = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor score = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_gap_fwd = saved[4];
        torch::Tensor grad_T_fwd = saved[5];

        float gap_val = static_cast<float>(ctx->saved_data["gap"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        NWCPUShape shape = checked_nw_scores_cpu(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_gap = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from score path
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_gap = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            nw_backward_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                beta.data_ptr<float>(), posteriors.data_ptr<float>(),
                tmp_gap.data_ptr<float>(), tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
            );

            grad_scores += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_gap += (grad_score * grad_gap_fwd).sum().reshape({1});
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from alignment path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            nw_hvp_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(), d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(), beta.data_ptr<float>(),
                d_beta.data_ptr<float>(), hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grads
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            nw_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_gap_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val, 0
            );
            total_grad_gap += (grad_posteriors * dP_dtheta).sum().reshape({1});

            nw_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_val, temp_val, 1
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        return {grad_scores, total_grad_gap, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions (CPU)
// =============================================================================

std::vector<torch::Tensor> soft_nw_cpu(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftNWCPUFunction::apply(scores, gap, temperature, lengths);
}

std::vector<torch::Tensor> soft_nw_cpu_float(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    NWCPUShape shape = checked_nw_scores_cpu(scores);
    int B = shape.B;
    int L1 = shape.max_L1;
    int L2 = shape.max_L2;

    torch::Tensor gap_t = torch::tensor({static_cast<float>(gap)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_nw_lengths_cpu(lengths_opt, B, L1, L2, scores.device());

    return SoftNWCPUFunction::apply(scores, gap_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_cpu_with_grads(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    NWCPUShape shape = checked_nw_scores_cpu(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    nw_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    nw_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    return std::make_tuple(score, posteriors, grad_gap, grad_T);
}

torch::Tensor soft_nw_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    NWCPUShape shape = checked_nw_scores_cpu(scores);
    D2P_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    nw_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    nw_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    return H_scores;
}

torch::Tensor soft_nw_param_jacobian_cpu(
    torch::Tensor scores, int64_t param_type, double gap, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    NWCPUShape shape = checked_nw_scores_cpu(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 1, "param_type must be 0 or 1");
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    nw_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    nw_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    torch::Tensor dS_dtheta = (param_type == 0) ? grad_gap : grad_T;

    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    nw_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), param_type
    );

    return dP_dtheta;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_backward_full_cpu(
    torch::Tensor scores, torch::Tensor grad_alignment,
    double gap, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    NWCPUShape shape = checked_nw_scores_cpu(scores);
    TORCH_CHECK(!grad_alignment.is_cuda(), "grad_alignment must be a CPU tensor");
    TORCH_CHECK(grad_alignment.sizes() == scores.sizes(), "grad_alignment must have same shape as scores");
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    grad_alignment = grad_alignment.contiguous().to(torch::kFloat32);

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    nw_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    nw_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta_fwd.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_fwd.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    // HVP
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    nw_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_alignment.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature)
    );

    // Param grads
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    nw_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_gap_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap = (grad_alignment * dP_dtheta).sum().reshape({1});

    nw_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_gap, total_grad_T);
}

// =============================================================================
// Namespaced API Wrappers (nw_*)
// =============================================================================

// nw::forward - returns (value, marginals)
std::vector<torch::Tensor> nw_forward_cpu_wrapper(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_cpu_float(scores, gap, temp, lengths);
}

// nw::forward_t - tensor params version
std::vector<torch::Tensor> nw_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_nw_cpu(scores, gap, temp, lengths);
}

// nw::value_grad_params - returns (grad_gap, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor> nw_value_grad_params_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_nw_cpu_with_grads(scores, gap, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result));
}

// nw::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nw_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_backward_full_cpu(scores, grad_marginals, gap, temp, lengths);
}

// nw::marginals_hvp - Hessian-vector product
torch::Tensor nw_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_hvp_cpu(scores, v, gap, temp, lengths);
}

// nw::marginals_grad_gap - d(marginals)/d(gap)
torch::Tensor nw_marginals_grad_gap_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_param_jacobian_cpu(scores, 0, gap, temp, lengths);
}

// nw::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor nw_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_param_jacobian_cpu(scores, 1, gap, temp, lengths);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_nw", soft_nw_cpu);
    m.impl("soft_nw_float", soft_nw_cpu_float);
    m.impl("soft_nw_with_grads", soft_nw_cpu_with_grads);
    m.impl("soft_nw_hvp", soft_nw_hvp_cpu);
    m.impl("soft_nw_param_jacobian", soft_nw_param_jacobian_cpu);
    m.impl("soft_nw_backward_full", soft_nw_backward_full_cpu);

    // Namespaced API
    m.impl("nw_forward", nw_forward_cpu_wrapper);
    m.impl("nw_forward_t", nw_forward_t_cpu);
    m.impl("nw_value_grad_params", nw_value_grad_params_cpu);
    m.impl("nw_marginals_backward", nw_marginals_backward_cpu);
    m.impl("nw_marginals_hvp", nw_marginals_hvp_cpu);
    m.impl("nw_marginals_grad_gap", nw_marginals_grad_gap_cpu);
    m.impl("nw_marginals_grad_temp", nw_marginals_grad_temp_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_nw", soft_nw_cpu);
    m.impl("soft_nw_float", soft_nw_cpu_float);

    // Namespaced API - autograd versions
    m.impl("nw_forward", nw_forward_cpu_wrapper);
    m.impl("nw_forward_t", nw_forward_t_cpu);
}

#endif // USE_TORCH_LIBRARY
