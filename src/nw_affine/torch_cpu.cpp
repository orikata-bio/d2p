// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief PyTorch CPU bindings for Soft Needleman-Wunsch Affine Gap
 *
 * CPU implementation of soft NW with affine gap penalty.
 * Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
 *
 * All operations support PyTorch autograd for automatic differentiation.
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

#include "common/torch_utils.h"
#include "nw_affine/kernels_cpu.h"

using namespace d2p::common;

namespace {

constexpr int64_t kNWAffineMaxWorkspaceElements = std::numeric_limits<int>::max();

struct NWAffineCPUShape {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

int checked_nw_affine_size_to_int(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0, name, " must be non-negative, got ", value);
    TORCH_CHECK(
        value <= std::numeric_limits<int>::max(),
        name, " is too large for the nw_affine CPU backend: ", value
    );
    return static_cast<int>(value);
}

int64_t checked_nw_affine_alpha_size(int max_L1, int max_L2) {
    const int64_t rows = static_cast<int64_t>(max_L1) + 1;
    const int64_t cols = static_cast<int64_t>(max_L2) + 1;
    const int64_t max_int64 = std::numeric_limits<int64_t>::max();

    TORCH_CHECK(
        rows <= max_int64 / cols,
        "nw_affine CPU DP workspace size overflow for L1=", max_L1, ", L2=", max_L2
    );
    const int64_t state_size = rows * cols;
    TORCH_CHECK(
        state_size <= max_int64 / 3,
        "nw_affine CPU DP workspace size overflow for L1=", max_L1, ", L2=", max_L2
    );
    const int64_t alpha_size = 3 * state_size;
    TORCH_CHECK(
        alpha_size <= kNWAffineMaxWorkspaceElements,
        "nw_affine CPU DP workspace is too large: 3 * (L1 + 1) * (L2 + 1) = ",
        alpha_size, " exceeds ", kNWAffineMaxWorkspaceElements
    );
    return alpha_size;
}

NWAffineCPUShape validate_nw_affine_scores_cpu(const torch::Tensor& scores) {
    D2P_CHECK_INPUT_CPU(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    NWAffineCPUShape shape;
    shape.B = checked_nw_affine_size_to_int(scores.size(0), "scores batch size");
    shape.max_L1 = checked_nw_affine_size_to_int(scores.size(1), "scores L1 dimension");
    shape.max_L2 = checked_nw_affine_size_to_int(scores.size(2), "scores L2 dimension");
    shape.alpha_size = checked_nw_affine_alpha_size(shape.max_L1, shape.max_L2);
    return shape;
}

void validate_nw_affine_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
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

torch::Tensor resolve_nw_affine_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_nw_affine_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// =============================================================================
// Soft NW Affine CPU Autograd Function (3-state DP)
// =============================================================================

class SoftNWAffineCPUFunction : public torch::autograd::Function<SoftNWAffineCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap_open,
        torch::Tensor gap_ext,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        ctx->set_materialize_grads(false);

        const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;  // 3 states: M, I, D

        D2P_CHECK_CPU(lengths);
        validate_nw_affine_lengths_cpu(lengths, B, max_L1, max_L2);

        float gap_open_val = gap_open.item<float>();
        float gap_ext_val = gap_ext.item<float>();
        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap_open = torch::zeros({B}, options);
        torch::Tensor grad_gap_ext = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        nw_affine_forward_cpu(
            scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
        );

        nw_affine_backward_cpu(
            alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
            beta.data_ptr<float>(), posteriors.data_ptr<float>(),
            grad_gap_open.data_ptr<float>(), grad_gap_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), score.clone(), lengths.clone(),
                                grad_gap_open.clone(), grad_gap_ext.clone(), grad_T.clone()});
        ctx->saved_data["gap_open"] = gap_open_val;
        ctx->saved_data["gap_ext"] = gap_ext_val;
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
        torch::Tensor grad_gap_open_fwd = saved[4];
        torch::Tensor grad_gap_ext_fwd = saved[5];
        torch::Tensor grad_T_fwd = saved[6];

        float gap_open_val = static_cast<float>(ctx->saved_data["gap_open"].toDouble());
        float gap_ext_val = static_cast<float>(ctx->saved_data["gap_ext"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        int B = checked_nw_affine_size_to_int(scores.size(0), "scores batch size");
        int max_L1 = checked_nw_affine_size_to_int(scores.size(1), "scores L1 dimension");
        int max_L2 = checked_nw_affine_size_to_int(scores.size(2), "scores L2 dimension");
        int64_t alpha_size = checked_nw_affine_alpha_size(max_L1, max_L2);

        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_gap_open = torch::zeros({1}, options);
        torch::Tensor total_grad_gap_ext = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from score path
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_gap_open = torch::zeros({B}, options);
            torch::Tensor tmp_gap_ext = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            nw_affine_backward_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                beta.data_ptr<float>(), posteriors.data_ptr<float>(),
                tmp_gap_open.data_ptr<float>(), tmp_gap_ext.data_ptr<float>(), tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
            );

            grad_scores += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_gap_open += (grad_score * grad_gap_open_fwd).sum().reshape({1});
            total_grad_gap_ext += (grad_score * grad_gap_ext_fwd).sum().reshape({1});
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from alignment path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(
                grad_posteriors.sizes() == scores.sizes(),
                "grad_posteriors must have same shape as scores"
            );
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            nw_affine_hvp_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(), d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(), beta.data_ptr<float>(),
                d_beta.data_ptr<float>(), hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grads
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            nw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_gap_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 0
            );
            total_grad_gap_open += (grad_posteriors * dP_dtheta).sum().reshape({1});

            nw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_gap_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 1
            );
            total_grad_gap_ext += (grad_posteriors * dP_dtheta).sum().reshape({1});

            nw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 2
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        return {grad_scores, total_grad_gap_open, total_grad_gap_ext, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions (CPU) - Affine Gap
// =============================================================================

std::vector<torch::Tensor> soft_nw_affine_cpu(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftNWAffineCPUFunction::apply(scores, gap_open, gap_ext, temperature, lengths);
}

std::vector<torch::Tensor> soft_nw_affine_cpu_float(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
    int B = shape.B;
    int L1 = shape.max_L1;
    int L2 = shape.max_L2;

    torch::Tensor gap_open_t = torch::tensor({static_cast<float>(gap_open)}, scores.options());
    torch::Tensor gap_ext_t = torch::tensor({static_cast<float>(gap_ext)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = lengths_opt.has_value() ? lengths_opt.value()
                                                    : make_default_lengths_2d(B, L1, L2, scores.device());

    return SoftNWAffineCPUFunction::apply(scores, gap_open_t, gap_ext_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_affine_cpu_with_grads(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    nw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    nw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_open.data_ptr<float>(), grad_gap_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return std::make_tuple(score, posteriors, grad_gap_open, grad_gap_ext, grad_T);
}

torch::Tensor soft_nw_affine_hvp_cpu_impl(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
    D2P_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");

    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    nw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    nw_affine_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return H_scores;
}

torch::Tensor soft_nw_affine_param_jacobian_cpu_impl(
    torch::Tensor scores, int64_t param_type,
    double gap_open, double gap_ext, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 2, "param_type must be 0, 1, or 2");

    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    nw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    nw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_open.data_ptr<float>(), grad_gap_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    torch::Tensor dS_dtheta;
    switch (param_type) {
        case 0: dS_dtheta = grad_gap_open; break;
        case 1: dS_dtheta = grad_gap_ext; break;
        case 2: dS_dtheta = grad_T; break;
    }

    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    nw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), param_type
    );

    return dP_dtheta;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_affine_backward_full_cpu_impl(
    torch::Tensor scores, torch::Tensor grad_alignment,
    double gap_open, double gap_ext, double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const NWAffineCPUShape shape = validate_nw_affine_scores_cpu(scores);
    TORCH_CHECK(!grad_alignment.is_cuda(), "grad_alignment must be a CPU tensor");
    TORCH_CHECK(
        grad_alignment.sizes() == scores.sizes(),
        "grad_alignment must have same shape as scores"
    );
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    grad_alignment = grad_alignment.contiguous().to(torch::kFloat32);

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open_fwd = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    nw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    nw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta_fwd.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_open_fwd.data_ptr<float>(), grad_gap_ext_fwd.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    // HVP
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    nw_affine_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_alignment.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    // Param grads
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    nw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_gap_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap_open = (grad_alignment * dP_dtheta).sum().reshape({1});

    nw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_gap_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_gap_ext = (grad_alignment * dP_dtheta).sum().reshape({1});

    nw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 2
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_gap_open, total_grad_gap_ext, total_grad_T);
}

// =============================================================================
// Namespaced API Wrappers
// =============================================================================

std::vector<torch::Tensor> nw_affine_forward_cpu_wrapper(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_cpu_float(scores, gap_open, gap_ext, temperature, lengths_opt);
}

std::vector<torch::Tensor> nw_affine_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return soft_nw_affine_cpu(scores, gap_open, gap_ext, temperature, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
nw_affine_value_grad_params_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    auto result = soft_nw_affine_cpu_with_grads(scores, gap_open, gap_ext, temperature, lengths_opt);
    return std::make_tuple(std::get<0>(result), std::get<2>(result), std::get<3>(result));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nw_affine_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_backward_full_cpu_impl(scores, grad_marginals, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_hvp_cpu_impl(scores, v, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_gap_open_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cpu_impl(scores, 0, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_gap_ext_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cpu_impl(scores, 1, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cpu_impl(scores, 2, gap_open, gap_ext, temperature, lengths_opt);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_nw_affine", soft_nw_affine_cpu);
    m.impl("soft_nw_affine_float", soft_nw_affine_cpu_float);
    m.impl("soft_nw_affine_with_grads", soft_nw_affine_cpu_with_grads);
    m.impl("soft_nw_affine_hvp", soft_nw_affine_hvp_cpu_impl);
    m.impl("soft_nw_affine_param_jacobian", soft_nw_affine_param_jacobian_cpu_impl);
    m.impl("soft_nw_affine_backward_full", soft_nw_affine_backward_full_cpu_impl);
    // Namespaced API
    m.impl("nw_affine_forward", nw_affine_forward_cpu_wrapper);
    m.impl("nw_affine_forward_t", nw_affine_forward_t_cpu);
    m.impl("nw_affine_value_grad_params", nw_affine_value_grad_params_cpu);
    m.impl("nw_affine_marginals_backward", nw_affine_marginals_backward_cpu);
    m.impl("nw_affine_marginals_hvp", nw_affine_marginals_hvp_cpu);
    m.impl("nw_affine_marginals_grad_gap_open", nw_affine_marginals_grad_gap_open_cpu);
    m.impl("nw_affine_marginals_grad_gap_ext", nw_affine_marginals_grad_gap_ext_cpu);
    m.impl("nw_affine_marginals_grad_temp", nw_affine_marginals_grad_temp_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_nw_affine", soft_nw_affine_cpu);
    m.impl("soft_nw_affine_float", soft_nw_affine_cpu_float);
    m.impl("nw_affine_forward", nw_affine_forward_cpu_wrapper);
    m.impl("nw_affine_forward_t", nw_affine_forward_t_cpu);
}

#endif // USE_TORCH_LIBRARY
