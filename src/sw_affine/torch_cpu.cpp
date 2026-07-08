// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief CPU PyTorch bindings for Affine Gap Smith-Waterman
 *
 * CPU implementations that mirror the CUDA interface.
 * Three-state DP (Match, Insert, Delete) with separate gap open/extend penalties.
 *
 * Operators registered:
 *   - soft_sw_affine: Tensor params for full differentiability
 *   - soft_sw_affine_float: Float params (convenience)
 *   - soft_sw_affine_with_grads: Forward + backward in one call
 *   - soft_sw_affine_hvp: Hessian-vector product
 *   - soft_sw_affine_param_jacobian: Parameter gradients
 *   - soft_sw_affine_backward_full: Full backward pass
 *   - sw_affine_* namespace: Clean API wrappers
 */

#include <torch/extension.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"

// CPU kernel declarations
#include "sw_affine/kernels_cpu.h"

using namespace d2p::common;

namespace {

struct SwAffineCpuShape {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

SwAffineCpuShape validate_sw_affine_scores_shape_cpu(const torch::Tensor& scores) {
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    const int64_t B = scores.size(0);
    const int64_t max_L1 = scores.size(1);
    const int64_t max_L2 = scores.size(2);
    const int64_t max_int = static_cast<int64_t>(std::numeric_limits<int>::max());

    TORCH_CHECK(B <= max_int, "scores batch dimension is too large for sw_affine CPU");
    TORCH_CHECK(max_L1 <= max_int, "scores L1 dimension is too large for sw_affine CPU");
    TORCH_CHECK(max_L2 <= max_int, "scores L2 dimension is too large for sw_affine CPU");

    const int64_t rows = max_L1 + 1;
    const int64_t cols = max_L2 + 1;
    const int64_t max_i64 = std::numeric_limits<int64_t>::max();
    TORCH_CHECK(rows <= max_i64 / cols, "sw_affine CPU DP table size overflows int64");
    const int64_t cells = rows * cols;
    TORCH_CHECK(cells <= max_i64 / 3, "sw_affine CPU DP table size overflows int64");

    const int64_t alpha_size = 3 * cells;
    TORCH_CHECK(
        alpha_size <= max_int,
        "sw_affine CPU DP table is too large: 3 * (L1 + 1) * (L2 + 1) = ",
        alpha_size
    );

    return {
        static_cast<int>(B),
        static_cast<int>(max_L1),
        static_cast<int>(max_L2),
        alpha_size
    };
}

void validate_sw_affine_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
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

torch::Tensor resolve_sw_affine_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_sw_affine_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// =============================================================================
// Affine SW CPU Autograd Function
//
// Three-state DP: Match (M), Insert (I), Delete (D)
// M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
// I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)
// D[i,j] = LSE_T(M[i,j-1] + gap_open, D[i,j-1] + gap_ext)
// =============================================================================

class SoftSWAffineCPUFunction : public torch::autograd::Function<SoftSWAffineCPUFunction> {
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

        D2P_CHECK_INPUT_CPU(scores);
        TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
        TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");

        const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
        const int B = shape.B;
        const int max_L1 = shape.max_L1;
        const int max_L2 = shape.max_L2;
        const int64_t alpha_size = shape.alpha_size;

        validate_sw_affine_lengths_cpu(lengths, B, max_L1, max_L2);

        float gap_open_val = gap_open.item<float>();
        float gap_ext_val = gap_ext.item<float>();
        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_open = torch::zeros({B}, options);
        torch::Tensor grad_ext = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        sw_affine_forward_cpu(
            scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
        );

        sw_affine_backward_cpu(
            alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
            beta.data_ptr<float>(), posteriors.data_ptr<float>(),
            grad_open.data_ptr<float>(), grad_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), partition.clone(), lengths.clone(),
                                grad_open.clone(), grad_ext.clone(), grad_T.clone()});
        ctx->saved_data["gap_open"] = gap_open_val;
        ctx->saved_data["gap_ext"] = gap_ext_val;
        ctx->saved_data["temperature"] = temp_val;

        return {partition, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor scores = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor partition = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_open_fwd = saved[4];
        torch::Tensor grad_ext_fwd = saved[5];
        torch::Tensor grad_T_fwd = saved[6];

        float gap_open_val = static_cast<float>(ctx->saved_data["gap_open"].toDouble());
        float gap_ext_val = static_cast<float>(ctx->saved_data["gap_ext"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
        const int B = shape.B;
        const int max_L1 = shape.max_L1;
        const int max_L2 = shape.max_L2;
        const int64_t alpha_size = shape.alpha_size;

        auto options = scores.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_open = torch::zeros({1}, options);
        torch::Tensor total_grad_ext = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from score path
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_open = torch::zeros({B}, options);
            torch::Tensor tmp_ext = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            sw_affine_backward_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                beta.data_ptr<float>(), posteriors.data_ptr<float>(),
                tmp_open.data_ptr<float>(), tmp_ext.data_ptr<float>(), tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
            );

            grad_scores += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_open += (grad_score * grad_open_fwd).sum().reshape({1});
            total_grad_ext += (grad_score * grad_ext_fwd).sum().reshape({1});
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from alignment path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            sw_affine_hvp_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(), d_alpha.data_ptr<float>(),
                d_partition.data_ptr<float>(), d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grads
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            // gap_open
            sw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 0
            );
            total_grad_open += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // gap_ext
            sw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 1
            );
            total_grad_ext += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // temperature
            sw_affine_param_grad_cpu(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 2
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        return {grad_scores, total_grad_open, total_grad_ext, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions (CPU)
// =============================================================================

std::vector<torch::Tensor> soft_sw_affine_cpu(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftSWAffineCPUFunction::apply(scores, gap_open, gap_ext, temperature, lengths);
}

std::vector<torch::Tensor> soft_sw_affine_cpu_float(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    int B = scores.size(0);
    int L1 = scores.size(1);
    int L2 = scores.size(2);

    torch::Tensor gap_open_t = torch::tensor({static_cast<float>(gap_open)}, scores.options());
    torch::Tensor gap_ext_t = torch::tensor({static_cast<float>(gap_ext)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_sw_affine_lengths_cpu(
        lengths_opt, B, L1, L2, scores.device()
    );

    return SoftSWAffineCPUFunction::apply(scores, gap_open_t, gap_ext_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_affine_cpu_with_grads(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CPU(scores);
    const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
    const int B = shape.B;
    const int max_L1 = shape.max_L1;
    const int max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open = torch::zeros({B}, options);
    torch::Tensor grad_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    sw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    sw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_open.data_ptr<float>(), grad_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return std::make_tuple(partition, posteriors, grad_open, grad_ext, grad_T);
}

torch::Tensor soft_sw_affine_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CPU(scores);
    D2P_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");

    const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
    const int B = shape.B;
    const int max_L1 = shape.max_L1;
    const int max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    sw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    sw_affine_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
        d_beta.data_ptr<float>(), H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return H_scores;
}

torch::Tensor soft_sw_affine_param_jacobian_cpu(
    torch::Tensor scores,
    int64_t param_type,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CPU(scores);
    const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
    const int B = shape.B;
    const int max_L1 = shape.max_L1;
    const int max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open = torch::zeros({B}, options);
    torch::Tensor grad_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    sw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    sw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_open.data_ptr<float>(), grad_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    torch::Tensor dS_dtheta;
    switch (param_type) {
        case 0: dS_dtheta = grad_open; break;
        case 1: dS_dtheta = grad_ext; break;
        case 2: dS_dtheta = grad_T; break;
        default: dS_dtheta = grad_open;
    }

    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    sw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature),
        param_type
    );

    return dP_dtheta;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_affine_backward_full_cpu(
    torch::Tensor scores,
    torch::Tensor grad_alignment,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CPU(scores);
    TORCH_CHECK(!grad_alignment.is_cuda(), "grad_alignment must be a CPU tensor");
    TORCH_CHECK(grad_alignment.dtype() == torch::kFloat32, "grad_alignment must be float32");
    TORCH_CHECK(
        grad_alignment.sizes() == scores.sizes(),
        "grad_alignment must have same shape as scores"
    );

    const SwAffineCpuShape shape = validate_sw_affine_scores_shape_cpu(scores);
    const int B = shape.B;
    const int max_L1 = shape.max_L1;
    const int max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    grad_alignment = grad_alignment.contiguous().to(torch::kFloat32);

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open_fwd = torch::zeros({B}, options);
    torch::Tensor grad_ext_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    sw_affine_forward_cpu(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), partition.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    sw_affine_backward_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        beta_fwd.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_open_fwd.data_ptr<float>(), grad_ext_fwd.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    // HVP
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    sw_affine_hvp_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_alignment.data_ptr<float>(), d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
        d_beta.data_ptr<float>(), grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    // Param grads
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    sw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_open = (grad_alignment * dP_dtheta).sum().reshape({1});

    sw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_ext = (grad_alignment * dP_dtheta).sum().reshape({1});

    sw_affine_param_grad_cpu(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), partition.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 2
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_open, total_grad_ext, total_grad_T);
}

// =============================================================================
// Namespaced API (sw_affine::*)
// These are thin wrappers around existing functions with cleaner names
// =============================================================================

// sw_affine::forward - returns (value, marginals)
std::vector<torch::Tensor> d2p_sw_affine_forward_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_cpu_float(scores, gap_open, gap_ext, temp, lengths);
}

// sw_affine::forward_t - tensor params version
std::vector<torch::Tensor> d2p_sw_affine_forward_t_cpu(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_sw_affine_cpu(scores, gap_open, gap_ext, temp, lengths);
}

// sw_affine::value_grad_params - returns (grad_gap_open, grad_gap_ext, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> d2p_sw_affine_value_grad_params_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_sw_affine_cpu_with_grads(scores, gap_open, gap_ext, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result), std::get<4>(result));
}

// sw_affine::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> d2p_sw_affine_marginals_backward_cpu(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_backward_full_cpu(scores, grad_marginals, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_hvp - Hessian-vector product
torch::Tensor d2p_sw_affine_marginals_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor v,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_hvp_cpu(scores, v, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_gap_open - d(marginals)/d(gap_open)
torch::Tensor d2p_sw_affine_marginals_grad_gap_open_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cpu(scores, 0, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_gap_ext - d(marginals)/d(gap_ext)
torch::Tensor d2p_sw_affine_marginals_grad_gap_ext_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cpu(scores, 1, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor d2p_sw_affine_marginals_grad_temp_cpu(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cpu(scores, 2, gap_open, gap_ext, temp, lengths);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU (Affine SW only)
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    // Core affine SW operators
    m.impl("soft_sw_affine", soft_sw_affine_cpu);
    m.impl("soft_sw_affine_float", soft_sw_affine_cpu_float);
    m.impl("soft_sw_affine_with_grads", soft_sw_affine_cpu_with_grads);
    m.impl("soft_sw_affine_hvp", soft_sw_affine_hvp_cpu);
    m.impl("soft_sw_affine_param_jacobian", soft_sw_affine_param_jacobian_cpu);
    m.impl("soft_sw_affine_backward_full", soft_sw_affine_backward_full_cpu);

    // sw_affine namespace
    m.impl("sw_affine_forward", d2p_sw_affine_forward_cpu);
    m.impl("sw_affine_forward_t", d2p_sw_affine_forward_t_cpu);
    m.impl("sw_affine_value_grad_params", d2p_sw_affine_value_grad_params_cpu);
    m.impl("sw_affine_marginals_backward", d2p_sw_affine_marginals_backward_cpu);
    m.impl("sw_affine_marginals_hvp", d2p_sw_affine_marginals_hvp_cpu);
    m.impl("sw_affine_marginals_grad_gap_open", d2p_sw_affine_marginals_grad_gap_open_cpu);
    m.impl("sw_affine_marginals_grad_gap_ext", d2p_sw_affine_marginals_grad_gap_ext_cpu);
    m.impl("sw_affine_marginals_grad_temp", d2p_sw_affine_marginals_grad_temp_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_sw_affine", soft_sw_affine_cpu);
    m.impl("soft_sw_affine_float", soft_sw_affine_cpu_float);

    // sw_affine namespace - autograd versions
    m.impl("sw_affine_forward", d2p_sw_affine_forward_cpu);
    m.impl("sw_affine_forward_t", d2p_sw_affine_forward_t_cpu);
}

#endif // USE_TORCH_LIBRARY
