// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft DTW CPU Extension with PyTorch Autograd
 *
 * CPU implementations that mirror the CUDA interface.
 * Registered with TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * Recurrence:
 *   alpha[i,j] = costs[i,j] + softmin_T(
 *       alpha[i-1,j-1],  // diagonal
 *       alpha[i-1,j],    // up
 *       alpha[i,j-1]     // left
 *   )
 */

#include <torch/extension.h>
#include <cstdint>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"

// CPU kernel declarations
#include "dtw/kernels_cpu.h"

using namespace d2p::common;

namespace {

constexpr int64_t kDtwCpuMaxIndex = std::numeric_limits<int>::max();

void validate_dtw_cost_shape_cpu(const torch::Tensor& costs) {
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, L1, L2)");
    TORCH_CHECK(
        costs.size(0) <= kDtwCpuMaxIndex,
        "DTW CPU batch size exceeds supported int32 range: got ", costs.size(0)
    );
    TORCH_CHECK(
        costs.size(1) <= kDtwCpuMaxIndex,
        "DTW CPU L1 dimension exceeds supported int32 range: got ", costs.size(1)
    );
    TORCH_CHECK(
        costs.size(2) <= kDtwCpuMaxIndex,
        "DTW CPU L2 dimension exceeds supported int32 range: got ", costs.size(2)
    );
}

int64_t checked_dtw_alpha_size_cpu(int64_t max_L1, int64_t max_L2) {
    TORCH_CHECK(
        max_L1 == 0 || max_L2 <= kDtwCpuMaxIndex / max_L1,
        "DTW CPU cost matrix exceeds supported int32 index range: max_L1 * max_L2 = ",
        max_L1, " * ", max_L2
    );
    int64_t cost_size = max_L1 * max_L2;
    TORCH_CHECK(
        cost_size <= kDtwCpuMaxIndex,
        "DTW CPU cost matrix exceeds supported int32 index range: got ",
        cost_size, " cells, max ", kDtwCpuMaxIndex
    );

    int64_t alpha_rows = max_L1 + 1;
    int64_t alpha_cols = max_L2 + 1;
    TORCH_CHECK(
        alpha_rows <= kDtwCpuMaxIndex / alpha_cols,
        "DTW CPU DP workspace exceeds supported int32 index range: (max_L1 + 1) * (max_L2 + 1) = ",
        alpha_rows, " * ", alpha_cols
    );
    int64_t alpha_size = alpha_rows * alpha_cols;
    TORCH_CHECK(
        alpha_size <= kDtwCpuMaxIndex,
        "DTW CPU DP workspace exceeds supported int32 index range: got ",
        alpha_size, " cells, max ", kDtwCpuMaxIndex
    );
    return alpha_size;
}

void validate_dtw_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
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

torch::Tensor resolve_dtw_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_dtw_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

}  // namespace

// =============================================================================
// DTW CPU Autograd Function
// =============================================================================

class SoftDTWCPUFunction : public torch::autograd::Function<SoftDTWCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor costs,
        torch::Tensor temperature,
        torch::Tensor lengths,
        int64_t bandwidth
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP/param-grad path.
        ctx->set_materialize_grads(false);

        validate_dtw_cost_shape_cpu(costs);
        D2P_CHECK_INPUT_CPU(costs);
        TORCH_CHECK(costs.dtype() == torch::kFloat32, "costs must be float32");

        int B = costs.size(0);
        int max_L1 = costs.size(1);
        int max_L2 = costs.size(2);
        int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

        validate_dtw_lengths_cpu(lengths, B, max_L1, max_L2);

        float temp_val = temperature.item<float>();

        auto options = costs.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        dtw_forward_cpu(
            costs.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, temp_val, bandwidth
        );

        dtw_backward_cpu(
            alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
            beta.data_ptr<float>(), posteriors.data_ptr<float>(), grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, temp_val, bandwidth
        );

        ctx->save_for_backward({costs.clone(), alpha.clone(), score.clone(), lengths.clone(), grad_T.clone()});
        ctx->saved_data["temperature"] = temp_val;
        ctx->saved_data["bandwidth"] = bandwidth;

        return {score, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor costs = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor score = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_T_fwd = saved[4];

        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());
        int64_t bandwidth = ctx->saved_data["bandwidth"].toInt();

        validate_dtw_cost_shape_cpu(costs);
        int B = costs.size(0);
        int max_L1 = costs.size(1);
        int max_L2 = costs.size(2);
        int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

        auto options = costs.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_costs = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from score path
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            dtw_backward_cpu(
                alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
                beta.data_ptr<float>(), posteriors.data_ptr<float>(), tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, temp_val, bandwidth
            );

            grad_costs += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from alignment path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_costs = torch::zeros({B, max_L1, max_L2}, options);

            dtw_hvp_cpu(
                alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(), d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(), beta.data_ptr<float>(),
                d_beta.data_ptr<float>(), hvp_grad_costs.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, temp_val, bandwidth
            );

            grad_costs += hvp_grad_costs;

            // Temperature param grad
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

            dtw_param_grad_cpu(
                alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
                U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, temp_val, bandwidth
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        return {grad_costs, total_grad_T, torch::Tensor(), torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions (CPU)
// =============================================================================

std::vector<torch::Tensor> soft_dtw_cpu(
    torch::Tensor costs,
    torch::Tensor temperature,
    torch::Tensor lengths,
    int64_t bandwidth
) {
    return SoftDTWCPUFunction::apply(costs, temperature, lengths, bandwidth);
}

std::vector<torch::Tensor> soft_dtw_cpu_float(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    validate_dtw_cost_shape_cpu(costs);
    int B = costs.size(0);
    int L1 = costs.size(1);
    int L2 = costs.size(2);

    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, costs.options());
    torch::Tensor lengths = resolve_dtw_lengths_cpu(
        lengths_opt, B, L1, L2, costs.device()
    );
    int64_t bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() : -1;

    return SoftDTWCPUFunction::apply(costs, temp_t, lengths, bandwidth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_dtw_cpu_with_grads(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    validate_dtw_cost_shape_cpu(costs);
    D2P_CHECK_INPUT_CPU(costs);
    int B = costs.size(0);
    int max_L1 = costs.size(1);
    int max_L2 = costs.size(2);
    int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int64_t bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() : -1;

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    dtw_forward_cpu(
        costs.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    dtw_backward_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return std::make_tuple(score, posteriors, grad_T);
}

torch::Tensor soft_dtw_hvp_cpu(
    torch::Tensor costs,
    torch::Tensor tangent,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    validate_dtw_cost_shape_cpu(costs);
    D2P_CHECK_INPUT_CPU(costs);
    D2P_CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(tangent.sizes() == costs.sizes(), "tangent must have same shape as costs");
    int B = costs.size(0);
    int max_L1 = costs.size(1);
    int max_L2 = costs.size(2);
    int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int64_t bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() : -1;

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_costs = torch::zeros({B, max_L1, max_L2}, options);

    dtw_forward_cpu(
        costs.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    dtw_hvp_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), H_costs.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return H_costs;
}

torch::Tensor soft_dtw_param_jacobian_cpu(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    validate_dtw_cost_shape_cpu(costs);
    D2P_CHECK_INPUT_CPU(costs);
    int B = costs.size(0);
    int max_L1 = costs.size(1);
    int max_L2 = costs.size(2);
    int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int64_t bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() : -1;

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    dtw_forward_cpu(
        costs.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    dtw_param_grad_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        U.data_ptr<float>(), beta.data_ptr<float>(), W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return dP_dT;
}

std::tuple<torch::Tensor, torch::Tensor>
soft_dtw_backward_full_cpu(
    torch::Tensor costs,
    torch::Tensor grad_alignment,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    validate_dtw_cost_shape_cpu(costs);
    D2P_CHECK_INPUT_CPU(costs);
    TORCH_CHECK(!grad_alignment.is_cuda(), "grad_alignment must be a CPU tensor");
    TORCH_CHECK(grad_alignment.sizes() == costs.sizes(), "grad_alignment must have same shape as costs");
    int B = costs.size(0);
    int max_L1 = costs.size(1);
    int max_L2 = costs.size(2);
    int64_t alpha_size = checked_dtw_alpha_size_cpu(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cpu(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int64_t bandwidth = bandwidth_opt.has_value() ? bandwidth_opt.value() : -1;

    grad_alignment = grad_alignment.contiguous().to(torch::kFloat32);

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    dtw_forward_cpu(
        costs.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    dtw_backward_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        beta_fwd.data_ptr<float>(), posteriors.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    // HVP
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_costs = torch::zeros({B, max_L1, max_L2}, options);

    dtw_hvp_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        grad_alignment.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), grad_costs.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    // Param grad
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    dtw_param_grad_cpu(
        alpha.data_ptr<float>(), costs.data_ptr<float>(), score.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_costs, total_grad_T);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_dtw", soft_dtw_cpu);
    m.impl("soft_dtw_float", soft_dtw_cpu_float);
    m.impl("soft_dtw_with_grads", soft_dtw_cpu_with_grads);
    m.impl("soft_dtw_hvp", soft_dtw_hvp_cpu);
    m.impl("soft_dtw_param_jacobian", soft_dtw_param_jacobian_cpu);
    m.impl("soft_dtw_backward_full", soft_dtw_backward_full_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_dtw", soft_dtw_cpu);
    m.impl("soft_dtw_float", soft_dtw_cpu_float);
}

#endif // USE_TORCH_LIBRARY
