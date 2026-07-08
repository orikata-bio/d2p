// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft DTW CUDA Extension with PyTorch Autograd
 *
 * GPU-accelerated soft Dynamic Time Warping with:
 *   - Global alignment using softmin (minimization)
 *   - Optional Sakoe-Chiba bandwidth constraint
 *   - Full gradient support through PyTorch autograd
 *
 * Recurrence:
 *   alpha[i,j] = costs[i,j] + softmin_T(
 *       alpha[i-1,j-1],  // diagonal
 *       alpha[i-1,j],    // up
 *       alpha[i,j-1]     // left
 *   )
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"
#include "common/cuda_utils.h"

// CUDA kernel declarations
#include "dtw/kernels.cuh"

using namespace d2p::common;

namespace {

int checked_dtw_batch_size(int64_t value) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<int>::max(),
        "DTW batch size is too large for CUDA kernels: ", value
    );
    return static_cast<int>(value);
}

int checked_dtw_sequence_dim(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<int>::max() - 1,
        "DTW ", name, " dimension is too large for CUDA kernels: ", value
    );
    return static_cast<int>(value);
}

int64_t checked_dtw_alpha_size(int max_L1, int max_L2) {
    int64_t rows = static_cast<int64_t>(max_L1) + 1;
    int64_t cols = static_cast<int64_t>(max_L2) + 1;
    TORCH_CHECK(
        rows <= std::numeric_limits<int64_t>::max() / cols,
        "DTW alpha table size overflows int64 for dimensions ",
        max_L1, "x", max_L2
    );
    int64_t alpha_size = rows * cols;
    TORCH_CHECK(
        alpha_size <= std::numeric_limits<int>::max(),
        "DTW alpha table is too large for CUDA kernels: ", alpha_size
    );
    TORCH_CHECK(
        static_cast<int64_t>(max_L1) + static_cast<int64_t>(max_L2)
            <= std::numeric_limits<int>::max(),
        "DTW dimensions are too large for CUDA wavefront indexing: ",
        max_L1, "x", max_L2
    );
    return alpha_size;
}

int checked_dtw_bandwidth(int64_t bandwidth) {
    TORCH_CHECK(
        bandwidth >= std::numeric_limits<int>::min()
            && bandwidth <= std::numeric_limits<int>::max(),
        "DTW bandwidth is out of CUDA int range: ", bandwidth
    );
    return static_cast<int>(bandwidth);
}

void validate_dtw_lengths_cuda(
    const torch::Tensor& lengths,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    D2P_CHECK_CUDA(lengths);
    D2P_CHECK_CONTIGUOUS(lengths);
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2);
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");
    TORCH_CHECK(
        lengths.device() == device,
        "lengths must be on same device as costs, got ", lengths.device(), " vs ", device
    );

    auto lengths_cpu = lengths.to(torch::kCPU);
    auto lengths_acc = lengths_cpu.accessor<int32_t, 2>();
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

torch::Tensor resolve_dtw_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_dtw_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

}  // namespace

// =============================================================================
// DTW CUDA Autograd Function
//
// Forward: costs -> (score, posteriors)
// Backward: uses HVP for grad_costs, chains grad_T with upstream grad
//
// DTW is a minimization problem (unlike SW which maximizes).
// =============================================================================

class SoftDTWCUDAFunction : public torch::autograd::Function<SoftDTWCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor costs,
        torch::Tensor temperature,
        torch::Tensor lengths,
        int64_t bandwidth
    ) {
        ctx->set_materialize_grads(false);

        D2P_CHECK_INPUT_CUDA(costs);
        TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, L1, L2)");
        TORCH_CHECK(costs.dtype() == torch::kFloat32, "costs must be float32");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");
        D2P_CUDA_GUARD(costs);

        int B = checked_dtw_batch_size(costs.size(0));
        int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
        int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
        int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);
        int bandwidth_checked = checked_dtw_bandwidth(bandwidth);

        validate_dtw_lengths_cuda(lengths, B, max_L1, max_L2, costs.device());

        float temp_val = temperature.cpu().item<float>();

        auto options = costs.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        // Forward pass: compute alpha and score
        d2p::common::record_streams_current({&costs, &alpha, &score, &lengths});
        dtw_forward(
            costs.data_ptr<float>(),
            alpha.data_ptr<float>(),
            score.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            temp_val, bandwidth_checked
        );

        // Backward pass (of the internal DP): compute posteriors and grad_T
        d2p::common::record_streams_current({&alpha, &costs, &score, &beta, &posteriors, &grad_T, &lengths});
        dtw_backward(
            alpha.data_ptr<float>(),
            costs.data_ptr<float>(),
            score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            temp_val, bandwidth_checked
        );

        // Save for backward (HVP computation)
        ctx->save_for_backward({costs.clone(), alpha.clone(), score.clone(), lengths.clone(), grad_T.clone()});
        ctx->saved_data["temperature"] = temp_val;
        ctx->saved_data["bandwidth"] = static_cast<int64_t>(bandwidth_checked);

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
        int bandwidth = checked_dtw_bandwidth(ctx->saved_data["bandwidth"].toInt());

        int B = checked_dtw_batch_size(costs.size(0));
        int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
        int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
        int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);
        D2P_CUDA_GUARD(costs);

        auto options = costs.options();

        torch::Tensor grad_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_costs = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // ============ Gradient from score ============
        if (grad_score.defined() && grad_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            d2p::common::record_streams_current({&alpha, &costs, &score, &beta, &posteriors, &tmp_T, &lengths});
            dtw_backward(
                alpha.data_ptr<float>(),
                costs.data_ptr<float>(),
                score.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                temp_val, bandwidth
            );

            grad_costs += grad_score.view({B, 1, 1}) * posteriors;
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // ============ Gradient from alignment (posteriors) ============
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(grad_posteriors.sizes() == costs.sizes(),
                        "grad_posteriors shape mismatch");
            TORCH_CHECK(grad_posteriors.is_cuda(),
                        "grad_posteriors must be on CUDA");

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            if (grad_posteriors.device() != costs.device()) {
                grad_posteriors = grad_posteriors.to(costs.device());
            }
            grad_posteriors = grad_posteriors.contiguous();

            // HVP: d^2S/dcosts^2 * grad_posteriors
            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_costs = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &costs, &score, &grad_posteriors, &d_alpha, &d_score, &beta, &d_beta, &hvp_grad_costs, &lengths});
            dtw_hvp(
                alpha.data_ptr<float>(),
                costs.data_ptr<float>(),
                score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_costs.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                temp_val, bandwidth
            );

            grad_costs += hvp_grad_costs;

            // Temperature param grad
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &costs, &score, &U_ws, &beta_ws, &W_ws, &dP_dT, &lengths});
            dtw_param_grad(
                alpha.data_ptr<float>(),
                costs.data_ptr<float>(),
                score.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                temp_val, bandwidth
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        // Return gradients: costs, temperature, lengths (no grad), bandwidth (no grad)
        return {grad_costs, total_grad_T, torch::Tensor(), torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions
// =============================================================================

// DTW with autograd (tensor params for full differentiability)
std::vector<torch::Tensor> soft_dtw_cuda(
    torch::Tensor costs,
    torch::Tensor temperature,
    torch::Tensor lengths,
    int64_t bandwidth
) {
    return SoftDTWCUDAFunction::apply(costs, temperature, lengths, bandwidth);
}

// DTW with float params (convenience function)
std::vector<torch::Tensor> soft_dtw_cuda_float(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    D2P_CHECK_INPUT_CUDA(costs);
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, L1, L2)");
    D2P_CUDA_GUARD(costs);

    int B = checked_dtw_batch_size(costs.size(0));
    int L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
    int L2 = checked_dtw_sequence_dim(costs.size(2), "L2");

    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, costs.options());
    torch::Tensor lengths = resolve_dtw_lengths_cuda(
        lengths_opt, B, L1, L2, costs.device()
    );
    int bandwidth = checked_dtw_bandwidth(bandwidth_opt.has_value() ? bandwidth_opt.value() : -1);

    return SoftDTWCUDAFunction::apply(costs, temp_t, lengths, bandwidth);
}

// DTW with explicit gradients (for debugging/inspection)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_dtw_cuda_with_grads(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    D2P_CHECK_INPUT_CUDA(costs);
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, L1, L2)");
    D2P_CUDA_GUARD(costs);

    int B = checked_dtw_batch_size(costs.size(0));
    int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
    int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
    int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int bandwidth = checked_dtw_bandwidth(bandwidth_opt.has_value() ? bandwidth_opt.value() : -1);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&costs, &alpha, &score, &lengths});
    dtw_forward(
        costs.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    d2p::common::record_streams_current({&alpha, &costs, &score, &beta, &posteriors, &grad_T, &lengths});
    dtw_backward(
        alpha.data_ptr<float>(),
        costs.data_ptr<float>(),
        score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return std::make_tuple(score, posteriors, grad_T);
}

// DTW HVP
torch::Tensor soft_dtw_hvp_cuda(
    torch::Tensor costs,
    torch::Tensor tangent,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    D2P_CHECK_INPUT_CUDA(costs);
    D2P_CHECK_INPUT_CUDA(tangent);
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D");
    TORCH_CHECK(tangent.sizes() == costs.sizes(), "tangent must have same shape as costs");
    TORCH_CHECK(
        tangent.device() == costs.device(),
        "tangent must be on same device as costs, got ", tangent.device(), " vs ", costs.device()
    );
    D2P_CUDA_GUARD(costs);

    int B = checked_dtw_batch_size(costs.size(0));
    int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
    int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
    int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int bandwidth = checked_dtw_bandwidth(bandwidth_opt.has_value() ? bandwidth_opt.value() : -1);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_costs = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&costs, &alpha, &score, &lengths});
    dtw_forward(
        costs.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    d2p::common::record_streams_current({&alpha, &costs, &score, &tangent, &d_alpha, &d_score, &beta, &d_beta, &H_costs, &lengths});
    dtw_hvp(
        alpha.data_ptr<float>(),
        costs.data_ptr<float>(),
        score.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_costs.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return H_costs;
}

// DTW param Jacobian: dP/dT where P = posteriors
torch::Tensor soft_dtw_param_jacobian_cuda(
    torch::Tensor costs,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    D2P_CHECK_INPUT_CUDA(costs);
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D");
    D2P_CUDA_GUARD(costs);

    int B = checked_dtw_batch_size(costs.size(0));
    int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
    int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
    int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int bandwidth = checked_dtw_bandwidth(bandwidth_opt.has_value() ? bandwidth_opt.value() : -1);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&costs, &alpha, &score, &lengths});
    dtw_forward(
        costs.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    d2p::common::record_streams_current({&alpha, &costs, &score, &U, &beta, &W, &dP_dT, &lengths});
    dtw_param_grad(
        alpha.data_ptr<float>(),
        costs.data_ptr<float>(),
        score.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(temperature), bandwidth
    );

    return dP_dT;
}

// Full backward for DTW - returns (grad_costs, grad_temperature)
std::tuple<torch::Tensor, torch::Tensor>
soft_dtw_backward_full_cuda(
    torch::Tensor costs,
    torch::Tensor grad_alignment,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt,
    c10::optional<int64_t> bandwidth_opt
) {
    D2P_CHECK_INPUT_CUDA(costs);
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, L1, L2)");
    TORCH_CHECK(grad_alignment.sizes() == costs.sizes(), "grad_alignment must have same shape as costs");
    TORCH_CHECK(grad_alignment.is_cuda(), "grad_alignment must be a CUDA tensor");
    TORCH_CHECK(
        grad_alignment.device() == costs.device(),
        "grad_alignment must be on same device as costs, got ",
        grad_alignment.device(), " vs ", costs.device()
    );
    D2P_CUDA_GUARD(costs);

    int B = checked_dtw_batch_size(costs.size(0));
    int max_L1 = checked_dtw_sequence_dim(costs.size(1), "L1");
    int max_L2 = checked_dtw_sequence_dim(costs.size(2), "L2");
    int64_t alpha_size = checked_dtw_alpha_size(max_L1, max_L2);

    auto options = costs.options();
    torch::Tensor lengths = resolve_dtw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, costs.device()
    );
    int bandwidth = checked_dtw_bandwidth(bandwidth_opt.has_value() ? bandwidth_opt.value() : -1);

    grad_alignment = grad_alignment.contiguous();
    if (grad_alignment.dtype() != torch::kFloat32) {
        grad_alignment = grad_alignment.to(torch::kFloat32);
    }

    // Forward pass
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&costs, &alpha, &score, &lengths});
    dtw_forward(
        costs.data_ptr<float>(), alpha.data_ptr<float>(),
        score.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(temperature), bandwidth
    );

    d2p::common::record_streams_current({&alpha, &costs, &score, &beta_fwd, &posteriors, &grad_T_fwd, &lengths});
    dtw_backward(
        alpha.data_ptr<float>(), costs.data_ptr<float>(),
        score.data_ptr<float>(), beta_fwd.data_ptr<float>(),
        posteriors.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(temperature), bandwidth
    );

    // HVP for grad_costs
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_costs = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &costs, &score, &grad_alignment, &d_alpha, &d_score, &beta, &d_beta, &grad_costs, &lengths});
    dtw_hvp(
        alpha.data_ptr<float>(), costs.data_ptr<float>(),
        score.data_ptr<float>(), grad_alignment.data_ptr<float>(),
        d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(),
        grad_costs.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(temperature), bandwidth
    );

    // Param grad
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &costs, &score, &U_ws, &beta_ws, &W_ws, &dP_dT, &lengths});
    dtw_param_grad(
        alpha.data_ptr<float>(), costs.data_ptr<float>(),
        score.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(),
        dP_dT.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(temperature), bandwidth
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_costs, total_grad_T);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

// Register CUDA implementations
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_dtw", soft_dtw_cuda);
    m.impl("soft_dtw_float", soft_dtw_cuda_float);
    m.impl("soft_dtw_with_grads", soft_dtw_cuda_with_grads);
    m.impl("soft_dtw_hvp", soft_dtw_hvp_cuda);
    m.impl("soft_dtw_param_jacobian", soft_dtw_param_jacobian_cuda);
    m.impl("soft_dtw_backward_full", soft_dtw_backward_full_cuda);
}

// Register Autograd implementations
TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_dtw", soft_dtw_cuda);
    m.impl("soft_dtw_float", soft_dtw_cuda_float);
}

#endif // USE_TORCH_LIBRARY
