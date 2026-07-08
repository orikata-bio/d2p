// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief PyTorch CUDA bindings for Soft Needleman-Wunsch Affine Gap
 *
 * Provides GPU-accelerated implementation of soft NW with affine gap penalty.
 * Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
 *
 * All operations support PyTorch autograd for automatic differentiation.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "common/torch_utils.h"
#include "common/cuda_utils.h"
#include "nw_affine/kernels.cuh"

using namespace d2p::common;

namespace {

struct NWAffineDims {
    int B;
    int L1;
    int L2;
    int64_t alpha_size;
};

int checked_int_dim(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<int>::max(),
        "scores dimension ", name, " must fit in int32, got ", value
    );
    return static_cast<int>(value);
}

int64_t checked_nw_affine_alpha_size(int max_L1, int max_L2) {
    const int64_t rows = static_cast<int64_t>(max_L1) + 1;
    const int64_t cols = static_cast<int64_t>(max_L2) + 1;
    TORCH_CHECK(
        rows <= std::numeric_limits<int64_t>::max() / 3 / cols,
        "nw_affine alpha workspace is too large for scores shape"
    );
    const int64_t alpha_size = 3 * rows * cols;
    TORCH_CHECK(
        alpha_size <= std::numeric_limits<int>::max(),
        "nw_affine alpha workspace size must fit in int32, got ", alpha_size
    );
    return alpha_size;
}

void validate_nw_affine_scores_cuda(const torch::Tensor& scores) {
    D2P_CHECK_INPUT_CUDA(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
}

NWAffineDims checked_nw_affine_dims(const torch::Tensor& scores) {
    const int B = checked_int_dim(scores.size(0), "B");
    const int max_L1 = checked_int_dim(scores.size(1), "L1");
    const int max_L2 = checked_int_dim(scores.size(2), "L2");
    TORCH_CHECK(
        max_L1 <= std::numeric_limits<int>::max() - max_L2,
        "scores dimensions L1 + L2 must fit in int32, got ",
        max_L1,
        " + ",
        max_L2
    );
    const int64_t alpha_size = checked_nw_affine_alpha_size(max_L1, max_L2);
    TORCH_CHECK(
        B == 0 || alpha_size <= std::numeric_limits<int64_t>::max() / B,
        "nw_affine alpha workspace is too large for scores shape"
    );
    return {B, max_L1, max_L2, alpha_size};
}

void validate_nw_affine_tangent_cuda(
    const torch::Tensor& scores,
    const torch::Tensor& tangent
) {
    D2P_CHECK_INPUT_CUDA(tangent);
    TORCH_CHECK(tangent.dim() == 3, "tangent must be 3D (B, L1, L2)");
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    TORCH_CHECK(
        tangent.device() == scores.device(),
        "tangent must be on same device as scores, got ",
        tangent.device(),
        " vs ",
        scores.device()
    );
}

void validate_nw_affine_grad_cuda(
    const torch::Tensor& scores,
    const torch::Tensor& grad,
    const char* name
) {
    D2P_CHECK_CUDA(grad);
    TORCH_CHECK(grad.dim() == 3, name, " must be 3D (B, L1, L2)");
    TORCH_CHECK(
        grad.sizes() == scores.sizes(),
        name, " must have same shape as scores"
    );
    TORCH_CHECK(
        grad.device() == scores.device(),
        name, " must be on same device as scores, got ",
        grad.device(),
        " vs ",
        scores.device()
    );
}

void validate_nw_affine_lengths_cuda(
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
        "lengths must be on same device as scores, got ", lengths.device(), " vs ", device
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

torch::Tensor resolve_nw_affine_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_nw_affine_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

}  // namespace

// =============================================================================
// Soft NW Affine CUDA Autograd Function (3-state DP)
// =============================================================================

class SoftNWAffineCUDAFunction : public torch::autograd::Function<SoftNWAffineCUDAFunction> {
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
        validate_nw_affine_scores_cuda(scores);
        TORCH_CHECK(gap_open.numel() == 1, "gap_open must be a scalar tensor");
        TORCH_CHECK(gap_ext.numel() == 1, "gap_ext must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        NWAffineDims dims = checked_nw_affine_dims(scores);
        int B = dims.B;
        int max_L1 = dims.L1;
        int max_L2 = dims.L2;
        int64_t alpha_size = dims.alpha_size;  // 3 states: M, I, D

        D2P_CUDA_GUARD(scores);
        validate_nw_affine_lengths_cuda(lengths, B, max_L1, max_L2, scores.device());

        float gap_open_val = gap_open.cpu().item<float>();
        float gap_ext_val = gap_ext.cpu().item<float>();
        float temp_val = temperature.cpu().item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap_open = torch::zeros({B}, options);
        torch::Tensor grad_gap_ext = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
        nw_affine_forward(
            scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
            lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val
        );

        d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap_open, &grad_gap_ext, &grad_T, &lengths});
        nw_affine_backward(
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

        NWAffineDims dims = checked_nw_affine_dims(scores);
        int B = dims.B;
        int max_L1 = dims.L1;
        int max_L2 = dims.L2;
        int64_t alpha_size = dims.alpha_size;
        D2P_CUDA_GUARD(scores);

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

            d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &tmp_gap_open, &tmp_gap_ext, &tmp_T, &lengths});
            nw_affine_backward(
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
            validate_nw_affine_grad_cuda(scores, grad_posteriors, "grad_posteriors");
            grad_posteriors = grad_posteriors.contiguous();
            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_posteriors, &d_alpha, &d_score, &beta, &d_beta, &hvp_grad_scores, &lengths});
            nw_affine_hvp(
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

            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_open_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            nw_affine_param_grad(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_gap_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 0
            );
            total_grad_gap_open += (grad_posteriors * dP_dtheta).sum().reshape({1});

            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_ext_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            nw_affine_param_grad(
                alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
                grad_gap_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(), B, max_L1, max_L2, gap_open_val, gap_ext_val, temp_val, 1
            );
            total_grad_gap_ext += (grad_posteriors * dP_dtheta).sum().reshape({1});

            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            nw_affine_param_grad(
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
// Python Interface Functions (CUDA) - Affine Gap
// =============================================================================

std::vector<torch::Tensor> soft_nw_affine_cuda(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    validate_nw_affine_scores_cuda(scores);
    D2P_CUDA_GUARD(scores);
    return SoftNWAffineCUDAFunction::apply(scores, gap_open, gap_ext, temperature, lengths);
}

std::vector<torch::Tensor> soft_nw_affine_cuda_float(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_nw_affine_scores_cuda(scores);
    NWAffineDims dims = checked_nw_affine_dims(scores);
    int B = dims.B;
    int L1 = dims.L1;
    int L2 = dims.L2;
    D2P_CUDA_GUARD(scores);

    torch::Tensor gap_open_t = torch::tensor({static_cast<float>(gap_open)}, scores.options());
    torch::Tensor gap_ext_t = torch::tensor({static_cast<float>(gap_ext)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = lengths_opt.has_value() ? lengths_opt.value()
                                                    : make_default_lengths_2d(B, L1, L2, scores.device());

    return SoftNWAffineCUDAFunction::apply(scores, gap_open_t, gap_ext_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_affine_cuda_with_grads(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_nw_affine_scores_cuda(scores);
    NWAffineDims dims = checked_nw_affine_dims(scores);
    int B = dims.B;
    int max_L1 = dims.L1;
    int max_L2 = dims.L2;
    int64_t alpha_size = dims.alpha_size;
    D2P_CUDA_GUARD(scores);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_affine_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap_open, &grad_gap_ext, &grad_T, &lengths});
    nw_affine_backward(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        beta.data_ptr<float>(), posteriors.data_ptr<float>(),
        grad_gap_open.data_ptr<float>(), grad_gap_ext.data_ptr<float>(), grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return std::make_tuple(score, posteriors, grad_gap_open, grad_gap_ext, grad_T);
}

torch::Tensor soft_nw_affine_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_nw_affine_scores_cuda(scores);
    validate_nw_affine_tangent_cuda(scores, tangent);
    NWAffineDims dims = checked_nw_affine_dims(scores);
    int B = dims.B;
    int max_L1 = dims.L1;
    int max_L2 = dims.L2;
    int64_t alpha_size = dims.alpha_size;
    D2P_CUDA_GUARD(scores);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_affine_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &tangent, &d_alpha, &d_score, &beta, &d_beta, &H_scores, &lengths});
    nw_affine_hvp(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        tangent.data_ptr<float>(), d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(), H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    return H_scores;
}

torch::Tensor soft_nw_affine_param_jacobian_cuda(
    torch::Tensor scores,
    int64_t param_type,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_nw_affine_scores_cuda(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 2, "param_type must be 0, 1, or 2");

    NWAffineDims dims = checked_nw_affine_dims(scores);
    int B = dims.B;
    int max_L1 = dims.L1;
    int max_L2 = dims.L2;
    int64_t alpha_size = dims.alpha_size;
    D2P_CUDA_GUARD(scores);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_affine_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap_open, &grad_gap_ext, &grad_T, &lengths});
    nw_affine_backward(
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

    d2p::common::record_streams_current({&alpha, &scores, &score, &dS_dtheta, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), param_type
    );

    return dP_dtheta;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_affine_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_alignment,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_nw_affine_scores_cuda(scores);
    validate_nw_affine_grad_cuda(scores, grad_alignment, "grad_alignment");
    NWAffineDims dims = checked_nw_affine_dims(scores);
    int B = dims.B;
    int max_L1 = dims.L1;
    int max_L2 = dims.L2;
    int64_t alpha_size = dims.alpha_size;
    D2P_CUDA_GUARD(scores);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, scores.device()
    );

    grad_alignment = grad_alignment.contiguous();
    if (grad_alignment.dtype() != torch::kFloat32) {
        grad_alignment = grad_alignment.to(torch::kFloat32);
    }

    // Forward
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_open_fwd = torch::zeros({B}, options);
    torch::Tensor grad_gap_ext_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_affine_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(), score.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &beta_fwd, &posteriors, &grad_gap_open_fwd, &grad_gap_ext_fwd, &grad_T_fwd, &lengths});
    nw_affine_backward(
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

    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_alignment, &d_alpha, &d_score, &beta, &d_beta, &grad_scores, &lengths});
    nw_affine_hvp(
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

    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_open_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_gap_open_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap_open = (grad_alignment * dP_dtheta).sum().reshape({1});

    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_ext_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(), score.data_ptr<float>(),
        grad_gap_ext_fwd.data_ptr<float>(), U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(), W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_gap_ext = (grad_alignment * dP_dtheta).sum().reshape({1});

    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_affine_param_grad(
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

std::vector<torch::Tensor> nw_affine_forward_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_cuda_float(scores, gap_open, gap_ext, temperature, lengths_opt);
}

std::vector<torch::Tensor> nw_affine_forward_t_cuda(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return soft_nw_affine_cuda(scores, gap_open, gap_ext, temperature, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
nw_affine_value_grad_params_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    auto result = soft_nw_affine_cuda_with_grads(scores, gap_open, gap_ext, temperature, lengths_opt);
    // Returns (score, grad_gap_open, grad_gap_ext) - omitting posteriors and grad_T
    return std::make_tuple(std::get<0>(result), std::get<2>(result), std::get<3>(result));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
nw_affine_marginals_backward_cuda(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_backward_full_cuda(scores, grad_marginals, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor v,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_hvp_cuda(scores, v, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_gap_open_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cuda(scores, 0, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_gap_ext_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cuda(scores, 1, gap_open, gap_ext, temperature, lengths_opt);
}

torch::Tensor nw_affine_marginals_grad_temp_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    return soft_nw_affine_param_jacobian_cuda(scores, 2, gap_open, gap_ext, temperature, lengths_opt);
}

// =============================================================================
// Module Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

// Register CUDA implementations
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_nw_affine", soft_nw_affine_cuda);
    m.impl("soft_nw_affine_float", soft_nw_affine_cuda_float);
    m.impl("soft_nw_affine_with_grads", soft_nw_affine_cuda_with_grads);
    m.impl("soft_nw_affine_hvp", soft_nw_affine_hvp_cuda);
    m.impl("soft_nw_affine_param_jacobian", soft_nw_affine_param_jacobian_cuda);
    m.impl("soft_nw_affine_backward_full", soft_nw_affine_backward_full_cuda);
    // Namespaced API
    m.impl("nw_affine_forward", nw_affine_forward_cuda);
    m.impl("nw_affine_forward_t", nw_affine_forward_t_cuda);
    m.impl("nw_affine_value_grad_params", nw_affine_value_grad_params_cuda);
    m.impl("nw_affine_marginals_backward", nw_affine_marginals_backward_cuda);
    m.impl("nw_affine_marginals_hvp", nw_affine_marginals_hvp_cuda);
    m.impl("nw_affine_marginals_grad_gap_open", nw_affine_marginals_grad_gap_open_cuda);
    m.impl("nw_affine_marginals_grad_gap_ext", nw_affine_marginals_grad_gap_ext_cuda);
    m.impl("nw_affine_marginals_grad_temp", nw_affine_marginals_grad_temp_cuda);
}

// Register Autograd implementations
TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_nw_affine", soft_nw_affine_cuda);
    m.impl("soft_nw_affine_float", soft_nw_affine_cuda_float);
    m.impl("nw_affine_forward", nw_affine_forward_cuda);
    m.impl("nw_affine_forward_t", nw_affine_forward_t_cuda);
}

#endif // USE_TORCH_LIBRARY
