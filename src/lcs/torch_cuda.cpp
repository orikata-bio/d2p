// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft LCS CUDA PyTorch Bindings
 *
 * Provides torch.ops.d2p.soft_lcs* operators for CUDA tensors.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <limits>
#include <vector>
#include <tuple>

#include "kernels.cuh"
#include "common/cuda_utils.h"

// ============================================================================
// Helper Functions
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

namespace {

int64_t checked_lcs_alpha_cells_cuda(int64_t L1, int64_t L2) {
    constexpr int64_t kMaxInt = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(L1 >= 0 && L2 >= 0, "scores dimensions must be non-negative");
    TORCH_CHECK(
        L1 < kMaxInt && L2 < kMaxInt,
        "scores dimensions are too large for LCS CUDA kernel indexing"
    );

    const uint64_t rows = static_cast<uint64_t>(L1) + 1;
    const uint64_t cols = static_cast<uint64_t>(L2) + 1;
    TORCH_CHECK(
        rows <= static_cast<uint64_t>(std::numeric_limits<int>::max()) / cols,
        "LCS CUDA DP table is too large: (L1 + 1) * (L2 + 1) must fit in int32"
    );
    return static_cast<int64_t>(rows * cols);
}

void validate_lcs_scores_cuda(const torch::Tensor& scores) {
    CHECK_CUDA(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D [B, L1, L2]");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be float32");
    TORCH_CHECK(
        scores.size(0) <= std::numeric_limits<int>::max(),
        "batch size is too large for LCS CUDA kernel indexing"
    );
    checked_lcs_alpha_cells_cuda(scores.size(1), scores.size(2));
    CHECK_CONTIGUOUS(scores);
}

void validate_lcs_lengths_cuda(
    const torch::Tensor& lengths,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    CHECK_INPUT_CUDA(lengths);
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

void validate_lcs_backward_input_cuda(const torch::Tensor& scores, const torch::Tensor& grad_posteriors) {
    TORCH_CHECK(grad_posteriors.is_cuda(), "grad_posteriors must be a CUDA tensor");
    TORCH_CHECK(
        grad_posteriors.sizes() == scores.sizes(),
        "grad_posteriors must have same shape as scores"
    );
    TORCH_CHECK(
        grad_posteriors.device() == scores.device(),
        "grad_posteriors must be on same device as scores, got ",
        grad_posteriors.device(),
        " vs ",
        scores.device()
    );
}

static torch::Tensor make_lengths_tensor(
    int B, int L1, int L2, torch::Device device
) {
    auto lens = torch::empty({B, 2}, torch::dtype(torch::kInt32).device(device));
    lens.select(1, 0).fill_(L1);
    lens.select(1, 1).fill_(L2);
    return lens;
}

torch::Tensor resolve_lcs_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_lengths_tensor(B, max_L1, max_L2, device);
    validate_lcs_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

}  // namespace

// ============================================================================
// Autograd Function
// ============================================================================

class SoftLCSCUDAFunction : public torch::autograd::Function<SoftLCSCUDAFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor scores,
        torch::Tensor temperature,
        c10::optional<torch::Tensor> lengths
    ) {
        // r71: hold temperature as a differentiable Tensor (like the CPU Function
        // and the Levenshtein CUDA Function) so backward can return the temperature
        // gradient instead of dropping it via .item() detachment. set_materialize_grads
        // leaves the unused posteriors output-grad undefined so a first-order backward
        // skips the zero-contribution second-order path, matching the CPU Function.
        ctx->set_materialize_grads(false);

        validate_lcs_scores_cuda(scores);
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        const int B = scores.size(0);
        const int L1 = scores.size(1);
        const int L2 = scores.size(2);
        const float T = temperature.cpu().item<float>();

        D2P_CUDA_GUARD(scores);

        auto scores_c = scores.contiguous();
        auto lengths_t = resolve_lcs_lengths_cuda(lengths, B, L1, L2, scores.device());
        const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

        // Allocate alpha and lcs_score
        auto alpha = torch::empty({static_cast<int64_t>(B), alpha_cells}, scores.options());
        auto lcs_score = torch::empty({B}, scores.options());

        // Forward pass
        d2p::common::record_streams_current({&scores_c, &alpha, &lcs_score, &lengths_t});
        d2p::lcs::lcs_forward(
            scores_c.data_ptr<float>(),
            alpha.data_ptr<float>(),
            lcs_score.data_ptr<float>(),
            lengths_t.data_ptr<int>(),
            B, L1, L2, T
        );

        // Backward pass to get posteriors
        auto beta = torch::empty_like(alpha);
        auto posteriors = torch::zeros_like(scores_c);
        auto grad_T = torch::zeros({B}, scores.options());

        d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &beta, &posteriors, &grad_T, &lengths_t});
        d2p::lcs::lcs_backward(
            alpha.data_ptr<float>(),
            scores_c.data_ptr<float>(),
            lcs_score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths_t.data_ptr<int>(),
            B, L1, L2, T
        );

        // Save for backward
        ctx->save_for_backward({scores_c, alpha, lcs_score, lengths_t});
        ctx->saved_data["temperature"] = static_cast<double>(T);

        return {lcs_score, posteriors};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto scores = saved[0];
        auto alpha = saved[1];
        auto lcs_score = saved[2];
        auto lengths_t = saved[3];
        double temperature = ctx->saved_data["temperature"].toDouble();
        float T = static_cast<float>(temperature);

        const int B = scores.size(0);
        const int L1 = scores.size(1);
        const int L2 = scores.size(2);

        D2P_CUDA_GUARD(scores);
        const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

        auto grad_lcs_score = grad_outputs[0];
        auto grad_posteriors = grad_outputs[1];

        auto grad_scores = torch::zeros_like(scores);
        auto total_grad_T = torch::zeros({1}, scores.options());

        if (grad_lcs_score.defined() && grad_lcs_score.numel() > 0) {
            auto beta = torch::empty_like(alpha);
            auto posteriors = torch::zeros_like(scores);
            auto grad_T = torch::zeros({B}, scores.options());

            d2p::common::record_streams_current({&alpha, &scores, &lcs_score, &beta, &posteriors, &grad_T, &lengths_t});
            d2p::lcs::lcs_backward(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                grad_T.data_ptr<float>(),
                lengths_t.data_ptr<int>(),
                B, L1, L2, T
            );

            grad_scores += grad_lcs_score.view({B, 1, 1}) * posteriors;
            // r71: score -> temperature path, matching the CPU Function.
            total_grad_T += (grad_lcs_score * grad_T).sum().reshape({1});
        }

        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(
                grad_posteriors.sizes() == scores.sizes(),
                "grad_posteriors must have same shape as scores"
            );
            TORCH_CHECK(grad_posteriors.is_cuda(), "grad_posteriors must be a CUDA tensor");
            TORCH_CHECK(
                grad_posteriors.device() == scores.device(),
                "grad_posteriors must be on same device as scores, got ",
                grad_posteriors.device(),
                " vs ",
                scores.device()
            );

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            grad_posteriors = grad_posteriors.contiguous();

            auto d_alpha = torch::empty_like(alpha);
            auto d_lcs_score = torch::empty({B}, scores.options());
            auto beta = torch::empty_like(alpha);
            auto d_beta = torch::empty_like(alpha);
            auto hvp_grad_scores = torch::zeros_like(scores);

            d2p::common::record_streams_current({&alpha, &scores, &lcs_score, &grad_posteriors, &d_alpha, &d_lcs_score, &beta, &d_beta, &hvp_grad_scores, &lengths_t});
            d2p::lcs::lcs_hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_lcs_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths_t.data_ptr<int>(),
                B, L1, L2, T
            );

            grad_scores += hvp_grad_scores;

            // r71: posteriors -> temperature (second-order) path, matching the CPU
            // Function and the soft_lcs_backward_full contraction. Contracts the
            // dP/dT parameter Jacobian with the upstream posteriors gradient.
            auto U_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
            auto beta_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
            auto W_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
            auto dP_dT = torch::zeros_like(scores);

            d2p::common::record_streams_current({&alpha, &scores, &lcs_score, &U_T, &beta_T, &W_T, &dP_dT, &lengths_t});
            d2p::lcs::lcs_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                lcs_score.data_ptr<float>(),
                U_T.data_ptr<float>(),
                beta_T.data_ptr<float>(),
                W_T.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths_t.data_ptr<int>(),
                B, L1, L2, T
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        return {grad_scores, total_grad_T, torch::Tensor()};
    }
};

// ============================================================================
// Operator Implementations
// ============================================================================

std::vector<torch::Tensor> soft_lcs_cuda(
    torch::Tensor scores,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    validate_lcs_scores_cuda(scores);
    TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

    // r71: pass temperature through as a differentiable Tensor (no .item() detach)
    // so autograd returns its gradient.
    c10::optional<torch::Tensor> lengths_opt;
    if (lengths.numel() > 0) {
        lengths_opt = lengths;
    }
    auto result = SoftLCSCUDAFunction::apply(scores, temperature, lengths_opt);
    return result;
}

std::vector<torch::Tensor> soft_lcs_float_cuda(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_lcs_scores_cuda(scores);

    auto temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    auto result = SoftLCSCUDAFunction::apply(scores, temp_t, lengths);
    return result;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> soft_lcs_with_grads_cuda(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_lcs_scores_cuda(scores);

    const int B = scores.size(0);
    const int L1 = scores.size(1);
    const int L2 = scores.size(2);
    const float T = static_cast<float>(temperature);

    D2P_CUDA_GUARD(scores);

    auto scores_c = scores.contiguous();
    auto lengths_t = resolve_lcs_lengths_cuda(lengths, B, L1, L2, scores.device());
    const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

    auto alpha = torch::empty({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto lcs_score = torch::empty({B}, scores.options());

    d2p::common::record_streams_current({&scores_c, &alpha, &lcs_score, &lengths_t});
    d2p::lcs::lcs_forward(
        scores_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    auto beta = torch::empty_like(alpha);
    auto posteriors = torch::zeros_like(scores_c);
    auto grad_T = torch::zeros({B}, scores.options());

    d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &beta, &posteriors, &grad_T, &lengths_t});
    d2p::lcs::lcs_backward(
        alpha.data_ptr<float>(),
        scores_c.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    return std::make_tuple(lcs_score, posteriors, grad_T);
}

torch::Tensor soft_lcs_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor V,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_lcs_scores_cuda(scores);
    CHECK_INPUT_CUDA(V);
    TORCH_CHECK(V.dim() == 3, "tangent must be 3D [B, L1, L2]");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "tangent must be float32");
    TORCH_CHECK(scores.sizes() == V.sizes(), "scores and tangent must have same shape");
    TORCH_CHECK(
        V.device() == scores.device(),
        "tangent must be on same device as scores, got ", V.device(), " vs ", scores.device()
    );

    const int B = scores.size(0);
    const int L1 = scores.size(1);
    const int L2 = scores.size(2);
    const float T = static_cast<float>(temperature);

    D2P_CUDA_GUARD(scores);

    auto scores_c = scores.contiguous();
    auto V_c = V.contiguous();
    auto lengths_t = resolve_lcs_lengths_cuda(lengths, B, L1, L2, scores.device());
    const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

    // Forward pass
    auto alpha = torch::empty({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto lcs_score = torch::empty({B}, scores.options());

    d2p::common::record_streams_current({&scores_c, &alpha, &lcs_score, &lengths_t});
    d2p::lcs::lcs_forward(
        scores_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    // HVP
    auto d_alpha = torch::empty_like(alpha);
    auto d_lcs_score = torch::empty({B}, scores.options());
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto H_scores = torch::zeros_like(scores_c);

    d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &V_c, &d_alpha, &d_lcs_score, &beta, &d_beta, &H_scores, &lengths_t});
    d2p::lcs::lcs_hvp(
        alpha.data_ptr<float>(),
        scores_c.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        V_c.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    return H_scores;
}

torch::Tensor soft_lcs_param_jacobian_cuda(
    torch::Tensor scores,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_lcs_scores_cuda(scores);

    const int B = scores.size(0);
    const int L1 = scores.size(1);
    const int L2 = scores.size(2);
    const float T = static_cast<float>(temperature);

    D2P_CUDA_GUARD(scores);

    auto scores_c = scores.contiguous();
    auto lengths_t = resolve_lcs_lengths_cuda(lengths, B, L1, L2, scores.device());
    const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

    // Forward pass
    auto alpha = torch::empty({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto lcs_score = torch::empty({B}, scores.options());

    d2p::common::record_streams_current({&scores_c, &alpha, &lcs_score, &lengths_t});
    d2p::lcs::lcs_forward(
        scores_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    // Param grad
    auto U = torch::empty_like(alpha);
    auto beta = torch::empty_like(alpha);
    auto W = torch::empty_like(alpha);
    auto dP_dT = torch::zeros_like(scores_c);

    d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &U, &beta, &W, &dP_dT, &lengths_t});
    d2p::lcs::lcs_param_grad(
        alpha.data_ptr<float>(),
        scores_c.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    return dP_dT;
}

std::tuple<torch::Tensor, torch::Tensor> soft_lcs_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_output,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_lcs_scores_cuda(scores);
    TORCH_CHECK(grad_output.dim() == 3, "grad_posteriors must be 3D");
    validate_lcs_backward_input_cuda(scores, grad_output);

    const int B = scores.size(0);
    const int L1 = scores.size(1);
    const int L2 = scores.size(2);
    const float T = static_cast<float>(temperature);

    D2P_CUDA_GUARD(scores);

    auto scores_c = scores.contiguous();
    auto grad_c = grad_output.contiguous();
    if (grad_c.scalar_type() != torch::kFloat32) {
        grad_c = grad_c.to(torch::kFloat32);
    }
    auto lengths_t = resolve_lcs_lengths_cuda(lengths, B, L1, L2, scores.device());
    const int64_t alpha_cells = checked_lcs_alpha_cells_cuda(L1, L2);

    // Forward pass
    auto alpha = torch::empty({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto lcs_score = torch::empty({B}, scores.options());

    d2p::common::record_streams_current({&scores_c, &alpha, &lcs_score, &lengths_t});
    d2p::lcs::lcs_forward(
        scores_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    // HVP
    auto d_alpha = torch::empty_like(alpha);
    auto d_lcs_score = torch::empty({B}, scores.options());
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto grad_scores = torch::zeros_like(scores_c);

    d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &grad_c, &d_alpha, &d_lcs_score, &beta, &d_beta, &grad_scores, &lengths_t});
    d2p::lcs::lcs_hvp(
        alpha.data_ptr<float>(),
        scores_c.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        grad_c.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_lcs_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    // Compute grad_T via the explicit dP/dT surface, matching the CPU path.
    auto U_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto beta_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto W_T = torch::zeros({static_cast<int64_t>(B), alpha_cells}, scores.options());
    auto dP_dT = torch::zeros_like(scores_c);

    d2p::common::record_streams_current({&alpha, &scores_c, &lcs_score, &U_T, &beta_T, &W_T, &dP_dT, &lengths_t});
    d2p::lcs::lcs_param_grad(
        alpha.data_ptr<float>(),
        scores_c.data_ptr<float>(),
        lcs_score.data_ptr<float>(),
        U_T.data_ptr<float>(),
        beta_T.data_ptr<float>(),
        W_T.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2, T
    );

    auto total_grad_T = (grad_c * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_T);
}

// ============================================================================
// Registration
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_lcs", soft_lcs_cuda);
    m.impl("soft_lcs_float", soft_lcs_float_cuda);
    m.impl("soft_lcs_with_grads", soft_lcs_with_grads_cuda);
    m.impl("soft_lcs_hvp", soft_lcs_hvp_cuda);
    m.impl("soft_lcs_param_jacobian", soft_lcs_param_jacobian_cuda);
    m.impl("soft_lcs_backward_full", soft_lcs_backward_full_cuda);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_lcs", soft_lcs_cuda);
    m.impl("soft_lcs_float", soft_lcs_float_cuda);
}

#endif
