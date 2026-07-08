// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Needleman-Wunsch CUDA Extension with PyTorch Autograd
 *
 * Global alignment with linear gap penalty. CUDA implementations registered
 * via TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * Recurrence (3 transitions - no "start new alignment" option):
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // align
 *       alpha[i-1,j] + gap,              // gap in seq2
 *       alpha[i,j-1] + gap               // gap in seq1
 *   )
 *
 * Key differences from Smith-Waterman:
 *   - No "sky" restart (global, not local alignment)
 *   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
 *   - Score = alpha[L1,L2] at terminal (not logsumexp over all cells)
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"
#include "common/cuda_utils.h"

// CUDA kernel declarations
#include "nw/kernels.cuh"

using namespace d2p::common;

namespace {

int64_t nw_alpha_size_cuda(int max_L1, int max_L2) {
    size_t alpha_size = (static_cast<size_t>(max_L1) + 1) *
                        (static_cast<size_t>(max_L2) + 1);
    TORCH_CHECK(
        alpha_size <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "NW alpha table is too large"
    );
    return static_cast<int64_t>(alpha_size);
}

void validate_same_cuda_device_as_scores(
    const torch::Tensor& tensor,
    const char* name,
    torch::Device scores_device
) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.device() == scores_device,
        name, " must be on same device as scores, got ", tensor.device(), " vs ", scores_device
    );
}

void validate_nw_lengths_cuda(
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

torch::Tensor resolve_nw_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_nw_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

}  // namespace

// =============================================================================
// NW Autograd Function
//
// Forward: scores -> posteriors (the "soft alignment")
// Backward: uses HVP for grad_scores, chains grad_gap/grad_T with upstream grad
// =============================================================================

class SoftNWCUDAFunction : public torch::autograd::Function<SoftNWCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap,
        torch::Tensor temperature,
        torch::Tensor lengths  // [B, 2] actual lengths per batch (int32)
    ) {
        D2P_CHECK_INPUT_CUDA(scores);
        TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
        TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
        TORCH_CHECK(gap.numel() == 1, "gap must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");
        validate_same_cuda_device_as_scores(gap, "gap", scores.device());
        validate_same_cuda_device_as_scores(temperature, "temperature", scores.device());
        D2P_CUDA_GUARD(scores);
        ctx->set_materialize_grads(false);

        int B = scores.size(0);
        int max_L1 = scores.size(1);
        int max_L2 = scores.size(2);
        int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);

        validate_nw_lengths_cuda(lengths, B, max_L1, max_L2, scores.device());

        float gap_val = gap.cpu().item<float>();
        float temp_val = temperature.cpu().item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        // Forward pass: compute alpha and score
        d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
        nw_forward(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            score.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            gap_val,
            temp_val
        );

        // Backward pass (of the internal DP): compute posteriors, grad_gap, grad_T
        d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
        nw_backward(
            alpha.data_ptr<float>(),
            scores.data_ptr<float>(),
            score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_gap.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            gap_val,
            temp_val
        );

        // Save for backward (HVP computation)
        ctx->save_for_backward({scores.clone(), alpha.clone(), score.clone(), lengths.clone(),
                                grad_gap.clone(), grad_T.clone()});
        ctx->saved_data["gap"] = gap_val;
        ctx->saved_data["temperature"] = temp_val;

        // Return (score, alignment) - both differentiable
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
        torch::Tensor grad_gap_fwd = saved[4];  // dS/dgap per batch (from forward)
        torch::Tensor grad_T_fwd = saved[5];    // dS/dT per batch (from forward)

        float gap_val = static_cast<float>(ctx->saved_data["gap"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        int B = scores.size(0);
        int max_L1 = scores.size(1);
        int max_L2 = scores.size(2);
        int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);
        D2P_CUDA_GUARD(scores);

        auto options = scores.options();

        // grad_outputs[0] is dL/dscore [B] (gradient w.r.t. alignment score)
        // grad_outputs[1] is dL/dalignment [B, L1, L2] (gradient w.r.t. posteriors)
        torch::Tensor grad_score = grad_outputs[0];      // [B]
        torch::Tensor grad_posteriors = grad_outputs[1]; // [B, L1, L2]

        // Initialize accumulated gradients
        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_gap = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // ============ Gradient from score ============
        if (grad_score.defined() && grad_score.numel() > 0) {
            // Recompute posteriors for this path
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_gap = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &tmp_gap, &tmp_T, &lengths});
            nw_backward(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                score.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_gap.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val
            );

            // dS/dscores = posteriors
            grad_scores += grad_score.view({B, 1, 1}) * posteriors;

            // dL/dgap += sum(grad_score * grad_gap_fwd)
            total_grad_gap += (grad_score * grad_gap_fwd).sum().reshape({1});

            // dL/dT += sum(grad_score * grad_T_fwd)
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // ============ Gradient from alignment (posteriors) ============
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            TORCH_CHECK(grad_posteriors.sizes() == scores.sizes(),
                        "grad_posteriors shape mismatch");
            TORCH_CHECK(grad_posteriors.is_cuda(),
                        "grad_posteriors must be on CUDA");
            TORCH_CHECK(
                grad_posteriors.device() == scores.device(),
                "grad_posteriors must be on same device as scores, got ",
                grad_posteriors.device(), " vs ", scores.device()
            );

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            grad_posteriors = grad_posteriors.contiguous();

            // HVP: d^2S/dscores^2 * grad_posteriors
            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_posteriors, &d_alpha, &d_score, &beta, &d_beta, &hvp_grad_scores, &lengths});
            nw_hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param gradients from alignment path
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            // dP/dgap
            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            nw_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                score.data_ptr<float>(),
                grad_gap_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val,
                0  // NW_PARAM_GAP
            );
            total_grad_gap += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // dP/dT
            d2p::common::record_streams_current({&alpha, &scores, &score, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            nw_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                score.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val,
                1  // NW_PARAM_TEMPERATURE
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        // Return gradients: scores, gap, temperature, lengths (no grad for lengths)
        return {grad_scores, total_grad_gap, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions
// =============================================================================

// NW with autograd (tensor params for full differentiability)
std::vector<torch::Tensor> soft_nw_cuda(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    D2P_CHECK_INPUT_CUDA(scores);
    validate_same_cuda_device_as_scores(gap, "gap", scores.device());
    validate_same_cuda_device_as_scores(temperature, "temperature", scores.device());
    D2P_CUDA_GUARD(scores);
    return SoftNWCUDAFunction::apply(scores, gap, temperature, lengths);
}

// NW with float params (convenience function)
std::vector<torch::Tensor> soft_nw_cuda_float(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    int B = scores.size(0);
    int L1 = scores.size(1);
    int L2 = scores.size(2);

    torch::Tensor gap_t = torch::tensor({static_cast<float>(gap)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_nw_lengths_cuda(lengths_opt, B, L1, L2, scores.device());

    return SoftNWCUDAFunction::apply(scores, gap_t, temp_t, lengths);
}

// NW with explicit gradients (for debugging/inspection)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_cuda_with_grads(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
    nw_backward(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    return std::make_tuple(score, posteriors, grad_gap, grad_T);
}

// NW HVP
torch::Tensor soft_nw_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CHECK_INPUT_CUDA(tangent);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    TORCH_CHECK(
        tangent.device() == scores.device(),
        "tangent must be on same device as scores, got ", tangent.device(), " vs ", scores.device()
    );

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &tangent, &d_alpha, &d_score, &beta, &d_beta, &H_scores, &lengths});
    nw_hvp(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        score.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    return H_scores;
}

// NW param Jacobian: dP/dtheta where P = posteriors, theta in {gap, T}
torch::Tensor soft_nw_param_jacobian_cuda(
    torch::Tensor scores,
    int64_t param_type,  // 0=gap, 1=temperature
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(param_type >= 0 && param_type <= 1, "param_type must be 0 or 1");

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    // Allocate buffers
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    // Forward pass
    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    // Backward pass to get grad_gap/grad_T
    d2p::common::record_streams_current({&alpha, &scores, &score, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
    nw_backward(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    // Select the appropriate dS/dtheta based on param_type
    torch::Tensor dS_dtheta;
    switch (param_type) {
        case 0: dS_dtheta = grad_gap; break;
        case 1: dS_dtheta = grad_T; break;
    }

    // Allocate workspaces for param grad computation
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    // Compute dP/dtheta
    d2p::common::record_streams_current({&alpha, &scores, &score, &dS_dtheta, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_param_grad(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        score.data_ptr<float>(),
        dS_dtheta.data_ptr<float>(),
        U_ws.data_ptr<float>(),
        beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(),
        dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature),
        param_type
    );

    return dP_dtheta;
}

// Full backward for NW - returns all gradients (scores, gap, temperature)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_nw_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_alignment,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(grad_alignment.sizes() == scores.sizes(), "grad_alignment must have same shape as scores");
    TORCH_CHECK(grad_alignment.is_cuda(), "grad_alignment must be a CUDA tensor");
    TORCH_CHECK(
        grad_alignment.device() == scores.device(),
        "grad_alignment must be on same device as scores, got ",
        grad_alignment.device(), " vs ", scores.device()
    );

    int B = scores.size(0);
    int max_L1 = scores.size(1);
    int max_L2 = scores.size(2);
    int64_t alpha_size = nw_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_nw_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    // Ensure grad_alignment is contiguous float32
    grad_alignment = grad_alignment.contiguous();
    if (grad_alignment.dtype() != torch::kFloat32) {
        grad_alignment = grad_alignment.to(torch::kFloat32);
    }

    // Forward pass (needed for alpha, score, dS/dtheta)
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor score = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &score, &lengths});
    nw_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(),
        score.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(gap), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &score, &beta_fwd, &posteriors, &grad_gap_fwd, &grad_T_fwd, &lengths});
    nw_backward(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        score.data_ptr<float>(), beta_fwd.data_ptr<float>(),
        posteriors.data_ptr<float>(), grad_gap_fwd.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(gap), static_cast<float>(temperature)
    );

    // HVP for grad_scores
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_alignment, &d_alpha, &d_score, &beta, &d_beta, &grad_scores, &lengths});
    nw_hvp(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        score.data_ptr<float>(), grad_alignment.data_ptr<float>(),
        d_alpha.data_ptr<float>(), d_score.data_ptr<float>(),
        beta.data_ptr<float>(), d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(gap), static_cast<float>(temperature)
    );

    // Param grads
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    // grad_gap (param_type = 0)
    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_gap_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        score.data_ptr<float>(), grad_gap_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap = (grad_alignment * dP_dtheta).sum().reshape({1});

    // grad_temperature (param_type = 1)
    d2p::common::record_streams_current({&alpha, &scores, &score, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    nw_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        score.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
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
std::vector<torch::Tensor> nw_forward_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_cuda_float(scores, gap, temp, lengths);
}

// nw::forward_t - tensor params version
std::vector<torch::Tensor> nw_forward_t_cuda(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_nw_cuda(scores, gap, temp, lengths);
}

// nw::value_grad_params - returns (grad_gap, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor> nw_value_grad_params_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_nw_cuda_with_grads(scores, gap, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result));
}

// nw::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> nw_marginals_backward_cuda(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_backward_full_cuda(scores, grad_marginals, gap, temp, lengths);
}

// nw::marginals_hvp - Hessian-vector product
torch::Tensor nw_marginals_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor v,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_hvp_cuda(scores, v, gap, temp, lengths);
}

// nw::marginals_grad_gap - d(marginals)/d(gap)
torch::Tensor nw_marginals_grad_gap_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_param_jacobian_cuda(scores, 0, gap, temp, lengths);
}

// nw::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor nw_marginals_grad_temp_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_nw_param_jacobian_cuda(scores, 1, gap, temp, lengths);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

// Register CUDA implementations
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_nw", soft_nw_cuda);
    m.impl("soft_nw_float", soft_nw_cuda_float);
    m.impl("soft_nw_with_grads", soft_nw_cuda_with_grads);
    m.impl("soft_nw_hvp", soft_nw_hvp_cuda);
    m.impl("soft_nw_param_jacobian", soft_nw_param_jacobian_cuda);
    m.impl("soft_nw_backward_full", soft_nw_backward_full_cuda);

    // Namespaced API
    m.impl("nw_forward", nw_forward_cuda);
    m.impl("nw_forward_t", nw_forward_t_cuda);
    m.impl("nw_value_grad_params", nw_value_grad_params_cuda);
    m.impl("nw_marginals_backward", nw_marginals_backward_cuda);
    m.impl("nw_marginals_hvp", nw_marginals_hvp_cuda);
    m.impl("nw_marginals_grad_gap", nw_marginals_grad_gap_cuda);
    m.impl("nw_marginals_grad_temp", nw_marginals_grad_temp_cuda);
}

// Register Autograd implementations
TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_nw", soft_nw_cuda);
    m.impl("soft_nw_float", soft_nw_cuda_float);

    // Namespaced API - autograd versions
    m.impl("nw_forward", nw_forward_cuda);
    m.impl("nw_forward_t", nw_forward_t_cuda);
}

#endif // USE_TORCH_LIBRARY
