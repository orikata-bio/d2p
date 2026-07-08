// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Regular Smith-Waterman CUDA Extension with PyTorch Autograd
 *
 * Single-state DP with linear gap penalty. CUDA implementations registered
 * via TORCH_LIBRARY_IMPL for automatic dispatch.
 *
 * Recurrence:
 *   alpha[i,j] = LSE_T(
 *       alpha[i-1,j-1] + scores[i,j],   // align
 *       alpha[i-1,j] + gap,              // gap in seq2
 *       alpha[i,j-1] + gap,              // gap in seq1
 *       0                                 // start new alignment
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
#include "sw/kernels.cuh"

using namespace d2p::common;

namespace {

struct SwCudaShape {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

int checked_sw_cuda_dim(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        name, " must fit int32 CUDA kernel indexing, got ", value
    );
    return static_cast<int>(value);
}

SwCudaShape checked_sw_cuda_shape(const torch::Tensor& scores) {
    int B = checked_sw_cuda_dim(scores.size(0), "scores batch dimension");
    int max_L1 = checked_sw_cuda_dim(scores.size(1), "scores L1 dimension");
    int max_L2 = checked_sw_cuda_dim(scores.size(2), "scores L2 dimension");

    int64_t rows = static_cast<int64_t>(max_L1) + 1;
    int64_t cols = static_cast<int64_t>(max_L2) + 1;
    int64_t alpha_size = rows * cols;
    TORCH_CHECK(
        alpha_size <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "SW CUDA DP table is too large for 32-bit kernel indexing: ",
        "(L1 + 1) * (L2 + 1) = ", alpha_size,
        " exceeds ", std::numeric_limits<int>::max()
    );

    return {B, max_L1, max_L2, alpha_size};
}

void validate_sw_lengths_cuda(
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

torch::Tensor resolve_sw_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_sw_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

inline void sw_regular_forward_dispatch(
    const float* scores,
    float* alpha,
    float* partition,
    const int* lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float temperature
) {
    if (B >= 4) {
        sw_regular_forward_warp(
            scores, alpha, partition, lengths,
            B, max_L1, max_L2, gap, temperature
        );
    } else {
        sw_regular_forward(
            scores, alpha, partition, lengths,
            B, max_L1, max_L2, gap, temperature
        );
    }
}

inline void sw_regular_backward_dispatch(
    const float* alpha,
    const float* scores,
    const float* partition,
    float* beta,
    float* posteriors,
    float* grad_gap,
    float* grad_T,
    const int* lengths,
    int B,
    int max_L1,
    int max_L2,
    float gap,
    float temperature
) {
    if (B >= 4) {
        sw_regular_backward_warp(
            alpha, scores, partition,
            beta, posteriors, grad_gap, grad_T,
            lengths, B, max_L1, max_L2, gap, temperature
        );
    } else {
        sw_regular_backward(
            alpha, scores, partition,
            beta, posteriors, grad_gap, grad_T,
            lengths, B, max_L1, max_L2, gap, temperature
        );
    }
}

}  // namespace

// =============================================================================
// Regular SW Autograd Function
//
// Forward: scores -> posteriors (the "soft alignment")
// Backward: uses HVP for grad_scores, chains grad_gap/grad_T with upstream grad
//
// This mirrors traditional SW which returns an alignment, but differentiable.
// =============================================================================

class SoftSWRegularCUDAFunction : public torch::autograd::Function<SoftSWRegularCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap,
        torch::Tensor temperature,
        torch::Tensor lengths  // [B, 2] actual lengths per batch (int32)
    ) {
        ctx->set_materialize_grads(false);

        D2P_CHECK_INPUT_CUDA(scores);
        TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
        TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
        TORCH_CHECK(gap.numel() == 1, "gap must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");
        D2P_CUDA_GUARD(scores);

        SwCudaShape shape = checked_sw_cuda_shape(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        validate_sw_lengths_cuda(lengths, B, max_L1, max_L2, scores.device());

        float gap_val = gap.cpu().item<float>();
        float temp_val = temperature.cpu().item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_gap = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        // Forward pass: compute alpha and partition
        d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
        sw_regular_forward_dispatch(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            gap_val,
            temp_val
        );

        // Backward pass (of the internal DP): compute posteriors, grad_gap, grad_T
        d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
        sw_regular_backward_dispatch(
            alpha.data_ptr<float>(),
            scores.data_ptr<float>(),
            partition.data_ptr<float>(),
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
        // Clone all tensors to ensure they stay valid across gc.collect()/empty_cache()
        ctx->save_for_backward({scores.clone(), alpha.clone(), partition.clone(), lengths.clone(), grad_gap.clone(), grad_T.clone()});
        ctx->saved_data["gap"] = gap_val;
        ctx->saved_data["temperature"] = temp_val;

        // Return (score, alignment) - both differentiable
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
        torch::Tensor grad_gap_fwd = saved[4];  // dS/dgap per batch (from forward)
        torch::Tensor grad_T_fwd = saved[5];    // dS/dT per batch (from forward)

        float gap_val = static_cast<float>(ctx->saved_data["gap"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        D2P_CUDA_GUARD(scores);

        SwCudaShape shape = checked_sw_cuda_shape(scores);
        int B = shape.B;
        int max_L1 = shape.max_L1;
        int max_L2 = shape.max_L2;
        int64_t alpha_size = shape.alpha_size;

        auto options = scores.options();

        // grad_outputs[0] is dL/dscore [B] (gradient w.r.t. partition function)
        // grad_outputs[1] is dL/dalignment [B, L1, L2] (gradient w.r.t. posteriors)
        torch::Tensor grad_score = grad_outputs[0];      // [B]
        torch::Tensor grad_posteriors = grad_outputs[1]; // [B, L1, L2]

        // Initialize accumulated gradients
        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_gap = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // ============ Gradient from score (partition function) ============
        // dL/dscores via score: dL/dS * dS/dscores = grad_score * posteriors
        // dL/dgap via score: dL/dS * dS/dgap = sum(grad_score * grad_gap_fwd)
        // dL/dT via score: dL/dS * dS/dT = sum(grad_score * grad_T_fwd)
        if (grad_score.defined() && grad_score.numel() > 0) {
            // Recompute posteriors for this path (we need them for dS/dscores)
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_gap = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &tmp_gap, &tmp_T, &lengths});
            sw_regular_backward_dispatch(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_gap.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val
            );

            // dS/dscores = posteriors, so dL/dscores += grad_score[:, None, None] * posteriors
            grad_scores += grad_score.view({B, 1, 1}) * posteriors;

            // dL/dgap += sum(grad_score * grad_gap_fwd)
            total_grad_gap += (grad_score * grad_gap_fwd).sum().reshape({1});

            // dL/dT += sum(grad_score * grad_T_fwd)
            total_grad_T += (grad_score * grad_T_fwd).sum().reshape({1});
        }

        // ============ Gradient from alignment (posteriors) ============
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            // Validate and prepare grad_posteriors
            TORCH_CHECK(grad_posteriors.sizes() == scores.sizes(),
                        "grad_posteriors shape mismatch: expected ", scores.sizes(),
                        " but got ", grad_posteriors.sizes());
            TORCH_CHECK(grad_posteriors.is_cuda(),
                        "grad_posteriors must be on CUDA, got ", grad_posteriors.device());

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            if (grad_posteriors.device() != scores.device()) {
                grad_posteriors = grad_posteriors.to(scores.device());
            }
            grad_posteriors = grad_posteriors.contiguous();

            // HVP: d^2S/dscores^2 * grad_posteriors
            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_posteriors, &d_alpha, &d_partition, &beta, &d_beta, &hvp_grad_scores, &lengths});
            sw_regular_hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_partition.data_ptr<float>(),
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
            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_gap_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            sw_regular_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_gap_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val,
                0  // PARAM_GAP
            );
            total_grad_gap += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // dP/dT
            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            sw_regular_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_T_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_val, temp_val,
                1  // PARAM_TEMPERATURE
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

// Regular SW with autograd (tensor params for full differentiability)
std::vector<torch::Tensor> soft_sw_regular_cuda(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftSWRegularCUDAFunction::apply(scores, gap, temperature, lengths);
}

// Regular SW with float params (convenience function)
std::vector<torch::Tensor> soft_sw_regular_cuda_float(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    SwCudaShape shape = checked_sw_cuda_shape(scores);
    int B = shape.B;
    int L1 = shape.max_L1;
    int L2 = shape.max_L2;

    torch::Tensor gap_t = torch::tensor({static_cast<float>(gap)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_sw_lengths_cuda(lengths_opt, B, L1, L2, scores.device());

    return SoftSWRegularCUDAFunction::apply(scores, gap_t, temp_t, lengths);
}

// Regular SW with explicit gradients (for debugging/inspection)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_regular_cuda_with_grads(
    torch::Tensor scores,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    D2P_CUDA_GUARD(scores);

    SwCudaShape shape = checked_sw_cuda_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_lengths_cuda(lengths_opt, B, max_L1, max_L2, options.device());

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_regular_forward_dispatch(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
    sw_regular_backward_dispatch(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_gap.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    return std::make_tuple(partition, posteriors, grad_gap, grad_T);
}

// Regular SW HVP
torch::Tensor soft_sw_regular_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CHECK_INPUT_CUDA(tangent);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    TORCH_CHECK(
        tangent.device() == scores.device(),
        "tangent must be on same device as scores, got ", tangent.device(), " vs ", scores.device()
    );
    D2P_CUDA_GUARD(scores);

    SwCudaShape shape = checked_sw_cuda_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_lengths_cuda(lengths_opt, B, max_L1, max_L2, options.device());

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_regular_forward_dispatch(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &tangent, &d_alpha, &d_partition, &beta, &d_beta, &H_scores, &lengths});
    sw_regular_hvp(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_partition.data_ptr<float>(),
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

// Regular SW param Jacobian: dP/dtheta where P = posteriors, theta in {gap, T}
torch::Tensor soft_sw_regular_param_jacobian_cuda(
    torch::Tensor scores,
    int64_t param_type,  // 0=gap, 1=temperature
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(param_type >= 0 && param_type <= 1, "param_type must be 0 or 1");
    D2P_CUDA_GUARD(scores);

    SwCudaShape shape = checked_sw_cuda_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_lengths_cuda(lengths_opt, B, max_L1, max_L2, options.device());

    // Allocate buffers
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    // Forward pass
    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_regular_forward_dispatch(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap),
        static_cast<float>(temperature)
    );

    // Backward pass to get grad_gap/grad_T
    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_gap, &grad_T, &lengths});
    sw_regular_backward_dispatch(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
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
    d2p::common::record_streams_current({&alpha, &scores, &partition, &dS_dtheta, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_regular_param_grad(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
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

// Full backward for regular SW - returns all gradients (scores, gap, temperature)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_alignment,
    double gap,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(grad_alignment.sizes() == scores.sizes(), "grad_alignment must have same shape as scores");
    TORCH_CHECK(grad_alignment.is_cuda(), "grad_alignment must be a CUDA tensor");
    TORCH_CHECK(
        grad_alignment.device() == scores.device(),
        "grad_alignment must be on same device as scores, got ",
        grad_alignment.device(), " vs ", scores.device()
    );
    D2P_CUDA_GUARD(scores);

    SwCudaShape shape = checked_sw_cuda_shape(scores);
    int B = shape.B;
    int max_L1 = shape.max_L1;
    int max_L2 = shape.max_L2;
    int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_lengths_cuda(lengths_opt, B, max_L1, max_L2, options.device());

    // Ensure grad_alignment is contiguous float32
    grad_alignment = grad_alignment.contiguous();
    if (grad_alignment.dtype() != torch::kFloat32) {
        grad_alignment = grad_alignment.to(torch::kFloat32);
    }

    // Forward pass (needed for alpha, partition, dS/dtheta)
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_gap_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_regular_forward_dispatch(
        scores.data_ptr<float>(), alpha.data_ptr<float>(),
        partition.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(gap), static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta_fwd, &posteriors, &grad_gap_fwd, &grad_T_fwd, &lengths});
    sw_regular_backward_dispatch(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), beta_fwd.data_ptr<float>(),
        posteriors.data_ptr<float>(), grad_gap_fwd.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2, static_cast<float>(gap), static_cast<float>(temperature)
    );

    // HVP for grad_scores
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_alignment, &d_alpha, &d_partition, &beta, &d_beta, &grad_scores, &lengths});
    sw_regular_hvp(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_alignment.data_ptr<float>(),
        d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
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
    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_gap_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_regular_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_gap_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_gap = (grad_alignment * dP_dtheta).sum().reshape({1});

    // grad_temperature (param_type = 1)
    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_regular_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap), static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_gap, total_grad_T);
}

// =============================================================================
// Namespaced API Wrappers (sw_*)
// =============================================================================

// sw::forward - returns (value, marginals)
std::vector<torch::Tensor> sw_forward_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_regular_cuda_float(scores, gap, temp, lengths);
}

// sw::forward_t - tensor params version
std::vector<torch::Tensor> sw_forward_t_cuda(
    torch::Tensor scores,
    torch::Tensor gap,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_sw_regular_cuda(scores, gap, temp, lengths);
}

// sw::value_grad_params - returns (grad_gap, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor> sw_value_grad_params_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_sw_regular_cuda_with_grads(scores, gap, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result));
}

// sw::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sw_marginals_backward_cuda(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_backward_full_cuda(scores, grad_marginals, gap, temp, lengths);
}

// sw::marginals_hvp - Hessian-vector product
torch::Tensor sw_marginals_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor v,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_regular_hvp_cuda(scores, v, gap, temp, lengths);
}

// sw::marginals_grad_gap - d(marginals)/d(gap)
torch::Tensor sw_marginals_grad_gap_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_regular_param_jacobian_cuda(scores, 0, gap, temp, lengths);
}

// sw::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor sw_marginals_grad_temp_cuda(
    torch::Tensor scores,
    double gap,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_regular_param_jacobian_cuda(scores, 1, gap, temp, lengths);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

// Register CUDA implementations
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_sw", soft_sw_regular_cuda);
    m.impl("soft_sw_float", soft_sw_regular_cuda_float);
    m.impl("soft_sw_with_grads", soft_sw_regular_cuda_with_grads);
    m.impl("soft_sw_hvp", soft_sw_regular_hvp_cuda);
    m.impl("soft_sw_param_jacobian", soft_sw_regular_param_jacobian_cuda);
    m.impl("soft_sw_backward_full", soft_sw_backward_full_cuda);

    // Namespaced API
    m.impl("sw_forward", sw_forward_cuda);
    m.impl("sw_forward_t", sw_forward_t_cuda);
    m.impl("sw_value_grad_params", sw_value_grad_params_cuda);
    m.impl("sw_marginals_backward", sw_marginals_backward_cuda);
    m.impl("sw_marginals_hvp", sw_marginals_hvp_cuda);
    m.impl("sw_marginals_grad_gap", sw_marginals_grad_gap_cuda);
    m.impl("sw_marginals_grad_temp", sw_marginals_grad_temp_cuda);
}

// Register Autograd implementations
TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_sw", soft_sw_regular_cuda);
    m.impl("soft_sw_float", soft_sw_regular_cuda_float);

    // Namespaced API - autograd versions
    m.impl("sw_forward", sw_forward_cuda);
    m.impl("sw_forward_t", sw_forward_t_cuda);
}

#endif // USE_TORCH_LIBRARY
