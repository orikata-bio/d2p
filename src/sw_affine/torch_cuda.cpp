// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief CUDA PyTorch bindings for Affine Gap Smith-Waterman
 *
 * Provides GPU-accelerated affine gap SW with autograd support.
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
#include <cuda_runtime.h>
#include <limits>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"
#include "common/cuda_utils.h"

// CUDA kernel declarations
#include "sw_affine/kernels.cuh"

using namespace d2p::common;

namespace {

int checked_sw_affine_size_to_int(int64_t value, const char* name) {
    TORCH_CHECK(value >= 0, name, " must be non-negative, got ", value);
    TORCH_CHECK(
        value <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        name, " exceeds CUDA int index range: ", value
    );
    return static_cast<int>(value);
}

int64_t checked_sw_affine_alpha_size(int max_L1, int max_L2) {
    TORCH_CHECK(
        max_L1 <= std::numeric_limits<int>::max() - max_L2,
        "sw_affine dimensions too large: L1 + L2 exceeds CUDA int index range"
    );

    const size_t rows = static_cast<size_t>(max_L1) + 1;
    const size_t cols = static_cast<size_t>(max_L2) + 1;
    TORCH_CHECK(
        rows <= std::numeric_limits<size_t>::max() / cols,
        "sw_affine DP table too large: (L1 + 1) * (L2 + 1) overflows size_t"
    );
    const size_t cells = rows * cols;
    TORCH_CHECK(
        cells <= std::numeric_limits<size_t>::max() / 3,
        "sw_affine DP table too large: 3 * (L1 + 1) * (L2 + 1) overflows size_t"
    );
    const size_t alpha_size = 3 * cells;
    TORCH_CHECK(
        alpha_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "sw_affine DP table too large: 3 * (L1 + 1) * (L2 + 1) exceeds CUDA int index range"
    );
    return static_cast<int64_t>(alpha_size);
}

void validate_sw_affine_lengths_cuda(
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

torch::Tensor resolve_sw_affine_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_2d(B, max_L1, max_L2, device);
    validate_sw_affine_lengths_cuda(lengths, B, max_L1, max_L2, device);
    return lengths;
}

}  // namespace

// =============================================================================
// Affine SW Autograd Function
//
// Three-state DP: Match (M), Insert (I), Delete (D)
// M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)
// I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)
// D[i,j] = LSE_T(M[i,j-1] + gap_open, D[i,j-1] + gap_ext)
// =============================================================================

class SoftSWAffineCUDAFunction : public torch::autograd::Function<SoftSWAffineCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor gap_open,
        torch::Tensor gap_ext,
        torch::Tensor temperature,
        torch::Tensor lengths  // [B, 2] actual lengths per batch (int32)
    ) {
        D2P_CHECK_INPUT_CUDA(scores);
        D2P_CUDA_GUARD(scores);
        TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
        TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
        TORCH_CHECK(gap_open.numel() == 1, "gap_open must be a scalar tensor");
        TORCH_CHECK(gap_ext.numel() == 1, "gap_ext must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
        int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
        int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
        int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

        validate_sw_affine_lengths_cuda(lengths, B, max_L1, max_L2, scores.device());

        float gap_open_val = gap_open.cpu().item<float>();
        float gap_ext_val = gap_ext.cpu().item<float>();
        float temp_val = temperature.cpu().item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_open = torch::zeros({B}, options);
        torch::Tensor grad_ext = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        // Forward pass: compute alpha and partition
        d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
        sw_affine_forward(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            partition.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            gap_open_val,
            gap_ext_val,
            temp_val
        );

        // Backward pass (of the internal DP): compute posteriors
        d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_open, &grad_ext, &grad_T, &lengths});
        sw_affine_backward(
            alpha.data_ptr<float>(),
            scores.data_ptr<float>(),
            partition.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_open.data_ptr<float>(),
            grad_ext.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            gap_open_val,
            gap_ext_val,
            temp_val
        );

        // Save for backward (HVP computation and param gradients)
        // Clone all tensors to ensure they stay valid across gc.collect()/empty_cache()
        // grad_open, grad_ext, grad_T are dS/dtheta needed for cross-derivatives d^2S/dscores/dtheta
        ctx->save_for_backward({
            scores.clone(), alpha.clone(), partition.clone(), lengths.clone(),
            grad_open.clone(), grad_ext.clone(), grad_T.clone()
        });
        ctx->saved_data["gap_open"] = gap_open_val;
        ctx->saved_data["gap_ext"] = gap_ext_val;
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
        torch::Tensor grad_open_fwd = saved[4];  // dS/dgap_open [B]
        torch::Tensor grad_ext_fwd = saved[5];   // dS/dgap_ext [B]
        torch::Tensor grad_T_fwd = saved[6];     // dS/dT [B]
        D2P_CUDA_GUARD(scores);

        float gap_open_val = static_cast<float>(ctx->saved_data["gap_open"].toDouble());
        float gap_ext_val = static_cast<float>(ctx->saved_data["gap_ext"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
        int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
        int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
        int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

        auto options = scores.options();

        // grad_outputs[0] is dL/dscore [B] (gradient w.r.t. partition function)
        // grad_outputs[1] is dL/dalignment [B, L1, L2] (gradient w.r.t. posteriors)
        torch::Tensor grad_score = grad_outputs[0];      // [B]
        torch::Tensor grad_posteriors = grad_outputs[1]; // [B, L1, L2]

        // Initialize accumulated gradients
        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_open = torch::zeros({1}, options);
        torch::Tensor total_grad_ext = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // ============ Gradient from score (partition function) ============
        // dL/dscores via score: dL/dS * dS/dscores = grad_score * posteriors
        // dL/dgap_open via score: dL/dS * dS/dgap_open = sum(grad_score * grad_open_fwd)
        // etc.
        if (grad_score.defined() && grad_score.numel() > 0) {
            // Recompute posteriors for this path (we need them for dS/dscores)
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_open = torch::zeros({B}, options);
            torch::Tensor tmp_ext = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &tmp_open, &tmp_ext, &tmp_T, &lengths});
            sw_affine_backward(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_open.data_ptr<float>(),
                tmp_ext.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_open_val, gap_ext_val, temp_val
            );

            // dS/dscores = posteriors, so dL/dscores += grad_score[:, None, None] * posteriors
            grad_scores += grad_score.view({B, 1, 1}) * posteriors;

            // dL/dgap_open += sum(grad_score * grad_open_fwd)
            total_grad_open += (grad_score * grad_open_fwd).sum().reshape({1});

            // dL/dgap_ext += sum(grad_score * grad_ext_fwd)
            total_grad_ext += (grad_score * grad_ext_fwd).sum().reshape({1});

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

            // Convert dtype if necessary (AMP may pass float16/bfloat16)
            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }

            // Ensure same device as scores
            if (grad_posteriors.device() != scores.device()) {
                grad_posteriors = grad_posteriors.to(scores.device());
            }

            // Force contiguous - this is critical for correct pointer arithmetic
            grad_posteriors = grad_posteriors.contiguous();

            // HVP: d^2S/dscores^2 * grad_posteriors = dposteriors/dscores * grad_posteriors
            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor hvp_beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_posteriors, &d_alpha, &d_partition, &hvp_beta_ws, &d_beta, &hvp_grad_scores, &lengths});
            sw_affine_hvp(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),  // tangent vector
                d_alpha.data_ptr<float>(),
                d_partition.data_ptr<float>(),
                hvp_beta_ws.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_open_val, gap_ext_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param gradients from alignment path
            torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

            // dP/dgap_open
            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_open_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            sw_affine_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_open_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_open_val, gap_ext_val, temp_val,
                0  // PARAM_GAP_OPEN
            );
            total_grad_open += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // dP/dgap_ext
            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_ext_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            sw_affine_param_grad(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_ext_fwd.data_ptr<float>(),
                U_ws.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W_ws.data_ptr<float>(),
                dP_dtheta.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                gap_open_val, gap_ext_val, temp_val,
                1  // PARAM_GAP_EXT
            );
            total_grad_ext += (grad_posteriors * dP_dtheta).sum().reshape({1});

            // dP/dT
            d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
            sw_affine_param_grad(
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
                gap_open_val, gap_ext_val, temp_val,
                2  // PARAM_TEMPERATURE
            );
            total_grad_T += (grad_posteriors * dP_dtheta).sum().reshape({1});
        }

        // Return gradients: scores, gap_open, gap_ext, temperature, lengths (no grad for lengths)
        return {grad_scores, total_grad_open, total_grad_ext, total_grad_T, torch::Tensor()};
    }
};

// =============================================================================
// Python Interface Functions
// =============================================================================

// Affine SW with autograd (tensor params for full differentiability)
// Returns (score, alignment) - both differentiable
std::vector<torch::Tensor> soft_sw_affine_cuda(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temperature,
    torch::Tensor lengths  // [B, 2] actual lengths per batch
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    return SoftSWAffineCUDAFunction::apply(scores, gap_open, gap_ext, temperature, lengths);
}

// Affine SW with float params (convenience function)
// Returns (score, alignment) - both differentiable
std::vector<torch::Tensor> soft_sw_affine_cuda_float(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
    int L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
    int L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");

    torch::Tensor gap_open_t = torch::tensor({static_cast<float>(gap_open)}, scores.options());
    torch::Tensor gap_ext_t = torch::tensor({static_cast<float>(gap_ext)}, scores.options());
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, scores.options());
    torch::Tensor lengths = resolve_sw_affine_lengths_cuda(
        lengths_opt, B, L1, L2, scores.device()
    );

    return SoftSWAffineCUDAFunction::apply(scores, gap_open_t, gap_ext_t, temp_t, lengths);
}

// Affine SW with explicit gradients
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_affine_cuda_with_grads(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");

    int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
    int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
    int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
    int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open = torch::zeros({B}, options);
    torch::Tensor grad_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_affine_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_open, &grad_ext, &grad_T, &lengths});
    sw_affine_backward(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_open.data_ptr<float>(),
        grad_ext.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    return std::make_tuple(partition, posteriors, grad_open, grad_ext, grad_T);
}

// Affine SW HVP
torch::Tensor soft_sw_affine_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor tangent,
    double gap_open,
    double gap_ext,
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
        "tangent must be on same device as scores, got ",
        tangent.device(), " vs ", scores.device()
    );

    int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
    int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
    int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
    int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor hvp_beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_affine_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &tangent, &d_alpha, &d_partition, &hvp_beta_ws, &d_beta, &H_scores, &lengths});
    sw_affine_hvp(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_partition.data_ptr<float>(),
        hvp_beta_ws.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    return H_scores;
}

// Affine SW param Jacobian: dP/dtheta where P = posteriors, theta in {gap_open, gap_ext, T}
// For testing/debugging - returns the full Jacobian matrix
torch::Tensor soft_sw_affine_param_jacobian_cuda(
    torch::Tensor scores,
    int64_t param_type,  // 0=gap_open, 1=gap_ext, 2=temperature
    double gap_open,
    double gap_ext,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    D2P_CHECK_INPUT_CUDA(scores);
    D2P_CUDA_GUARD(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D");
    TORCH_CHECK(param_type >= 0 && param_type <= 2, "param_type must be 0, 1, or 2");

    int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
    int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
    int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
    int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    // Allocate buffers
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open = torch::zeros({B}, options);
    torch::Tensor grad_ext = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    // Forward pass
    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_affine_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        partition.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    // Backward pass to get grad_open/ext/T
    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta, &posteriors, &grad_open, &grad_ext, &grad_T, &lengths});
    sw_affine_backward(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_open.data_ptr<float>(),
        grad_ext.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    // Select the appropriate dS/dtheta based on param_type
    torch::Tensor dS_dtheta;
    switch (param_type) {
        case 0: dS_dtheta = grad_open; break;
        case 1: dS_dtheta = grad_ext; break;
        case 2: dS_dtheta = grad_T; break;
    }

    // Allocate workspaces for param grad computation
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    // Compute dP/dtheta
    d2p::common::record_streams_current({&alpha, &scores, &partition, &dS_dtheta, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_affine_param_grad(
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
        static_cast<float>(gap_open),
        static_cast<float>(gap_ext),
        static_cast<float>(temperature),
        param_type
    );

    return dP_dtheta;
}

// Full backward for affine SW - returns all gradients (scores, gap_open, gap_ext, temperature)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_sw_affine_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_alignment,
    double gap_open,
    double gap_ext,
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

    int B = checked_sw_affine_size_to_int(scores.size(0), "scores.size(0)");
    int max_L1 = checked_sw_affine_size_to_int(scores.size(1), "scores.size(1)");
    int max_L2 = checked_sw_affine_size_to_int(scores.size(2), "scores.size(2)");
    int64_t alpha_size = checked_sw_affine_alpha_size(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_sw_affine_lengths_cuda(
        lengths_opt, B, max_L1, max_L2, options.device()
    );

    // Ensure grad_alignment is contiguous float32
    grad_alignment = grad_alignment.contiguous();
    if (grad_alignment.dtype() != torch::kFloat32) {
        grad_alignment = grad_alignment.to(torch::kFloat32);
    }

    // ========== Forward pass (needed for alpha, partition, dS/dtheta) ==========
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_open_fwd = torch::zeros({B}, options);
    torch::Tensor grad_ext_fwd = torch::zeros({B}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &partition, &lengths});
    sw_affine_forward(
        scores.data_ptr<float>(), alpha.data_ptr<float>(),
        partition.data_ptr<float>(), lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    d2p::common::record_streams_current({&alpha, &scores, &partition, &beta_fwd, &posteriors, &grad_open_fwd, &grad_ext_fwd, &grad_T_fwd, &lengths});
    sw_affine_backward(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), beta_fwd.data_ptr<float>(),
        posteriors.data_ptr<float>(), grad_open_fwd.data_ptr<float>(),
        grad_ext_fwd.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    // ========== HVP for grad_scores ==========
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor hvp_beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_alignment, &d_alpha, &d_partition, &hvp_beta_ws, &d_beta, &grad_scores, &lengths});
    sw_affine_hvp(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_alignment.data_ptr<float>(),
        d_alpha.data_ptr<float>(), d_partition.data_ptr<float>(),
        hvp_beta_ws.data_ptr<float>(), d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature)
    );

    // ========== Param grads ==========
    torch::Tensor U_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ws = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dtheta = torch::zeros({B, max_L1, max_L2}, options);

    // grad_gap_open (param_type = 0)
    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_open_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_open_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature), 0
    );
    torch::Tensor total_grad_open = (grad_alignment * dP_dtheta).sum().reshape({1});

    // grad_gap_ext (param_type = 1)
    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_ext_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_ext_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature), 1
    );
    torch::Tensor total_grad_ext = (grad_alignment * dP_dtheta).sum().reshape({1});

    // grad_temperature (param_type = 2)
    d2p::common::record_streams_current({&alpha, &scores, &partition, &grad_T_fwd, &U_ws, &beta_ws, &W_ws, &dP_dtheta, &lengths});
    sw_affine_param_grad(
        alpha.data_ptr<float>(), scores.data_ptr<float>(),
        partition.data_ptr<float>(), grad_T_fwd.data_ptr<float>(),
        U_ws.data_ptr<float>(), beta_ws.data_ptr<float>(),
        W_ws.data_ptr<float>(), dP_dtheta.data_ptr<float>(),
        lengths.data_ptr<int>(), B, max_L1, max_L2,
        static_cast<float>(gap_open), static_cast<float>(gap_ext),
        static_cast<float>(temperature), 2
    );
    torch::Tensor total_grad_T = (grad_alignment * dP_dtheta).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_open, total_grad_ext, total_grad_T);
}

// =============================================================================
// Namespaced API (sw_affine::*)
// These are thin wrappers around existing functions with cleaner names
// =============================================================================

// sw_affine::forward - returns (value, marginals)
std::vector<torch::Tensor> d2p_sw_affine_forward_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_cuda_float(scores, gap_open, gap_ext, temp, lengths);
}

// sw_affine::forward_t - tensor params version
std::vector<torch::Tensor> d2p_sw_affine_forward_t_cuda(
    torch::Tensor scores,
    torch::Tensor gap_open,
    torch::Tensor gap_ext,
    torch::Tensor temp,
    torch::Tensor lengths
) {
    return soft_sw_affine_cuda(scores, gap_open, gap_ext, temp, lengths);
}

// sw_affine::value_grad_params - returns (grad_gap_open, grad_gap_ext, grad_temp) per batch
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> d2p_sw_affine_value_grad_params_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    auto result = soft_sw_affine_cuda_with_grads(scores, gap_open, gap_ext, temp, lengths);
    return std::make_tuple(std::get<2>(result), std::get<3>(result), std::get<4>(result));
}

// sw_affine::marginals_backward - full backward through marginals
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> d2p_sw_affine_marginals_backward_cuda(
    torch::Tensor scores,
    torch::Tensor grad_marginals,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_backward_full_cuda(scores, grad_marginals, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_hvp - Hessian-vector product
torch::Tensor d2p_sw_affine_marginals_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor v,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_hvp_cuda(scores, v, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_gap_open - d(marginals)/d(gap_open)
torch::Tensor d2p_sw_affine_marginals_grad_gap_open_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cuda(scores, 0, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_gap_ext - d(marginals)/d(gap_ext)
torch::Tensor d2p_sw_affine_marginals_grad_gap_ext_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cuda(scores, 1, gap_open, gap_ext, temp, lengths);
}

// sw_affine::marginals_grad_temp - d(marginals)/d(temperature)
torch::Tensor d2p_sw_affine_marginals_grad_temp_cuda(
    torch::Tensor scores,
    double gap_open,
    double gap_ext,
    double temp,
    c10::optional<torch::Tensor> lengths
) {
    return soft_sw_affine_param_jacobian_cuda(scores, 2, gap_open, gap_ext, temp, lengths);
}

// =============================================================================
// Module Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

// Register CUDA implementations for affine SW operators
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    // Core affine SW operators
    m.impl("soft_sw_affine", soft_sw_affine_cuda);
    m.impl("soft_sw_affine_float", soft_sw_affine_cuda_float);
    m.impl("soft_sw_affine_with_grads", soft_sw_affine_cuda_with_grads);
    m.impl("soft_sw_affine_hvp", soft_sw_affine_hvp_cuda);
    m.impl("soft_sw_affine_param_jacobian", soft_sw_affine_param_jacobian_cuda);
    m.impl("soft_sw_affine_backward_full", soft_sw_affine_backward_full_cuda);

    // sw_affine namespace
    m.impl("sw_affine_forward", d2p_sw_affine_forward_cuda);
    m.impl("sw_affine_forward_t", d2p_sw_affine_forward_t_cuda);
    m.impl("sw_affine_value_grad_params", d2p_sw_affine_value_grad_params_cuda);
    m.impl("sw_affine_marginals_backward", d2p_sw_affine_marginals_backward_cuda);
    m.impl("sw_affine_marginals_hvp", d2p_sw_affine_marginals_hvp_cuda);
    m.impl("sw_affine_marginals_grad_gap_open", d2p_sw_affine_marginals_grad_gap_open_cuda);
    m.impl("sw_affine_marginals_grad_gap_ext", d2p_sw_affine_marginals_grad_gap_ext_cuda);
    m.impl("sw_affine_marginals_grad_temp", d2p_sw_affine_marginals_grad_temp_cuda);
}

// Register Autograd implementations for affine SW
TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    // Affine SW with tensor params uses autograd function internally
    m.impl("soft_sw_affine", soft_sw_affine_cuda);
    m.impl("soft_sw_affine_float", soft_sw_affine_cuda_float);

    // sw_affine namespace - autograd versions
    m.impl("sw_affine_forward", d2p_sw_affine_forward_cuda);
    m.impl("sw_affine_forward_t", d2p_sw_affine_forward_t_cuda);
}

#endif // USE_TORCH_LIBRARY
