// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft CKY CUDA Extension with PyTorch Autograd
 *
 * GPU-accelerated soft CKY parsing with:
 *   - Span-based DP (constituency parsing)
 *   - Temperature-scaled logsumexp for differentiability
 *   - Full gradient support through PyTorch autograd
 *
 * Operations:
 *   - soft_cky: Main autograd function (tensor params) -> (logZ, posteriors)
 *   - soft_cky_float: Convenience function (float params)
 *   - soft_cky_with_grads: Explicit gradients for debugging
 *   - soft_cky_hvp: Hessian-vector product
 *   - soft_cky_param_jacobian: dP/dT (posteriors derivative w.r.t. temperature)
 *   - soft_cky_backward_full: Complete backward given grad_posteriors
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Shared utilities
#include "common/torch_utils.h"
#include "common/cuda_utils.h"

// CUDA kernel declarations
#include "cky/kernels.cuh"

using namespace d2p::common;

namespace {

void validate_cky_chart_shapes_cuda(
    const torch::Tensor& merge_scores,
    const torch::Tensor& leaf_scores
) {
    TORCH_CHECK(merge_scores.dim() == 4, "merge_scores must be 4D [B, n, n, n]");
    TORCH_CHECK(leaf_scores.dim() == 2, "leaf_scores must be 2D [B, n]");

    const int64_t B = merge_scores.size(0);
    const int64_t n = merge_scores.size(1);

    TORCH_CHECK(n > 0, "soft_cky requires n > 0");
    TORCH_CHECK(
        merge_scores.size(2) == n && merge_scores.size(3) == n,
        "merge_scores must be [B, n, n, n]"
    );
    TORCH_CHECK(
        leaf_scores.size(0) == B && leaf_scores.size(1) == n,
        "leaf_scores must be [B, n]"
    );
    TORCH_CHECK(
        leaf_scores.device() == merge_scores.device(),
        "leaf_scores must be on same device as merge_scores, got ",
        leaf_scores.device(),
        " vs ",
        merge_scores.device()
    );
}

void validate_cky_temperature_tensor_cuda(
    const torch::Tensor& temperature,
    int64_t B,
    int64_t n,
    torch::Device device
) {
    TORCH_CHECK(
        temperature.device() == device,
        "temperature must be on same device as merge_scores, got ",
        temperature.device(),
        " vs ",
        device
    );
    const bool T_is_scalar = temperature.numel() == 1;
    if (!T_is_scalar) {
        TORCH_CHECK(
            temperature.dim() == 3 &&
            temperature.size(0) == B &&
            temperature.size(1) == n &&
            temperature.size(2) == n,
            "temperature must be scalar or [B, n, n]"
        );
    }
}

}  // namespace

// =============================================================================
// Soft CKY CUDA Autograd Function
//
// Forward: merge_scores, leaf_scores, temperature -> (logZ, posteriors)
// Backward: uses HVP for grad w.r.t. scores, param_grad for temperature
//
// CKY uses logsumexp (maximization), similar to SW.
// =============================================================================

class SoftCKYCUDAFunction : public torch::autograd::Function<SoftCKYCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor merge_scores,
        torch::Tensor leaf_scores,
        torch::Tensor temperature
    ) {
        D2P_CHECK_INPUT_CUDA(merge_scores);
        D2P_CHECK_INPUT_CUDA(leaf_scores);
        D2P_CHECK_INPUT_CUDA(temperature);
        TORCH_CHECK(merge_scores.dtype() == torch::kFloat32, "merge_scores must be float32");
        TORCH_CHECK(leaf_scores.dtype() == torch::kFloat32, "leaf_scores must be float32");
        TORCH_CHECK(temperature.dtype() == torch::kFloat32, "temperature must be float32");

        int B = merge_scores.size(0);
        int n = merge_scores.size(1);
        validate_cky_chart_shapes_cuda(merge_scores, leaf_scores);

        torch::Tensor temperature_contig = temperature.contiguous();
        validate_cky_temperature_tensor_cuda(temperature_contig, B, n, merge_scores.device());
        bool T_is_scalar = temperature_contig.numel() == 1;
        D2P_CUDA_GUARD(merge_scores);
        auto options = merge_scores.options();
        ctx->set_materialize_grads(false);

        // Allocate outputs
        torch::Tensor Z = torch::zeros({B, n, n}, options);
        torch::Tensor partition = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, n, n}, options);
        torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
        torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_leaf = torch::zeros({B, n}, options);
        torch::Tensor grad_T = T_is_scalar
            ? torch::zeros({B}, options)
            : torch::zeros({B, n, n}, options);

        // Forward pass (inside algorithm)
        d2p::common::record_streams_current({&merge_scores, &leaf_scores, &Z, &partition, &temperature_contig});
        cky_forward(
            merge_scores.data_ptr<float>(),
            leaf_scores.data_ptr<float>(),
            Z.data_ptr<float>(),
            partition.data_ptr<float>(),
            temperature_contig.data_ptr<float>(),
            B, n, T_is_scalar
        );

        // Backward pass of DP (outside algorithm) - computes posteriors
        d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &beta, &Pcond, &Pjoint, &grad_merge, &grad_leaf, &grad_T, &temperature_contig});
        cky_backward(
            Z.data_ptr<float>(),
            merge_scores.data_ptr<float>(),
            leaf_scores.data_ptr<float>(),
            partition.data_ptr<float>(),
            beta.data_ptr<float>(),
            Pcond.data_ptr<float>(),
            Pjoint.data_ptr<float>(),
            grad_merge.data_ptr<float>(),
            grad_leaf.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            temperature_contig.data_ptr<float>(),
            B, n, T_is_scalar
        );

        // Save for backward
        ctx->save_for_backward({merge_scores.clone(), leaf_scores.clone(), Z.clone(),
                               partition.clone(), grad_leaf.clone(), grad_T.clone(), temperature_contig.clone()});
        ctx->saved_data["T_is_scalar"] = T_is_scalar;

        // Return (logZ, posteriors) - both differentiable
        return {partition, Pjoint};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor merge_scores = saved[0];
        torch::Tensor leaf_scores = saved[1];
        torch::Tensor Z = saved[2];
        torch::Tensor partition = saved[3];
        torch::Tensor grad_leaf_fwd = saved[4];  // dZ/d(leaf) per batch
        torch::Tensor grad_T_fwd = saved[5];     // dZ/dT per batch
        torch::Tensor temperature = saved[6];
        bool T_is_scalar = ctx->saved_data["T_is_scalar"].toBool();
        D2P_CUDA_GUARD(merge_scores);
        float temp_val = T_is_scalar ? temperature.item<float>() : 0.0f;

        int B = merge_scores.size(0);
        int n = merge_scores.size(1);

        auto options = merge_scores.options();

        torch::Tensor grad_logZ = grad_outputs[0];      // [B]
        torch::Tensor grad_Pjoint = grad_outputs[1];    // [B, n, n, n]
        bool has_grad_Pjoint = grad_Pjoint.defined() && grad_Pjoint.numel() > 0;
        if (has_grad_Pjoint && !T_is_scalar) {
            has_grad_Pjoint = grad_Pjoint.abs().sum().item<float>() != 0.0f;
        }

        // Initialize accumulated gradients
        torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
        torch::Tensor grad_leaf = torch::zeros({B, n}, options);
        torch::Tensor total_grad_T = T_is_scalar
            ? torch::zeros({B}, options)
            : torch::zeros({B, n, n}, options);

        // ============ Gradient from logZ (partition function) ============
        if (grad_logZ.defined() && grad_logZ.numel() > 0) {
            // Recompute posteriors
            torch::Tensor beta = torch::zeros({B, n, n}, options);
            torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
            torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
            torch::Tensor gm = torch::zeros({B, n, n, n}, options);
            torch::Tensor gl = torch::zeros({B, n}, options);
            torch::Tensor gT = T_is_scalar
                ? torch::zeros({B}, options)
                : torch::zeros({B, n, n}, options);

            d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &beta, &Pcond, &Pjoint, &gm, &gl, &gT, &temperature});
            cky_backward(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                Pcond.data_ptr<float>(),
                Pjoint.data_ptr<float>(),
                gm.data_ptr<float>(),
                gl.data_ptr<float>(),
                gT.data_ptr<float>(),
                temperature.data_ptr<float>(),
                B, n, T_is_scalar
            );

            // dZ/d(merge) = Pjoint
            grad_merge += grad_logZ.view({B, 1, 1, 1}) * Pjoint;

            // dZ/d(leaf)
            grad_leaf += grad_logZ.view({B, 1}) * grad_leaf_fwd;

            // dL/dT via logZ path
            if (T_is_scalar) {
                total_grad_T += grad_logZ.view({B}) * grad_T_fwd;
            } else {
                total_grad_T += grad_logZ.view({B, 1, 1}) * grad_T_fwd;
            }
        }

        // ============ Gradient from Pjoint (posteriors) ============
        if (has_grad_Pjoint) {
            TORCH_CHECK(
                T_is_scalar,
                "soft_cky backward through marginals is only supported for scalar temperature; "
                "use the score output or parent CKYTape local mode for tensor temperature"
            );
            grad_Pjoint = grad_Pjoint.contiguous().to(torch::kFloat32);

            // HVP: d^2Z/d(merge)^2 * grad_Pjoint
            torch::Tensor d_Z = torch::zeros({B, n, n}, options);
            torch::Tensor d_partition = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, n, n}, options);
            torch::Tensor d_beta = torch::zeros({B, n, n}, options);
            torch::Tensor hvp_merge = torch::zeros({B, n, n, n}, options);
            torch::Tensor hvp_leaf = torch::zeros({B, n}, options);

            d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &grad_Pjoint, &d_Z, &d_partition, &beta, &d_beta, &hvp_merge, &hvp_leaf});
            cky_hvp(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                grad_Pjoint.data_ptr<float>(),
                torch::zeros({B, n}, options).data_ptr<float>(),
                d_Z.data_ptr<float>(),
                d_partition.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_merge.data_ptr<float>(),
                hvp_leaf.data_ptr<float>(),
                B, n, temp_val
            );

            grad_merge += hvp_merge;
            grad_leaf += hvp_leaf;

            // Param grad: dP/dT
            torch::Tensor U = torch::zeros({B, n, n}, options);
            torch::Tensor beta_ws = torch::zeros({B, n, n}, options);
            torch::Tensor W = torch::zeros({B, n, n}, options);
            torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
            torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

            d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &U, &beta_ws, &W, &dP_dT_merge, &dP_dT_leaf});
            cky_param_grad(
                Z.data_ptr<float>(),
                merge_scores.data_ptr<float>(),
                leaf_scores.data_ptr<float>(),
                partition.data_ptr<float>(),
                U.data_ptr<float>(),
                beta_ws.data_ptr<float>(),
                W.data_ptr<float>(),
                dP_dT_merge.data_ptr<float>(),
                dP_dT_leaf.data_ptr<float>(),
                B, n, temp_val
            );

            total_grad_T[0] += (grad_Pjoint * dP_dT_merge).sum();
        }

        torch::Tensor grad_temperature = T_is_scalar
            ? total_grad_T.sum().reshape(temperature.sizes())
            : total_grad_T;
        return {grad_merge, grad_leaf, grad_temperature};
    }
};

// =============================================================================
// Python Interface Functions
// =============================================================================

// Main autograd function with tensor params
std::vector<torch::Tensor> soft_cky_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor temperature
) {
    return SoftCKYCUDAFunction::apply(merge_scores, leaf_scores, temperature);
}

// Convenience function with float params
std::vector<torch::Tensor> soft_cky_float_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    auto options = merge_scores.options();
    auto temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    return SoftCKYCUDAFunction::apply(merge_scores, leaf_scores, temp_t);
}

// With explicit gradients (for debugging/inspection)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_cky_with_grads_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    D2P_CHECK_INPUT_CUDA(merge_scores);
    D2P_CHECK_INPUT_CUDA(leaf_scores);
    validate_cky_chart_shapes_cuda(merge_scores, leaf_scores);
    D2P_CUDA_GUARD(merge_scores);

    int B = merge_scores.size(0);
    int n = merge_scores.size(1);

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
    torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_leaf = torch::zeros({B, n}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    d2p::common::record_streams_current({&merge_scores, &leaf_scores, &Z, &partition, &temp_t});
    cky_forward(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &beta, &Pcond, &Pjoint, &grad_merge, &grad_leaf, &grad_T, &temp_t});
    cky_backward(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        Pcond.data_ptr<float>(),
        Pjoint.data_ptr<float>(),
        grad_merge.data_ptr<float>(),
        grad_leaf.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    return std::make_tuple(partition, Pjoint, grad_leaf, grad_T);
}

// HVP: Hessian-vector product
torch::Tensor soft_cky_hvp_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor v_merge,
    torch::Tensor v_leaf,
    double temperature
) {
    D2P_CHECK_INPUT_CUDA(merge_scores);
    D2P_CHECK_INPUT_CUDA(leaf_scores);
    D2P_CHECK_INPUT_CUDA(v_merge);
    D2P_CHECK_INPUT_CUDA(v_leaf);
    validate_cky_chart_shapes_cuda(merge_scores, leaf_scores);
    TORCH_CHECK(
        v_merge.device() == merge_scores.device(),
        "v_merge must be on same device as merge_scores, got ",
        v_merge.device(),
        " vs ",
        merge_scores.device()
    );
    TORCH_CHECK(
        v_leaf.device() == merge_scores.device(),
        "v_leaf must be on same device as merge_scores, got ",
        v_leaf.device(),
        " vs ",
        merge_scores.device()
    );
    TORCH_CHECK(v_merge.sizes() == merge_scores.sizes(), "v_merge must match merge_scores shape");
    TORCH_CHECK(v_leaf.sizes() == leaf_scores.sizes(), "v_leaf must match leaf_scores shape");
    D2P_CUDA_GUARD(merge_scores);

    int B = merge_scores.size(0);
    int n = merge_scores.size(1);

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::common::record_streams_current({&merge_scores, &leaf_scores, &Z, &partition, &temp_t});
    cky_forward(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    torch::Tensor d_Z = torch::zeros({B, n, n}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta = torch::zeros({B, n, n}, options);
    torch::Tensor hvp_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor hvp_leaf = torch::zeros({B, n}, options);

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &v_merge, &v_leaf, &d_Z, &d_partition, &beta, &d_beta, &hvp_merge, &hvp_leaf});
    cky_hvp(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        v_merge.data_ptr<float>(),
        v_leaf.data_ptr<float>(),
        d_Z.data_ptr<float>(),
        d_partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        hvp_merge.data_ptr<float>(),
        hvp_leaf.data_ptr<float>(),
        B, n, T
    );

    return hvp_merge;
}

// Param Jacobian: dP/dT
torch::Tensor soft_cky_param_jacobian_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    double temperature
) {
    D2P_CHECK_INPUT_CUDA(merge_scores);
    D2P_CHECK_INPUT_CUDA(leaf_scores);
    validate_cky_chart_shapes_cuda(merge_scores, leaf_scores);
    D2P_CUDA_GUARD(merge_scores);

    int B = merge_scores.size(0);
    int n = merge_scores.size(1);

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);

    d2p::common::record_streams_current({&merge_scores, &leaf_scores, &Z, &partition, &temp_t});
    cky_forward(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    torch::Tensor U = torch::zeros({B, n, n}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor W = torch::zeros({B, n, n}, options);
    torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &U, &beta, &W, &dP_dT_merge, &dP_dT_leaf});
    cky_param_grad(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT_merge.data_ptr<float>(),
        dP_dT_leaf.data_ptr<float>(),
        B, n, T
    );

    return dP_dT_merge;
}

// Full backward: returns (grad_merge, grad_leaf, grad_T) given grad_posteriors
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
soft_cky_backward_full_cuda(
    torch::Tensor merge_scores,
    torch::Tensor leaf_scores,
    torch::Tensor grad_posteriors,
    double temperature
) {
    D2P_CHECK_INPUT_CUDA(merge_scores);
    D2P_CHECK_INPUT_CUDA(leaf_scores);
    D2P_CHECK_INPUT_CUDA(grad_posteriors);
    validate_cky_chart_shapes_cuda(merge_scores, leaf_scores);
    TORCH_CHECK(
        grad_posteriors.device() == merge_scores.device(),
        "grad_posteriors must be on same device as merge_scores, got ",
        grad_posteriors.device(),
        " vs ",
        merge_scores.device()
    );
    TORCH_CHECK(
        grad_posteriors.sizes() == merge_scores.sizes(),
        "grad_posteriors must match merge_scores shape"
    );
    D2P_CUDA_GUARD(merge_scores);

    int B = merge_scores.size(0);
    int n = merge_scores.size(1);

    auto options = merge_scores.options();
    float T = static_cast<float>(temperature);
    auto temp_t = torch::full({1}, T, options);

    grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

    // Forward pass
    torch::Tensor Z = torch::zeros({B, n, n}, options);
    torch::Tensor partition = torch::zeros({B}, options);
    torch::Tensor beta_fwd = torch::zeros({B, n, n}, options);
    torch::Tensor Pcond = torch::zeros({B, n, n, n}, options);
    torch::Tensor Pjoint = torch::zeros({B, n, n, n}, options);
    torch::Tensor gm = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_leaf_fwd = torch::zeros({B, n}, options);
    torch::Tensor grad_T_fwd = torch::zeros({B}, options);

    d2p::common::record_streams_current({&merge_scores, &leaf_scores, &Z, &partition, &temp_t});
    cky_forward(
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        Z.data_ptr<float>(),
        partition.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &beta_fwd, &Pcond, &Pjoint, &gm, &grad_leaf_fwd, &grad_T_fwd, &temp_t});
    cky_backward(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        beta_fwd.data_ptr<float>(),
        Pcond.data_ptr<float>(),
        Pjoint.data_ptr<float>(),
        gm.data_ptr<float>(),
        grad_leaf_fwd.data_ptr<float>(),
        grad_T_fwd.data_ptr<float>(),
        temp_t.data_ptr<float>(),
        B, n, true
    );

    // HVP for grad_merge
    torch::Tensor d_Z = torch::zeros({B, n, n}, options);
    torch::Tensor d_partition = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, n, n}, options);
    torch::Tensor d_beta = torch::zeros({B, n, n}, options);
    torch::Tensor grad_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor grad_leaf = torch::zeros({B, n}, options);

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &grad_posteriors, &d_Z, &d_partition, &beta, &d_beta, &grad_merge, &grad_leaf});
    cky_hvp(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        grad_posteriors.data_ptr<float>(),
        torch::zeros({B, n}, options).data_ptr<float>(),
        d_Z.data_ptr<float>(),
        d_partition.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_merge.data_ptr<float>(),
        grad_leaf.data_ptr<float>(),
        B, n, T
    );

    // Param grad for temperature
    torch::Tensor U = torch::zeros({B, n, n}, options);
    torch::Tensor beta_ws = torch::zeros({B, n, n}, options);
    torch::Tensor W = torch::zeros({B, n, n}, options);
    torch::Tensor dP_dT_merge = torch::zeros({B, n, n, n}, options);
    torch::Tensor dP_dT_leaf = torch::zeros({B, n}, options);

    d2p::common::record_streams_current({&Z, &merge_scores, &leaf_scores, &partition, &U, &beta_ws, &W, &dP_dT_merge, &dP_dT_leaf});
    cky_param_grad(
        Z.data_ptr<float>(),
        merge_scores.data_ptr<float>(),
        leaf_scores.data_ptr<float>(),
        partition.data_ptr<float>(),
        U.data_ptr<float>(),
        beta_ws.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dT_merge.data_ptr<float>(),
        dP_dT_leaf.data_ptr<float>(),
        B, n, T
    );

    torch::Tensor total_grad_T = (grad_posteriors * dP_dT_merge).sum().reshape({1});

    return std::make_tuple(grad_merge, grad_leaf, total_grad_T);
}

// =============================================================================
// TORCH_LIBRARY_IMPL Registration
// =============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_cky", soft_cky_cuda);
    m.impl("soft_cky_float", soft_cky_float_cuda);
    m.impl("soft_cky_with_grads", soft_cky_with_grads_cuda);
    m.impl("soft_cky_hvp", soft_cky_hvp_cuda);
    m.impl("soft_cky_param_jacobian", soft_cky_param_jacobian_cuda);
    m.impl("soft_cky_backward_full", soft_cky_backward_full_cuda);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_cky", soft_cky_cuda);
    m.impl("soft_cky_float", soft_cky_float_cuda);
}

#endif // USE_TORCH_LIBRARY
