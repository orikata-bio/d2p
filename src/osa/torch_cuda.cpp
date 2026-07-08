// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft OSA CUDA PyTorch Bindings
 *
 * Provides torch.ops.d2p.soft_osa* operators for CUDA tensors.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <vector>
#include <tuple>

#include "kernels.cuh"
#include "common/cuda_utils.h"

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

constexpr int64_t kOsaMaxKernelDim = static_cast<int64_t>(std::numeric_limits<int>::max());

void validate_osa_size_bounds(const torch::Tensor& sub_costs) {
    const int64_t B = sub_costs.size(0);
    const int64_t L1 = sub_costs.size(1);
    const int64_t L2 = sub_costs.size(2);
    TORCH_CHECK(B <= kOsaMaxKernelDim, "batch size must fit int32 CUDA indexing");
    TORCH_CHECK(L1 < kOsaMaxKernelDim, "L1 must be less than INT_MAX for CUDA indexing");
    TORCH_CHECK(L2 < kOsaMaxKernelDim, "L2 must be less than INT_MAX for CUDA indexing");
    TORCH_CHECK(L1 + L2 <= kOsaMaxKernelDim, "L1 + L2 must fit int32 CUDA diagonal indexing");
}

int64_t checked_osa_alpha_size(int L1, int L2) {
    return (static_cast<int64_t>(L1) + 1) * (static_cast<int64_t>(L2) + 1);
}

void validate_osa_inputs_cuda(const torch::Tensor& sub_costs, const torch::Tensor& trans_mask) {
    TORCH_CHECK(sub_costs.dim() == 3, "sub_costs must be 3D [B, L1, L2]");
    TORCH_CHECK(trans_mask.dim() == 3, "trans_mask must be 3D [B, L1, L2]");
    TORCH_CHECK(sub_costs.is_cuda(), "sub_costs must be CUDA tensor");
    TORCH_CHECK(trans_mask.is_cuda(), "trans_mask must be CUDA tensor");
    TORCH_CHECK(sub_costs.scalar_type() == torch::kFloat32, "sub_costs must be float32");
    TORCH_CHECK(trans_mask.scalar_type() == torch::kFloat32, "trans_mask must be float32");
    TORCH_CHECK(
        trans_mask.sizes() == sub_costs.sizes(),
        "trans_mask must have same shape as sub_costs"
    );
    TORCH_CHECK(
        trans_mask.device() == sub_costs.device(),
        "trans_mask must be on same device as sub_costs, got ",
        trans_mask.device(),
        " vs ",
        sub_costs.device()
    );
    validate_osa_size_bounds(sub_costs);
}

void validate_osa_same_shape_cuda(
    const torch::Tensor& sub_costs,
    const torch::Tensor& tensor,
    const char* name
) {
    TORCH_CHECK(tensor.dim() == 3, name, " must be 3D [B, L1, L2]");
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.sizes() == sub_costs.sizes(), name, " must have same shape as sub_costs");
    TORCH_CHECK(
        tensor.device() == sub_costs.device(),
        name,
        " must be on same device as sub_costs, got ",
        tensor.device(),
        " vs ",
        sub_costs.device()
    );
}

void validate_osa_lengths_cuda(
    const torch::Tensor& lengths,
    int B,
    int max_L1,
    int max_L2,
    torch::Device device
) {
    TORCH_CHECK(lengths.is_cuda(), "lengths must be a CUDA tensor");
    TORCH_CHECK(
        lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2,
        "lengths must be [B, 2], got ",
        lengths.sizes()
    );
    TORCH_CHECK(lengths.scalar_type() == torch::kInt32, "lengths must be int32");
    TORCH_CHECK(
        lengths.device() == device,
        "lengths must be on same device as sub_costs, got ",
        lengths.device(),
        " vs ",
        device
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

torch::Tensor make_default_lengths_tensor(int B, int L1, int L2, torch::Device device) {
    auto lens = torch::empty({B, 2}, torch::dtype(torch::kInt32).device(device));
    lens.select(1, 0).fill_(L1);
    lens.select(1, 1).fill_(L2);
    return lens;
}

torch::Tensor resolve_osa_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int L1,
    int L2,
    torch::Device device
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_tensor(B, L1, L2, device);
    validate_osa_lengths_cuda(lengths, B, L1, L2, device);
    return lengths.contiguous();
}

}  // namespace

// ============================================================================
// Autograd Function
// ============================================================================

class SoftOSACUDAFunction : public torch::autograd::Function<SoftOSACUDAFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor sub_costs,
        torch::Tensor trans_mask,
        torch::Tensor ins_cost_t,
        torch::Tensor del_cost_t,
        torch::Tensor trans_cost_t,
        torch::Tensor temperature,
        c10::optional<torch::Tensor> lengths
    ) {
        // r71: hold ins/del/trans/temperature as differentiable Tensors (like the CPU
        // Function and the Levenshtein CUDA Function) so backward returns their
        // gradients instead of dropping them via .item() detachment. set_materialize_grads
        // leaves the unused posteriors output-grad undefined so a first-order backward
        // skips the zero-contribution second-order path, matching the CPU Function.
        ctx->set_materialize_grads(false);

        validate_osa_inputs_cuda(sub_costs, trans_mask);
        TORCH_CHECK(ins_cost_t.numel() == 1, "ins_cost must be a scalar tensor");
        TORCH_CHECK(del_cost_t.numel() == 1, "del_cost must be a scalar tensor");
        TORCH_CHECK(trans_cost_t.numel() == 1, "trans_cost must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        const int B = sub_costs.size(0);
        const int L1 = sub_costs.size(1);
        const int L2 = sub_costs.size(2);
        const int64_t alpha_size = checked_osa_alpha_size(L1, L2);
        const float T = temperature.cpu().item<float>();
        const float ins = ins_cost_t.cpu().item<float>();
        const float del = del_cost_t.cpu().item<float>();
        const float trans = trans_cost_t.cpu().item<float>();

        D2P_CUDA_GUARD(sub_costs);

        auto sub_costs_c = sub_costs.contiguous();
        auto trans_mask_c = trans_mask.contiguous();
        auto lengths_t = resolve_osa_lengths_cuda(lengths, B, L1, L2, sub_costs.device());

        // Allocate alpha and osa_score
        auto alpha = torch::empty({B, alpha_size}, sub_costs.options());
        auto osa_score = torch::empty({B}, sub_costs.options());

        // Forward pass
        d2p::common::record_streams_current({&sub_costs_c, &trans_mask_c, &alpha, &osa_score, &lengths_t});
        d2p::osa::osa_forward(
            sub_costs_c.data_ptr<float>(),
            trans_mask_c.data_ptr<float>(),
            alpha.data_ptr<float>(),
            osa_score.data_ptr<float>(),
            lengths_t.data_ptr<int>(),
            ins, del, trans,
            B, L1, L2, T
        );

        // Backward pass to get posteriors
        auto beta = torch::empty_like(alpha);
        auto posteriors = torch::zeros_like(sub_costs_c);
        auto grad_T = torch::zeros({B}, sub_costs.options());
        auto grad_ins = torch::zeros({B}, sub_costs.options());
        auto grad_del = torch::zeros({B}, sub_costs.options());
        auto grad_trans = torch::zeros({B}, sub_costs.options());

        d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &beta, &posteriors, &grad_T, &grad_ins, &grad_del, &grad_trans, &lengths_t});
        d2p::osa::osa_backward(
            alpha.data_ptr<float>(),
            sub_costs_c.data_ptr<float>(),
            trans_mask_c.data_ptr<float>(),
            osa_score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            grad_ins.data_ptr<float>(),
            grad_del.data_ptr<float>(),
            grad_trans.data_ptr<float>(),
            lengths_t.data_ptr<int>(),
            ins, del, trans,
            B, L1, L2, T
        );

        // Save for backward. osa_score + grad_T feed the score->{sub_costs, temperature}
        // paths; the per-batch cost score-sensitivities are recomputed in backward.
        ctx->save_for_backward({sub_costs_c, trans_mask_c, alpha, osa_score, lengths_t, grad_T});
        ctx->saved_data["temperature"] = static_cast<double>(T);
        ctx->saved_data["ins_cost"] = static_cast<double>(ins);
        ctx->saved_data["del_cost"] = static_cast<double>(del);
        ctx->saved_data["trans_cost"] = static_cast<double>(trans);

        return {osa_score, posteriors};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto sub_costs = saved[0];
        auto trans_mask = saved[1];
        auto alpha = saved[2];
        auto osa_score = saved[3];
        auto lengths_t = saved[4];
        auto grad_T_fwd = saved[5];

        double temperature = ctx->saved_data["temperature"].toDouble();
        double ins_cost = ctx->saved_data["ins_cost"].toDouble();
        double del_cost = ctx->saved_data["del_cost"].toDouble();
        double trans_cost_val = ctx->saved_data["trans_cost"].toDouble();

        float T = static_cast<float>(temperature);
        float ins = static_cast<float>(ins_cost);
        float del = static_cast<float>(del_cost);
        float trans = static_cast<float>(trans_cost_val);

        const int B = sub_costs.size(0);
        const int L1 = sub_costs.size(1);
        const int L2 = sub_costs.size(2);
        const int64_t alpha_size = checked_osa_alpha_size(L1, L2);

        D2P_CUDA_GUARD(sub_costs);

        auto grad_osa_score = grad_outputs[0];
        auto grad_posteriors = grad_outputs[1];

        auto grad_sub_costs = torch::zeros_like(sub_costs);
        auto total_grad_T = torch::zeros({1}, sub_costs.options());
        auto total_grad_ins = torch::zeros({1}, sub_costs.options());
        auto total_grad_del = torch::zeros({1}, sub_costs.options());
        auto total_grad_trans = torch::zeros({1}, sub_costs.options());

        // Gradient from osa_score path: honor grad_outputs[0] (r71). Mirrors the CPU
        // Function and the Levenshtein CUDA backward so a loss through the distance/score
        // output produces the same gradients as CPU.
        if (grad_osa_score.defined() && grad_osa_score.numel() > 0) {
            auto beta = torch::empty_like(alpha);
            auto posteriors = torch::zeros_like(sub_costs);
            auto tmp_T = torch::zeros({B}, sub_costs.options());
            auto tmp_ins = torch::zeros({B}, sub_costs.options());
            auto tmp_del = torch::zeros({B}, sub_costs.options());
            auto tmp_trans = torch::zeros({B}, sub_costs.options());

            d2p::common::record_streams_current({&alpha, &sub_costs, &trans_mask, &osa_score, &beta, &posteriors, &tmp_T, &tmp_ins, &tmp_del, &tmp_trans, &lengths_t});
            d2p::osa::osa_backward(
                alpha.data_ptr<float>(),
                sub_costs.data_ptr<float>(),
                trans_mask.data_ptr<float>(),
                osa_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                tmp_ins.data_ptr<float>(),
                tmp_del.data_ptr<float>(),
                tmp_trans.data_ptr<float>(),
                lengths_t.data_ptr<int>(),
                ins, del, trans,
                B, L1, L2, T
            );

            grad_sub_costs += grad_osa_score.view({B, 1, 1}) * posteriors;
            total_grad_T += (grad_osa_score * grad_T_fwd).sum().reshape({1});
            total_grad_ins += (grad_osa_score * tmp_ins).sum().reshape({1});
            total_grad_del += (grad_osa_score * tmp_del).sum().reshape({1});
            total_grad_trans += (grad_osa_score * tmp_trans).sum().reshape({1});
        }

        // Gradient from posteriors path (HVP for sub_costs + second-order parameter
        // Jacobians for ins/del/trans/temperature). Mirrors the CPU Function.
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            validate_osa_same_shape_cuda(sub_costs, grad_posteriors, "grad_posteriors");
            grad_posteriors = grad_posteriors.contiguous();

            auto d_alpha = torch::empty_like(alpha);
            auto d_osa_score = torch::empty({B}, sub_costs.options());
            auto beta = torch::empty_like(alpha);
            auto d_beta = torch::empty_like(alpha);
            auto hvp_grad_sub_costs = torch::zeros_like(sub_costs);

            d2p::common::record_streams_current({&alpha, &sub_costs, &trans_mask, &osa_score, &grad_posteriors, &d_alpha, &d_osa_score, &beta, &d_beta, &hvp_grad_sub_costs, &lengths_t});
            d2p::osa::osa_hvp(
                alpha.data_ptr<float>(),
                sub_costs.data_ptr<float>(),
                trans_mask.data_ptr<float>(),
                osa_score.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_osa_score.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_sub_costs.data_ptr<float>(),
                lengths_t.data_ptr<int>(),
                ins, del, trans,
                B, L1, L2, T
            );

            grad_sub_costs += hvp_grad_sub_costs;

            // posteriors -> {ins, del, trans, temperature} second-order terms. Contract
            // each parameter Jacobian dP/dparam with the upstream posteriors gradient.
            // param_type: 0=ins, 1=del, 2=trans, 3=temperature.
            auto param_grad_term = [&](int param_type) -> torch::Tensor {
                auto U = torch::zeros({B, alpha_size}, sub_costs.options());
                auto beta_p = torch::zeros({B, alpha_size}, sub_costs.options());
                auto W = torch::zeros({B, alpha_size}, sub_costs.options());
                auto dP_dparam = torch::zeros({B, L1, L2}, sub_costs.options());
                d2p::common::record_streams_current({&alpha, &sub_costs, &trans_mask, &osa_score, &U, &beta_p, &W, &dP_dparam, &lengths_t});
                d2p::osa::osa_param_grad(
                    alpha.data_ptr<float>(),
                    sub_costs.data_ptr<float>(),
                    trans_mask.data_ptr<float>(),
                    osa_score.data_ptr<float>(),
                    U.data_ptr<float>(),
                    beta_p.data_ptr<float>(),
                    W.data_ptr<float>(),
                    dP_dparam.data_ptr<float>(),
                    lengths_t.data_ptr<int>(),
                    B, L1, L2,
                    ins, del, trans, T,
                    param_type
                );
                return (grad_posteriors * dP_dparam).sum().reshape({1});
            };
            total_grad_ins += param_grad_term(0);
            total_grad_del += param_grad_term(1);
            total_grad_trans += param_grad_term(2);
            total_grad_T += param_grad_term(3);
        }

        // Return gradients: sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
        return {grad_sub_costs, torch::Tensor(), total_grad_ins, total_grad_del, total_grad_trans, total_grad_T, torch::Tensor()};
    }
};

// ============================================================================
// Operator Implementations
// ============================================================================

std::vector<torch::Tensor> soft_osa_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor ins_cost_t,
    torch::Tensor del_cost_t,
    torch::Tensor trans_cost_t,
    torch::Tensor temperature_t,
    torch::Tensor lengths
) {
    // r71: pass ins/del/trans/temperature through as differentiable Tensors (no
    // .item() detach) so autograd returns their gradients.
    c10::optional<torch::Tensor> lengths_opt;
    if (lengths.numel() > 0) {
        lengths_opt = lengths;
    }
    auto result = SoftOSACUDAFunction::apply(sub_costs, trans_mask, ins_cost_t, del_cost_t, trans_cost_t, temperature_t, lengths_opt);
    return result;
}

std::vector<torch::Tensor> soft_osa_float_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    auto options = sub_costs.options();
    auto ins_t = torch::tensor({static_cast<float>(ins_cost)}, options);
    auto del_t = torch::tensor({static_cast<float>(del_cost)}, options);
    auto trans_t = torch::tensor({static_cast<float>(trans_cost)}, options);
    auto temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    auto result = SoftOSACUDAFunction::apply(sub_costs, trans_mask, ins_t, del_t, trans_t, temp_t, lengths);
    return result;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_osa_with_grads_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_osa_inputs_cuda(sub_costs, trans_mask);

    const int B = sub_costs.size(0);
    const int L1 = sub_costs.size(1);
    const int L2 = sub_costs.size(2);
    const int64_t alpha_size = checked_osa_alpha_size(L1, L2);
    const float T = static_cast<float>(temperature);
    const float ins = static_cast<float>(ins_cost);
    const float del = static_cast<float>(del_cost);
    const float trans = static_cast<float>(trans_cost);

    D2P_CUDA_GUARD(sub_costs);

    auto sub_costs_c = sub_costs.contiguous();
    auto trans_mask_c = trans_mask.contiguous();
    auto lengths_t = resolve_osa_lengths_cuda(lengths, B, L1, L2, sub_costs.device());

    auto alpha = torch::empty({B, alpha_size}, sub_costs.options());
    auto osa_score = torch::empty({B}, sub_costs.options());

    d2p::common::record_streams_current({&sub_costs_c, &trans_mask_c, &alpha, &osa_score, &lengths_t});
    d2p::osa::osa_forward(
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    auto beta = torch::empty_like(alpha);
    auto posteriors = torch::zeros_like(sub_costs_c);
    auto grad_T = torch::zeros({B}, sub_costs.options());
    auto grad_ins_out = torch::zeros({B}, sub_costs.options());
    auto grad_del_out = torch::zeros({B}, sub_costs.options());
    auto grad_trans_out = torch::zeros({B}, sub_costs.options());

    d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &beta, &posteriors, &grad_T, &grad_ins_out, &grad_del_out, &grad_trans_out, &lengths_t});
    d2p::osa::osa_backward(
        alpha.data_ptr<float>(),
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        grad_ins_out.data_ptr<float>(),
        grad_del_out.data_ptr<float>(),
        grad_trans_out.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    return std::make_tuple(osa_score, posteriors, grad_T, grad_ins_out, grad_del_out, grad_trans_out);
}

torch::Tensor soft_osa_hvp_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor V,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_osa_inputs_cuda(sub_costs, trans_mask);
    validate_osa_same_shape_cuda(sub_costs, V, "V");

    const int B = sub_costs.size(0);
    const int L1 = sub_costs.size(1);
    const int L2 = sub_costs.size(2);
    const int64_t alpha_size = checked_osa_alpha_size(L1, L2);
    const float T = static_cast<float>(temperature);
    const float ins = static_cast<float>(ins_cost);
    const float del = static_cast<float>(del_cost);
    const float trans = static_cast<float>(trans_cost);

    D2P_CUDA_GUARD(sub_costs);

    auto sub_costs_c = sub_costs.contiguous();
    auto trans_mask_c = trans_mask.contiguous();
    auto V_c = V.contiguous();
    auto lengths_t = resolve_osa_lengths_cuda(lengths, B, L1, L2, sub_costs.device());

    // Forward pass
    auto alpha = torch::empty({B, alpha_size}, sub_costs.options());
    auto osa_score = torch::empty({B}, sub_costs.options());

    d2p::common::record_streams_current({&sub_costs_c, &trans_mask_c, &alpha, &osa_score, &lengths_t});
    d2p::osa::osa_forward(
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    // HVP
    auto d_alpha = torch::empty_like(alpha);
    auto d_osa_score = torch::empty({B}, sub_costs.options());
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto H_scores = torch::zeros_like(sub_costs_c);

    d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &V_c, &d_alpha, &d_osa_score, &beta, &d_beta, &H_scores, &lengths_t});
    d2p::osa::osa_hvp(
        alpha.data_ptr<float>(),
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        V_c.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    return H_scores;
}

torch::Tensor soft_osa_param_jacobian_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    int64_t param_type,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_osa_inputs_cuda(sub_costs, trans_mask);
    TORCH_CHECK(param_type >= 0 && param_type <= 3, "param_type must be 0 (ins), 1 (del), 2 (trans), or 3 (temperature)");

    const int B = sub_costs.size(0);
    const int L1 = sub_costs.size(1);
    const int L2 = sub_costs.size(2);
    const int64_t alpha_size = checked_osa_alpha_size(L1, L2);
    const float T = static_cast<float>(temperature);
    const float ins = static_cast<float>(ins_cost);
    const float del = static_cast<float>(del_cost);
    const float trans = static_cast<float>(trans_cost);

    D2P_CUDA_GUARD(sub_costs);

    auto sub_costs_c = sub_costs.contiguous();
    auto trans_mask_c = trans_mask.contiguous();
    auto lengths_t = resolve_osa_lengths_cuda(lengths, B, L1, L2, sub_costs.device());

    // Forward pass
    auto alpha = torch::empty({B, alpha_size}, sub_costs.options());
    auto osa_score = torch::empty({B}, sub_costs.options());

    d2p::common::record_streams_current({&sub_costs_c, &trans_mask_c, &alpha, &osa_score, &lengths_t});
    d2p::osa::osa_forward(
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    // Param grad
    auto U = torch::zeros({B, alpha_size}, sub_costs.options());
    auto beta = torch::zeros({B, alpha_size}, sub_costs.options());
    auto W = torch::zeros({B, alpha_size}, sub_costs.options());
    auto dP_dparam = torch::zeros({B, L1, L2}, sub_costs.options());

    d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &U, &beta, &W, &dP_dparam, &lengths_t});
    d2p::osa::osa_param_grad(
        alpha.data_ptr<float>(),
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dparam.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        B, L1, L2,
        ins, del, trans, T,
        static_cast<int>(param_type)
    );

    return dP_dparam;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_osa_backward_full_cuda(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor grad_output,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths
) {
    validate_osa_inputs_cuda(sub_costs, trans_mask);
    validate_osa_same_shape_cuda(sub_costs, grad_output, "grad_output");

    const int B = sub_costs.size(0);
    const int L1 = sub_costs.size(1);
    const int L2 = sub_costs.size(2);
    const int64_t alpha_size = checked_osa_alpha_size(L1, L2);
    const float T = static_cast<float>(temperature);
    const float ins = static_cast<float>(ins_cost);
    const float del = static_cast<float>(del_cost);
    const float trans = static_cast<float>(trans_cost);

    D2P_CUDA_GUARD(sub_costs);

    auto sub_costs_c = sub_costs.contiguous();
    auto trans_mask_c = trans_mask.contiguous();
    auto grad_c = grad_output.contiguous();
    auto lengths_t = resolve_osa_lengths_cuda(lengths, B, L1, L2, sub_costs.device());

    // Forward pass
    auto alpha = torch::empty({B, alpha_size}, sub_costs.options());
    auto osa_score = torch::empty({B}, sub_costs.options());

    d2p::common::record_streams_current({&sub_costs_c, &trans_mask_c, &alpha, &osa_score, &lengths_t});
    d2p::osa::osa_forward(
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    // HVP for grad_sub_costs
    auto d_alpha = torch::empty_like(alpha);
    auto d_osa_score = torch::empty({B}, sub_costs.options());
    auto beta = torch::empty_like(alpha);
    auto d_beta = torch::empty_like(alpha);
    auto grad_sub_costs = torch::zeros_like(sub_costs_c);

    d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &grad_c, &d_alpha, &d_osa_score, &beta, &d_beta, &grad_sub_costs, &lengths_t});
    d2p::osa::osa_hvp(
        alpha.data_ptr<float>(),
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        grad_c.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_sub_costs.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    // Get cost gradients from backward
    auto beta2 = torch::empty_like(alpha);
    auto posteriors = torch::zeros_like(sub_costs_c);
    auto grad_T = torch::zeros({B}, sub_costs.options());
    auto grad_ins_out = torch::zeros({B}, sub_costs.options());
    auto grad_del_out = torch::zeros({B}, sub_costs.options());
    auto grad_trans_out = torch::zeros({B}, sub_costs.options());

    d2p::common::record_streams_current({&alpha, &sub_costs_c, &trans_mask_c, &osa_score, &beta2, &posteriors, &grad_T, &grad_ins_out, &grad_del_out, &grad_trans_out, &lengths_t});
    d2p::osa::osa_backward(
        alpha.data_ptr<float>(),
        sub_costs_c.data_ptr<float>(),
        trans_mask_c.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        beta2.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        grad_ins_out.data_ptr<float>(),
        grad_del_out.data_ptr<float>(),
        grad_trans_out.data_ptr<float>(),
        lengths_t.data_ptr<int>(),
        ins, del, trans,
        B, L1, L2, T
    );

    // Weight by grad_output
    grad_T = (grad_T * grad_c.sum({1, 2}));
    grad_ins_out = (grad_ins_out * grad_c.sum({1, 2}));
    grad_del_out = (grad_del_out * grad_c.sum({1, 2}));
    grad_trans_out = (grad_trans_out * grad_c.sum({1, 2}));

    return std::make_tuple(grad_sub_costs, grad_T, grad_ins_out, grad_del_out, grad_trans_out);
}

// ============================================================================
// Registration
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_osa", soft_osa_cuda);
    m.impl("soft_osa_float", soft_osa_float_cuda);
    m.impl("soft_osa_with_grads", soft_osa_with_grads_cuda);
    m.impl("soft_osa_hvp", soft_osa_hvp_cuda);
    m.impl("soft_osa_param_jacobian", soft_osa_param_jacobian_cuda);
    m.impl("soft_osa_backward_full", soft_osa_backward_full_cuda);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_osa", soft_osa_cuda);
    m.impl("soft_osa_float", soft_osa_float_cuda);
}

#endif
