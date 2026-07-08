// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft OSA CPU PyTorch Bindings
 *
 * Provides torch.ops.d2p.soft_osa* operators for CPU tensors.
 */

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <limits>

#include "kernels_cpu.h"

// ============================================================================
// Helper Macros
// ============================================================================

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

// ============================================================================
// Helper Functions
// ============================================================================

static torch::Tensor make_default_lengths(int B, int L1, int L2) {
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto lengths = torch::empty({B, 2}, options);
    auto acc = lengths.accessor<int32_t, 2>();
    for (int b = 0; b < B; b++) {
        acc[b][0] = L1;
        acc[b][1] = L2;
    }
    return lengths;
}

namespace {

struct OSAInputMeta {
    int B;
    int max_L1;
    int max_L2;
    int64_t alpha_size;
};

int checked_int_dim(int64_t value, const char* name) {
    TORCH_CHECK(
        value >= 0 && value <= std::numeric_limits<int>::max(),
        name, " must fit in int32, got ", value
    );
    return static_cast<int>(value);
}

OSAInputMeta checked_osa_meta_cpu(const torch::Tensor& sub_costs) {
    int B = checked_int_dim(sub_costs.size(0), "batch size");
    int max_L1 = checked_int_dim(sub_costs.size(1), "sub_costs.size(1)");
    int max_L2 = checked_int_dim(sub_costs.size(2), "sub_costs.size(2)");

    const int64_t alpha_rows = static_cast<int64_t>(max_L1) + 1;
    const int64_t alpha_cols = static_cast<int64_t>(max_L2) + 1;
    TORCH_CHECK(
        alpha_rows <= std::numeric_limits<int64_t>::max() / alpha_cols,
        "OSA DP matrix is too large"
    );

    return {B, max_L1, max_L2, alpha_rows * alpha_cols};
}

void validate_osa_inputs_cpu(const torch::Tensor& sub_costs, const torch::Tensor& trans_mask) {
    CHECK_INPUT_CPU(sub_costs);
    CHECK_INPUT_CPU(trans_mask);
    TORCH_CHECK(sub_costs.dim() == 3, "sub_costs must be 3D (B, L1, L2)");
    TORCH_CHECK(trans_mask.dim() == 3, "trans_mask must be 3D (B, L1, L2)");
    TORCH_CHECK(sub_costs.dtype() == torch::kFloat32, "sub_costs must be float32");
    TORCH_CHECK(trans_mask.dtype() == torch::kFloat32, "trans_mask must be float32");
    TORCH_CHECK(
        trans_mask.sizes() == sub_costs.sizes(),
        "trans_mask must have same shape as sub_costs"
    );
}

void validate_osa_lengths_cpu(const torch::Tensor& lengths, int B, int max_L1, int max_L2) {
    CHECK_INPUT_CPU(lengths);
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

torch::Tensor resolve_osa_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths(B, max_L1, max_L2);
    validate_osa_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

void validate_osa_hvp_input_cpu(
    const torch::Tensor& sub_costs,
    const torch::Tensor& tangent
) {
    CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(tangent.dtype() == torch::kFloat32, "tangent must be float32");
    TORCH_CHECK(
        sub_costs.sizes() == tangent.sizes(),
        "sub_costs and tangent must have same shape"
    );
}

void validate_osa_backward_input_cpu(
    const torch::Tensor& sub_costs,
    const torch::Tensor& grad_posteriors
) {
    CHECK_INPUT_CPU(grad_posteriors);
    TORCH_CHECK(grad_posteriors.dim() == 3, "grad_posteriors must be 3D");
    TORCH_CHECK(
        grad_posteriors.sizes() == sub_costs.sizes(),
        "grad_posteriors must have same shape as sub_costs"
    );
}

}  // namespace

// ============================================================================
// Autograd Function
// ============================================================================

class SoftOSACPUFunction : public torch::autograd::Function<SoftOSACPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor sub_costs,
        torch::Tensor trans_mask,
        torch::Tensor ins_cost_t,
        torch::Tensor del_cost_t,
        torch::Tensor trans_cost_t,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP path.
        ctx->set_materialize_grads(false);

        validate_osa_inputs_cpu(sub_costs, trans_mask);
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
        int B = meta.B;
        int max_L1 = meta.max_L1;
        int max_L2 = meta.max_L2;
        int64_t alpha_size = meta.alpha_size;

        validate_osa_lengths_cpu(lengths, B, max_L1, max_L2);

        float temp_val = temperature.item<float>();
        float ins_cost = ins_cost_t.item<float>();
        float del_cost = del_cost_t.item<float>();
        float trans_cost = trans_cost_t.item<float>();

        auto options = sub_costs.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor osa_score = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);
        torch::Tensor grad_ins = torch::zeros({B}, options);
        torch::Tensor grad_del = torch::zeros({B}, options);
        torch::Tensor grad_trans = torch::zeros({B}, options);

        d2p::osa::cpu::osa_forward_cpu(
            sub_costs.data_ptr<float>(),
            trans_mask.data_ptr<float>(),
            alpha.data_ptr<float>(),
            osa_score.data_ptr<float>(),
            lengths.data_ptr<int>(),
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, temp_val
        );

        d2p::osa::cpu::osa_backward_cpu(
            alpha.data_ptr<float>(),
            sub_costs.data_ptr<float>(),
            trans_mask.data_ptr<float>(),
            osa_score.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            grad_ins.data_ptr<float>(),
            grad_del.data_ptr<float>(),
            grad_trans.data_ptr<float>(),
            lengths.data_ptr<int>(),
            ins_cost, del_cost, trans_cost,
            B, max_L1, max_L2, temp_val
        );

        ctx->save_for_backward({sub_costs.clone(), trans_mask.clone(), alpha.clone(), osa_score.clone(), lengths.clone(), grad_T.clone()});
        ctx->saved_data["temperature"] = temp_val;
        ctx->saved_data["ins_cost"] = ins_cost;
        ctx->saved_data["del_cost"] = del_cost;
        ctx->saved_data["trans_cost"] = trans_cost;

        return {osa_score, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor sub_costs = saved[0];
        torch::Tensor trans_mask = saved[1];
        torch::Tensor alpha = saved[2];
        torch::Tensor osa_score = saved[3];
        torch::Tensor lengths = saved[4];
        torch::Tensor grad_T_fwd = saved[5];

        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());
        float ins_cost = static_cast<float>(ctx->saved_data["ins_cost"].toDouble());
        float del_cost = static_cast<float>(ctx->saved_data["del_cost"].toDouble());
        float trans_cost = static_cast<float>(ctx->saved_data["trans_cost"].toDouble());

        const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
        int B = meta.B;
        int max_L1 = meta.max_L1;
        int max_L2 = meta.max_L2;
        int64_t alpha_size = meta.alpha_size;

        auto options = sub_costs.options();

        torch::Tensor grad_osa_score = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_sub_costs = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);
        torch::Tensor total_grad_ins = torch::zeros({1}, options);
        torch::Tensor total_grad_del = torch::zeros({1}, options);
        torch::Tensor total_grad_trans = torch::zeros({1}, options);

        // Gradient from osa_score path
        if (grad_osa_score.defined() && grad_osa_score.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);
            torch::Tensor tmp_ins = torch::zeros({B}, options);
            torch::Tensor tmp_del = torch::zeros({B}, options);
            torch::Tensor tmp_trans = torch::zeros({B}, options);

            d2p::osa::cpu::osa_backward_cpu(
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
                lengths.data_ptr<int>(),
                ins_cost, del_cost, trans_cost,
                B, max_L1, max_L2, temp_val
            );

            grad_sub_costs += grad_osa_score.view({B, 1, 1}) * posteriors;
            // r71: score -> {temperature, ins, del, trans} paths. osa_backward_cpu
            // already computed these per-batch score sensitivities; contract them with
            // the upstream score gradient, mirroring the Levenshtein backward so CPU
            // and CUDA return the same complete cost/temperature gradient set.
            total_grad_T += (grad_osa_score * grad_T_fwd).sum().reshape({1});
            total_grad_ins += (grad_osa_score * tmp_ins).sum().reshape({1});
            total_grad_del += (grad_osa_score * tmp_del).sum().reshape({1});
            total_grad_trans += (grad_osa_score * tmp_trans).sum().reshape({1});
        }

        // Gradient from posteriors path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            validate_osa_backward_input_cpu(sub_costs, grad_posteriors);
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_osa_score = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_sub_costs = torch::zeros({B, max_L1, max_L2}, options);

            d2p::osa::cpu::osa_hvp_cpu(
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
                lengths.data_ptr<int>(),
                ins_cost, del_cost, trans_cost,
                B, max_L1, max_L2, temp_val
            );

            grad_sub_costs += hvp_grad_sub_costs;

            // r71: posteriors -> {ins, del, trans, temperature} (second-order) paths.
            // Contract each parameter Jacobian dP/dparam with the upstream posteriors
            // gradient, mirroring the Levenshtein backward. osa_param_grad_cpu computes
            // dP/dparam for param_type 0=ins, 1=del, 2=trans, 3=temperature.
            auto param_grad_term = [&](int param_type) -> torch::Tensor {
                torch::Tensor U = torch::zeros({B, alpha_size}, options);
                torch::Tensor beta_p = torch::zeros({B, alpha_size}, options);
                torch::Tensor W = torch::zeros({B, alpha_size}, options);
                torch::Tensor dP_dparam = torch::zeros({B, max_L1, max_L2}, options);
                d2p::osa::cpu::osa_param_grad_cpu(
                    alpha.data_ptr<float>(),
                    sub_costs.data_ptr<float>(),
                    trans_mask.data_ptr<float>(),
                    osa_score.data_ptr<float>(),
                    U.data_ptr<float>(),
                    beta_p.data_ptr<float>(),
                    W.data_ptr<float>(),
                    dP_dparam.data_ptr<float>(),
                    lengths.data_ptr<int>(),
                    B, max_L1, max_L2,
                    ins_cost, del_cost, trans_cost, temp_val,
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

std::vector<torch::Tensor> soft_osa_cpu(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor ins_cost,
    torch::Tensor del_cost,
    torch::Tensor trans_cost,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftOSACPUFunction::apply(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths);
}

std::vector<torch::Tensor> soft_osa_cpu_float(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_osa_inputs_cpu(sub_costs, trans_mask);

    const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
    int B = meta.B;
    int L1 = meta.max_L1;
    int L2 = meta.max_L2;

    auto options = sub_costs.options();
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    torch::Tensor ins_t = torch::tensor({static_cast<float>(ins_cost)}, options);
    torch::Tensor del_t = torch::tensor({static_cast<float>(del_cost)}, options);
    torch::Tensor trans_t = torch::tensor({static_cast<float>(trans_cost)}, options);
    torch::Tensor lengths = resolve_osa_lengths_cpu(lengths_opt, B, L1, L2);

    return SoftOSACPUFunction::apply(sub_costs, trans_mask, ins_t, del_t, trans_t, temp_t, lengths);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_osa_cpu_with_grads(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_osa_inputs_cpu(sub_costs, trans_mask);

    const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
    int B = meta.B;
    int max_L1 = meta.max_L1;
    int max_L2 = meta.max_L2;
    int64_t alpha_size = meta.alpha_size;

    auto options = sub_costs.options();
    torch::Tensor lengths = resolve_osa_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);
    float ins = static_cast<float>(ins_cost);
    float del = static_cast<float>(del_cost);
    float trans = static_cast<float>(trans_cost);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor osa_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);
    torch::Tensor grad_ins_out = torch::zeros({B}, options);
    torch::Tensor grad_del_out = torch::zeros({B}, options);
    torch::Tensor grad_trans_out = torch::zeros({B}, options);

    d2p::osa::cpu::osa_forward_cpu(
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    d2p::osa::cpu::osa_backward_cpu(
        alpha.data_ptr<float>(),
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        grad_ins_out.data_ptr<float>(),
        grad_del_out.data_ptr<float>(),
        grad_trans_out.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    return std::make_tuple(osa_score, posteriors, grad_T, grad_ins_out, grad_del_out, grad_trans_out);
}

torch::Tensor soft_osa_hvp_cpu(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor tangent,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_osa_inputs_cpu(sub_costs, trans_mask);
    validate_osa_hvp_input_cpu(sub_costs, tangent);

    const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
    int B = meta.B;
    int max_L1 = meta.max_L1;
    int max_L2 = meta.max_L2;
    int64_t alpha_size = meta.alpha_size;

    auto options = sub_costs.options();
    torch::Tensor lengths = resolve_osa_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);
    float ins = static_cast<float>(ins_cost);
    float del = static_cast<float>(del_cost);
    float trans = static_cast<float>(trans_cost);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor osa_score = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_osa_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    d2p::osa::cpu::osa_forward_cpu(
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    d2p::osa::cpu::osa_hvp_cpu(
        alpha.data_ptr<float>(),
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    return H_scores;
}

torch::Tensor soft_osa_param_jacobian_cpu(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    int64_t param_type,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_osa_inputs_cpu(sub_costs, trans_mask);
    TORCH_CHECK(param_type >= 0 && param_type <= 3, "param_type must be 0 (ins), 1 (del), 2 (trans), or 3 (temperature)");

    const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
    int B = meta.B;
    int max_L1 = meta.max_L1;
    int max_L2 = meta.max_L2;
    int64_t alpha_size = meta.alpha_size;

    auto options = sub_costs.options();
    torch::Tensor lengths = resolve_osa_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float temp_val = static_cast<float>(temperature);
    float ins = static_cast<float>(ins_cost);
    float del = static_cast<float>(del_cost);
    float trans = static_cast<float>(trans_cost);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor osa_score = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dparam = torch::zeros({B, max_L1, max_L2}, options);

    d2p::osa::cpu::osa_forward_cpu(
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    d2p::osa::cpu::osa_param_grad_cpu(
        alpha.data_ptr<float>(),
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dparam.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins, del, trans, temp_val,
        static_cast<int>(param_type)
    );

    return dP_dparam;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_osa_backward_full_cpu(
    torch::Tensor sub_costs,
    torch::Tensor trans_mask,
    torch::Tensor grad_output,
    double ins_cost,
    double del_cost,
    double trans_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_osa_inputs_cpu(sub_costs, trans_mask);
    validate_osa_backward_input_cpu(sub_costs, grad_output);

    const OSAInputMeta meta = checked_osa_meta_cpu(sub_costs);
    int B = meta.B;
    int max_L1 = meta.max_L1;
    int max_L2 = meta.max_L2;
    int64_t alpha_size = meta.alpha_size;

    auto options = sub_costs.options();
    torch::Tensor lengths = resolve_osa_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    grad_output = grad_output.contiguous().to(torch::kFloat32);

    float temp_val = static_cast<float>(temperature);
    float ins = static_cast<float>(ins_cost);
    float del = static_cast<float>(del_cost);
    float trans = static_cast<float>(trans_cost);

    // Forward pass
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor osa_score = torch::zeros({B}, options);

    d2p::osa::cpu::osa_forward_cpu(
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        alpha.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    // HVP for grad_sub_costs
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_osa_score = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_sub_costs = torch::zeros({B, max_L1, max_L2}, options);

    d2p::osa::cpu::osa_hvp_cpu(
        alpha.data_ptr<float>(),
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_osa_score.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_sub_costs.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    // Get cost gradients from backward
    torch::Tensor beta2 = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);
    torch::Tensor grad_ins_out = torch::zeros({B}, options);
    torch::Tensor grad_del_out = torch::zeros({B}, options);
    torch::Tensor grad_trans_out = torch::zeros({B}, options);

    d2p::osa::cpu::osa_backward_cpu(
        alpha.data_ptr<float>(),
        sub_costs.data_ptr<float>(),
        trans_mask.data_ptr<float>(),
        osa_score.data_ptr<float>(),
        beta2.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        grad_ins_out.data_ptr<float>(),
        grad_del_out.data_ptr<float>(),
        grad_trans_out.data_ptr<float>(),
        lengths.data_ptr<int>(),
        ins, del, trans,
        B, max_L1, max_L2, temp_val
    );

    // Weight by grad_output
    torch::Tensor total_grad_T = (grad_T * grad_output.sum({1, 2}));
    torch::Tensor total_grad_ins = (grad_ins_out * grad_output.sum({1, 2}));
    torch::Tensor total_grad_del = (grad_del_out * grad_output.sum({1, 2}));
    torch::Tensor total_grad_trans = (grad_trans_out * grad_output.sum({1, 2}));

    return std::make_tuple(grad_sub_costs, total_grad_T, total_grad_ins, total_grad_del, total_grad_trans);
}

// ============================================================================
// Registration
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_osa", soft_osa_cpu);
    m.impl("soft_osa_float", soft_osa_cpu_float);
    m.impl("soft_osa_with_grads", soft_osa_cpu_with_grads);
    m.impl("soft_osa_hvp", soft_osa_hvp_cpu);
    m.impl("soft_osa_param_jacobian", soft_osa_param_jacobian_cpu);
    m.impl("soft_osa_backward_full", soft_osa_backward_full_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_osa", soft_osa_cpu);
    m.impl("soft_osa_float", soft_osa_cpu_float);
}

#endif // USE_TORCH_LIBRARY
