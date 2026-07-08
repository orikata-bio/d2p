// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cpu.cpp
 * @brief Soft Levenshtein CPU PyTorch Bindings
 *
 * CPU implementations registered with TORCH_LIBRARY_IMPL.
 */

#include <torch/extension.h>
#include <cstdint>
#include <limits>
#include <vector>
#include "kernels_cpu.h"

namespace d2p {
namespace lev {

// ============================================================================
// Helper Macros
// ============================================================================

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

static torch::Tensor make_default_lengths_cpu(int64_t B, int64_t L1, int64_t L2);

namespace {

struct LevShapeCPU {
    int64_t B;
    int64_t max_L1;
    int64_t max_L2;
    int64_t alpha_size;
};

void validate_lev_scores_cpu(const torch::Tensor& scores) {
    CHECK_INPUT_CPU(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
}

LevShapeCPU checked_lev_shape_cpu(const torch::Tensor& scores) {
    validate_lev_scores_cpu(scores);

    const int64_t B = scores.size(0);
    const int64_t max_L1 = scores.size(1);
    const int64_t max_L2 = scores.size(2);
    constexpr int64_t kIntMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();

    TORCH_CHECK(
        B <= kIntMax && max_L1 <= kIntMax && max_L2 <= kIntMax,
        "scores dimensions must fit int32 length bounds for the CPU kernel, got ",
        scores.sizes()
    );

    const int64_t alpha_rows = max_L1 + 1;
    const int64_t alpha_cols = max_L2 + 1;
    TORCH_CHECK(
        alpha_cols == 0 || alpha_rows <= kInt64Max / alpha_cols,
        "lev alpha workspace size overflow for scores shape ", scores.sizes()
    );
    const int64_t alpha_size = alpha_rows * alpha_cols;
    TORCH_CHECK(
        B == 0 || alpha_size <= kInt64Max / B,
        "lev batched alpha workspace size overflow for scores shape ", scores.sizes()
    );

    return {B, max_L1, max_L2, alpha_size};
}

void validate_lev_lengths_cpu(const torch::Tensor& lengths, int64_t B, int64_t max_L1, int64_t max_L2) {
    CHECK_INPUT_CPU(lengths);
    TORCH_CHECK(lengths.dim() == 2 && lengths.size(0) == B && lengths.size(1) == 2);
    TORCH_CHECK(lengths.dtype() == torch::kInt32, "lengths must be int32");

    auto lengths_acc = lengths.accessor<int32_t, 2>();
    for (int64_t b = 0; b < B; ++b) {
        int32_t l1 = lengths_acc[b][0];
        int32_t l2 = lengths_acc[b][1];
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

torch::Tensor resolve_lev_lengths_cpu(
    const c10::optional<torch::Tensor>& lengths_opt,
    int64_t B,
    int64_t max_L1,
    int64_t max_L2
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths_cpu(B, max_L1, max_L2);
    validate_lev_lengths_cpu(lengths, B, max_L1, max_L2);
    return lengths;
}

void validate_lev_backward_input_cpu(const torch::Tensor& scores, const torch::Tensor& grad_posteriors) {
    TORCH_CHECK(!grad_posteriors.is_cuda(), "grad_posteriors must be a CPU tensor");
    TORCH_CHECK(
        grad_posteriors.sizes() == scores.sizes(),
        "grad_posteriors must have same shape as scores"
    );
}

}  // namespace

// ============================================================================
// Helper: Create default lengths tensor
// ============================================================================

static torch::Tensor make_default_lengths_cpu(int64_t B, int64_t L1, int64_t L2) {
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto lengths = torch::empty({B, 2}, options);
    auto acc = lengths.accessor<int32_t, 2>();
    for (int64_t b = 0; b < B; b++) {
        acc[b][0] = static_cast<int32_t>(L1);
        acc[b][1] = static_cast<int32_t>(L2);
    }
    return lengths;
}

// ============================================================================
// Levenshtein CPU Autograd Function
// ============================================================================

class SoftLevenshteinCPUFunction : public torch::autograd::Function<SoftLevenshteinCPUFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor ins_cost_t,
        torch::Tensor del_cost_t,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        // r70: leave the unused posteriors output-grad undefined so a first-order
        // backward skips the zero-contribution second-order HVP/param-grad path.
        ctx->set_materialize_grads(false);

        const LevShapeCPU shape = checked_lev_shape_cpu(scores);
        TORCH_CHECK(ins_cost_t.numel() == 1, "ins_cost must be a scalar tensor");
        TORCH_CHECK(del_cost_t.numel() == 1, "del_cost must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        const int64_t B = shape.B;
        const int64_t max_L1 = shape.max_L1;
        const int64_t max_L2 = shape.max_L2;
        const int64_t alpha_size = shape.alpha_size;

        validate_lev_lengths_cpu(lengths, B, max_L1, max_L2);

        float ins_val = ins_cost_t.item<float>();
        float del_val = del_cost_t.item<float>();
        float temp_val = temperature.item<float>();

        auto options = scores.options();
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor distance = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_ins = torch::zeros({B}, options);
        torch::Tensor grad_del = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        lev_forward_cpu(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            distance.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            ins_val, del_val, temp_val
        );

        lev_backward_cpu(
            alpha.data_ptr<float>(),
            scores.data_ptr<float>(),
            distance.data_ptr<float>(),
            beta.data_ptr<float>(),
            posteriors.data_ptr<float>(),
            grad_ins.data_ptr<float>(),
            grad_del.data_ptr<float>(),
            grad_T.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            ins_val, del_val, temp_val
        );

        ctx->save_for_backward({scores.clone(), alpha.clone(), distance.clone(), lengths.clone(),
                                grad_ins.clone(), grad_del.clone(), grad_T.clone()});
        ctx->saved_data["ins_cost"] = ins_val;
        ctx->saved_data["del_cost"] = del_val;
        ctx->saved_data["temperature"] = temp_val;

        return {distance, posteriors};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        torch::Tensor scores = saved[0];
        torch::Tensor alpha = saved[1];
        torch::Tensor distance = saved[2];
        torch::Tensor lengths = saved[3];
        torch::Tensor grad_ins_fwd = saved[4];
        torch::Tensor grad_del_fwd = saved[5];
        torch::Tensor grad_T_fwd = saved[6];

        float ins_val = static_cast<float>(ctx->saved_data["ins_cost"].toDouble());
        float del_val = static_cast<float>(ctx->saved_data["del_cost"].toDouble());
        float temp_val = static_cast<float>(ctx->saved_data["temperature"].toDouble());

        const LevShapeCPU shape = checked_lev_shape_cpu(scores);
        const int64_t B = shape.B;
        const int64_t max_L1 = shape.max_L1;
        const int64_t max_L2 = shape.max_L2;
        const int64_t alpha_size = shape.alpha_size;

        auto options = scores.options();

        torch::Tensor grad_distance = grad_outputs[0];
        torch::Tensor grad_posteriors = grad_outputs[1];

        torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor total_grad_ins = torch::zeros({1}, options);
        torch::Tensor total_grad_del = torch::zeros({1}, options);
        torch::Tensor total_grad_T = torch::zeros({1}, options);

        // Gradient from distance path
        if (grad_distance.defined() && grad_distance.numel() > 0) {
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
            torch::Tensor tmp_ins = torch::zeros({B}, options);
            torch::Tensor tmp_del = torch::zeros({B}, options);
            torch::Tensor tmp_T = torch::zeros({B}, options);

            lev_backward_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                distance.data_ptr<float>(),
                beta.data_ptr<float>(),
                posteriors.data_ptr<float>(),
                tmp_ins.data_ptr<float>(),
                tmp_del.data_ptr<float>(),
                tmp_T.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                ins_val, del_val, temp_val
            );

            grad_scores += grad_distance.view({B, 1, 1}) * posteriors;
            total_grad_ins += (grad_distance * grad_ins_fwd).sum().reshape({1});
            total_grad_del += (grad_distance * grad_del_fwd).sum().reshape({1});
            total_grad_T += (grad_distance * grad_T_fwd).sum().reshape({1});
        }

        // Gradient from posteriors path (HVP)
        if (grad_posteriors.defined() && grad_posteriors.numel() > 0) {
            grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_distance = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            lev_hvp_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                distance.data_ptr<float>(),
                grad_posteriors.data_ptr<float>(),
                d_alpha.data_ptr<float>(),
                d_distance.data_ptr<float>(),
                beta.data_ptr<float>(),
                d_beta.data_ptr<float>(),
                hvp_grad_scores.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                ins_val, del_val, temp_val
            );

            grad_scores += hvp_grad_scores;

            // Param grad for ins_cost
            torch::Tensor U_ins = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_ins = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_ins = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dIns = torch::zeros({B, max_L1, max_L2}, options);

            lev_param_grad_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                distance.data_ptr<float>(),
                U_ins.data_ptr<float>(),
                beta_ins.data_ptr<float>(),
                W_ins.data_ptr<float>(),
                dP_dIns.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                ins_val, del_val, temp_val,
                LEV_PARAM_INS_CPU
            );
            total_grad_ins += (grad_posteriors * dP_dIns).sum().reshape({1});

            // Param grad for del_cost
            torch::Tensor U_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dDel = torch::zeros({B, max_L1, max_L2}, options);

            lev_param_grad_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                distance.data_ptr<float>(),
                U_del.data_ptr<float>(),
                beta_del.data_ptr<float>(),
                W_del.data_ptr<float>(),
                dP_dDel.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                ins_val, del_val, temp_val,
                LEV_PARAM_DEL_CPU
            );
            total_grad_del += (grad_posteriors * dP_dDel).sum().reshape({1});

            // Param grad for temperature
            torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

            lev_param_grad_cpu(
                alpha.data_ptr<float>(),
                scores.data_ptr<float>(),
                distance.data_ptr<float>(),
                U_T.data_ptr<float>(),
                beta_T.data_ptr<float>(),
                W_T.data_ptr<float>(),
                dP_dT.data_ptr<float>(),
                lengths.data_ptr<int>(),
                B, max_L1, max_L2,
                ins_val, del_val, temp_val,
                LEV_PARAM_TEMPERATURE_CPU
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        return {grad_scores, total_grad_ins, total_grad_del, total_grad_T, torch::Tensor()};
    }
};

// ============================================================================
// Python Interface Functions (CPU)
// ============================================================================

static std::vector<torch::Tensor> soft_levenshtein_cpu(
    torch::Tensor scores,
    torch::Tensor ins_cost,
    torch::Tensor del_cost,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    return SoftLevenshteinCPUFunction::apply(scores, ins_cost, del_cost, temperature, lengths);
}

static std::vector<torch::Tensor> soft_levenshtein_cpu_float(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const LevShapeCPU shape = checked_lev_shape_cpu(scores);
    const int64_t B = shape.B;
    const int64_t L1 = shape.max_L1;
    const int64_t L2 = shape.max_L2;

    auto options = scores.options();
    torch::Tensor ins_t = torch::tensor({static_cast<float>(ins_cost)}, options);
    torch::Tensor del_t = torch::tensor({static_cast<float>(del_cost)}, options);
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    torch::Tensor lengths = resolve_lev_lengths_cpu(lengths_opt, B, L1, L2);

    return SoftLevenshteinCPUFunction::apply(scores, ins_t, del_t, temp_t, lengths);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_levenshtein_cpu_with_grads(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const LevShapeCPU shape = checked_lev_shape_cpu(scores);
    const int64_t B = shape.B;
    const int64_t max_L1 = shape.max_L1;
    const int64_t max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
    torch::Tensor grad_ins = torch::zeros({B}, options);
    torch::Tensor grad_del = torch::zeros({B}, options);
    torch::Tensor grad_T = torch::zeros({B}, options);

    lev_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    lev_backward_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        beta.data_ptr<float>(),
        posteriors.data_ptr<float>(),
        grad_ins.data_ptr<float>(),
        grad_del.data_ptr<float>(),
        grad_T.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    return std::make_tuple(distance, posteriors, grad_ins, grad_del, grad_T);
}

static torch::Tensor soft_levenshtein_hvp_cpu(
    torch::Tensor scores,
    torch::Tensor tangent,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const LevShapeCPU shape = checked_lev_shape_cpu(scores);
    CHECK_INPUT_CPU(tangent);
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");

    const int64_t B = shape.B;
    const int64_t max_L1 = shape.max_L1;
    const int64_t max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_distance = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor H_scores = torch::zeros({B, max_L1, max_L2}, options);

    lev_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    lev_hvp_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        tangent.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_distance.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        H_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    return H_scores;
}

static torch::Tensor soft_levenshtein_param_jacobian_cpu(
    torch::Tensor scores,
    int64_t param_type,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const LevShapeCPU shape = checked_lev_shape_cpu(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 2, "param_type must be 0 (ins), 1 (del), or 2 (temperature)");

    const int64_t B = shape.B;
    const int64_t max_L1 = shape.max_L1;
    const int64_t max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dparam = torch::zeros({B, max_L1, max_L2}, options);

    lev_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    lev_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        U.data_ptr<float>(),
        beta.data_ptr<float>(),
        W.data_ptr<float>(),
        dP_dparam.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val,
        static_cast<int>(param_type)
    );

    return dP_dparam;
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_levenshtein_backward_full_cpu(
    torch::Tensor scores,
    torch::Tensor grad_posteriors,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    const LevShapeCPU shape = checked_lev_shape_cpu(scores);
    validate_lev_backward_input_cpu(scores, grad_posteriors);

    const int64_t B = shape.B;
    const int64_t max_L1 = shape.max_L1;
    const int64_t max_L2 = shape.max_L2;
    const int64_t alpha_size = shape.alpha_size;

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cpu(lengths_opt, B, max_L1, max_L2);

    grad_posteriors = grad_posteriors.contiguous().to(torch::kFloat32);

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    // Forward pass
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);

    lev_forward_cpu(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    // HVP for grad_scores
    torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_distance = torch::zeros({B}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor grad_scores = torch::zeros({B, max_L1, max_L2}, options);

    lev_hvp_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        grad_posteriors.data_ptr<float>(),
        d_alpha.data_ptr<float>(),
        d_distance.data_ptr<float>(),
        beta.data_ptr<float>(),
        d_beta.data_ptr<float>(),
        grad_scores.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    // Param grad for ins_cost
    torch::Tensor U_ins = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_ins = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_ins = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dIns = torch::zeros({B, max_L1, max_L2}, options);

    lev_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        U_ins.data_ptr<float>(),
        beta_ins.data_ptr<float>(),
        W_ins.data_ptr<float>(),
        dP_dIns.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val,
        LEV_PARAM_INS_CPU
    );
    torch::Tensor total_grad_ins = (grad_posteriors * dP_dIns).sum().reshape({1});

    // Param grad for del_cost
    torch::Tensor U_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dDel = torch::zeros({B, max_L1, max_L2}, options);

    lev_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        U_del.data_ptr<float>(),
        beta_del.data_ptr<float>(),
        W_del.data_ptr<float>(),
        dP_dDel.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val,
        LEV_PARAM_DEL_CPU
    );
    torch::Tensor total_grad_del = (grad_posteriors * dP_dDel).sum().reshape({1});

    // Param grad for temperature
    torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    lev_param_grad_cpu(
        alpha.data_ptr<float>(),
        scores.data_ptr<float>(),
        distance.data_ptr<float>(),
        U_T.data_ptr<float>(),
        beta_T.data_ptr<float>(),
        W_T.data_ptr<float>(),
        dP_dT.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val,
        LEV_PARAM_TEMPERATURE_CPU
    );
    torch::Tensor total_grad_T = (grad_posteriors * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_ins, total_grad_del, total_grad_T);
}

}  // namespace lev
}  // namespace d2p

// ============================================================================
// TORCH_LIBRARY_IMPL Registration for CPU
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CPU, m) {
    m.impl("soft_levenshtein", d2p::lev::soft_levenshtein_cpu);
    m.impl("soft_levenshtein_float", d2p::lev::soft_levenshtein_cpu_float);
    m.impl("soft_levenshtein_with_grads", d2p::lev::soft_levenshtein_cpu_with_grads);
    m.impl("soft_levenshtein_hvp", d2p::lev::soft_levenshtein_hvp_cpu);
    m.impl("soft_levenshtein_param_jacobian", d2p::lev::soft_levenshtein_param_jacobian_cpu);
    m.impl("soft_levenshtein_backward_full", d2p::lev::soft_levenshtein_backward_full_cpu);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) {
    m.impl("soft_levenshtein", d2p::lev::soft_levenshtein_cpu);
    m.impl("soft_levenshtein_float", d2p::lev::soft_levenshtein_cpu_float);
}

#endif // USE_TORCH_LIBRARY
