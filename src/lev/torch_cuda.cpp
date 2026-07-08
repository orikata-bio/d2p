// SPDX-License-Identifier: Apache-2.0
/**
 * @file torch_cuda.cpp
 * @brief Soft Levenshtein CUDA PyTorch Bindings
 *
 * CUDA implementations registered with TORCH_LIBRARY_IMPL.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <limits>
#include <vector>
#include "kernels.cuh"
#include "common/cuda_utils.h"

namespace d2p {
namespace lev {

// ============================================================================
// Helper Macros
// ============================================================================

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static torch::Tensor make_default_lengths(int B, int L1, int L2, const torch::TensorOptions& options);

namespace {

constexpr int64_t kLevCudaMaxIndex = static_cast<int64_t>(std::numeric_limits<int>::max());

int checked_lev_dim_to_int_cuda(int64_t value, const char* name) {
    TORCH_CHECK(
        value <= kLevCudaMaxIndex,
        name, " exceeds CUDA kernel int range: ", value, " > ", kLevCudaMaxIndex
    );
    return static_cast<int>(value);
}

int64_t checked_lev_alpha_size_cuda(int64_t max_L1, int64_t max_L2) {
    TORCH_CHECK(max_L1 >= 0 && max_L2 >= 0, "scores dimensions must be non-negative");
    const int64_t rows = max_L1 + 1;
    const int64_t cols = max_L2 + 1;
    TORCH_CHECK(
        cols == 0 || rows <= kLevCudaMaxIndex / cols,
        "lev DP workspace cell count exceeds CUDA kernel index range: (L1 + 1) * (L2 + 1) = ",
        rows, " * ", cols, " > ", kLevCudaMaxIndex
    );
    return rows * cols;
}

void validate_lev_scores_cuda(const torch::Tensor& scores) {
    CHECK_INPUT(scores);
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D (B, L1, L2)");
    TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be float32");
    checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
    checked_lev_alpha_size_cuda(scores.size(1), scores.size(2));
}

void validate_lev_lengths_cuda(
    const torch::Tensor& lengths,
    int B,
    int max_L1,
    int max_L2,
    const torch::Device& device
) {
    CHECK_CUDA(lengths);
    CHECK_CONTIGUOUS(lengths);
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

torch::Tensor resolve_lev_lengths_cuda(
    const c10::optional<torch::Tensor>& lengths_opt,
    int B,
    int max_L1,
    int max_L2,
    const torch::TensorOptions& options
) {
    torch::Tensor lengths = lengths_opt.has_value()
        ? lengths_opt.value()
        : make_default_lengths(B, max_L1, max_L2, options);
    validate_lev_lengths_cuda(lengths, B, max_L1, max_L2, options.device());
    return lengths;
}

void validate_lev_backward_input_cuda(const torch::Tensor& scores, const torch::Tensor& grad_posteriors) {
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

}  // namespace

// ============================================================================
// Helper: Create default lengths tensor
// ============================================================================

static torch::Tensor make_default_lengths(int B, int L1, int L2, const torch::TensorOptions& options) {
    auto cpu_options = torch::TensorOptions().dtype(torch::kInt32);
    auto lengths_cpu = torch::empty({B, 2}, cpu_options);
    auto lengths_acc = lengths_cpu.accessor<int32_t, 2>();
    for (int b = 0; b < B; b++) {
        lengths_acc[b][0] = L1;
        lengths_acc[b][1] = L2;
    }
    return lengths_cpu.to(options.device());
}

// ============================================================================
// Levenshtein CUDA Autograd Function
// ============================================================================

class SoftLevenshteinCUDAFunction : public torch::autograd::Function<SoftLevenshteinCUDAFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor scores,
        torch::Tensor ins_cost_t,
        torch::Tensor del_cost_t,
        torch::Tensor temperature,
        torch::Tensor lengths
    ) {
        validate_lev_scores_cuda(scores);
        D2P_CUDA_GUARD(scores);
        TORCH_CHECK(ins_cost_t.numel() == 1, "ins_cost must be a scalar tensor");
        TORCH_CHECK(del_cost_t.numel() == 1, "del_cost must be a scalar tensor");
        TORCH_CHECK(temperature.numel() == 1, "temperature must be a scalar tensor");

        const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
        const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
        const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
        const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

        validate_lev_lengths_cuda(lengths, B, max_L1, max_L2, scores.device());

        float ins_val = ins_cost_t.cpu().item<float>();
        float del_val = del_cost_t.cpu().item<float>();
        float temp_val = temperature.cpu().item<float>();

        auto options = scores.options();
        ctx->set_materialize_grads(false);
        torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
        torch::Tensor distance = torch::zeros({B}, options);
        torch::Tensor beta = torch::zeros({B, alpha_size}, options);
        torch::Tensor posteriors = torch::zeros({B, max_L1, max_L2}, options);
        torch::Tensor grad_ins = torch::zeros({B}, options);
        torch::Tensor grad_del = torch::zeros({B}, options);
        torch::Tensor grad_T = torch::zeros({B}, options);

        d2p::common::record_streams_current({&scores, &alpha, &distance, &lengths});
        lev_forward(
            scores.data_ptr<float>(),
            alpha.data_ptr<float>(),
            distance.data_ptr<float>(),
            lengths.data_ptr<int>(),
            B, max_L1, max_L2,
            ins_val, del_val, temp_val
        );

        d2p::common::record_streams_current({&alpha, &scores, &distance, &beta, &posteriors, &grad_ins, &grad_del, &grad_T, &lengths});
        lev_backward(
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

        D2P_CUDA_GUARD(scores);

        const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
        const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
        const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
        const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

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

            d2p::common::record_streams_current({&alpha, &scores, &distance, &beta, &posteriors, &tmp_ins, &tmp_del, &tmp_T, &lengths});
            lev_backward(
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
            TORCH_CHECK(grad_posteriors.sizes() == scores.sizes(),
                        "grad_posteriors shape mismatch");
            TORCH_CHECK(grad_posteriors.is_cuda(),
                        "grad_posteriors must be on CUDA");

            if (grad_posteriors.dtype() != torch::kFloat32) {
                grad_posteriors = grad_posteriors.to(torch::kFloat32);
            }
            if (grad_posteriors.device() != scores.device()) {
                grad_posteriors = grad_posteriors.to(scores.device());
            }
            grad_posteriors = grad_posteriors.contiguous();

            torch::Tensor d_alpha = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_distance = torch::zeros({B}, options);
            torch::Tensor beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor d_beta = torch::zeros({B, alpha_size}, options);
            torch::Tensor hvp_grad_scores = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &distance, &grad_posteriors, &d_alpha, &d_distance, &beta, &d_beta, &hvp_grad_scores, &lengths});
            lev_hvp(
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

            d2p::common::record_streams_current({&alpha, &scores, &distance, &U_ins, &beta_ins, &W_ins, &dP_dIns, &lengths});
            lev_param_grad(
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
                LEV_PARAM_INS
            );
            total_grad_ins += (grad_posteriors * dP_dIns).sum().reshape({1});

            // Param grad for del_cost
            torch::Tensor U_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_del = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dDel = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &distance, &U_del, &beta_del, &W_del, &dP_dDel, &lengths});
            lev_param_grad(
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
                LEV_PARAM_DEL
            );
            total_grad_del += (grad_posteriors * dP_dDel).sum().reshape({1});

            // Param grad for temperature
            torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
            torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

            d2p::common::record_streams_current({&alpha, &scores, &distance, &U_T, &beta_T, &W_T, &dP_dT, &lengths});
            lev_param_grad(
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
                LEV_PARAM_TEMPERATURE
            );
            total_grad_T += (grad_posteriors * dP_dT).sum().reshape({1});
        }

        return {grad_scores, total_grad_ins, total_grad_del, total_grad_T, torch::Tensor()};
    }
};

// ============================================================================
// Python Interface Functions
// ============================================================================

static std::vector<torch::Tensor> soft_levenshtein_cuda(
    torch::Tensor scores,
    torch::Tensor ins_cost,
    torch::Tensor del_cost,
    torch::Tensor temperature,
    torch::Tensor lengths
) {
    validate_lev_scores_cuda(scores);
    D2P_CUDA_GUARD(scores);
    return SoftLevenshteinCUDAFunction::apply(scores, ins_cost, del_cost, temperature, lengths);
}

static std::vector<torch::Tensor> soft_levenshtein_cuda_float(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lev_scores_cuda(scores);
    D2P_CUDA_GUARD(scores);

    const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    const int L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    const int L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");

    auto options = scores.options();
    torch::Tensor ins_t = torch::tensor({static_cast<float>(ins_cost)}, options);
    torch::Tensor del_t = torch::tensor({static_cast<float>(del_cost)}, options);
    torch::Tensor temp_t = torch::tensor({static_cast<float>(temperature)}, options);
    torch::Tensor lengths = resolve_lev_lengths_cuda(lengths_opt, B, L1, L2, options);

    return SoftLevenshteinCUDAFunction::apply(scores, ins_t, del_t, temp_t, lengths);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
soft_levenshtein_cuda_with_grads(
    torch::Tensor scores,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lev_scores_cuda(scores);
    D2P_CUDA_GUARD(scores);

    const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
    const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cuda(lengths_opt, B, max_L1, max_L2, options);

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

    d2p::common::record_streams_current({&scores, &alpha, &distance, &lengths});
    lev_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    d2p::common::record_streams_current({&alpha, &scores, &distance, &beta, &posteriors, &grad_ins, &grad_del, &grad_T, &lengths});
    lev_backward(
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

static torch::Tensor soft_levenshtein_hvp_cuda(
    torch::Tensor scores,
    torch::Tensor tangent,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lev_scores_cuda(scores);
    CHECK_INPUT(tangent);
    TORCH_CHECK(scores.sizes() == tangent.sizes(), "scores and tangent must have same shape");
    TORCH_CHECK(
        tangent.device() == scores.device(),
        "tangent must be on same device as scores, got ", tangent.device(), " vs ", scores.device()
    );
    D2P_CUDA_GUARD(scores);

    const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
    const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cuda(lengths_opt, B, max_L1, max_L2, options);

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

    d2p::common::record_streams_current({&scores, &alpha, &distance, &lengths});
    lev_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    d2p::common::record_streams_current({&alpha, &scores, &distance, &tangent, &d_alpha, &d_distance, &beta, &d_beta, &H_scores, &lengths});
    lev_hvp(
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

static torch::Tensor soft_levenshtein_param_jacobian_cuda(
    torch::Tensor scores,
    int64_t param_type,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lev_scores_cuda(scores);
    TORCH_CHECK(param_type >= 0 && param_type <= 2, "param_type must be 0 (ins), 1 (del), or 2 (temperature)");
    D2P_CUDA_GUARD(scores);

    const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
    const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cuda(lengths_opt, B, max_L1, max_L2, options);

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);
    torch::Tensor U = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta = torch::zeros({B, alpha_size}, options);
    torch::Tensor W = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dparam = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&scores, &alpha, &distance, &lengths});
    lev_forward(
        scores.data_ptr<float>(),
        alpha.data_ptr<float>(),
        distance.data_ptr<float>(),
        lengths.data_ptr<int>(),
        B, max_L1, max_L2,
        ins_val, del_val, temp_val
    );

    d2p::common::record_streams_current({&alpha, &scores, &distance, &U, &beta, &W, &dP_dparam, &lengths});
    lev_param_grad(
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
soft_levenshtein_backward_full_cuda(
    torch::Tensor scores,
    torch::Tensor grad_posteriors,
    double ins_cost,
    double del_cost,
    double temperature,
    c10::optional<torch::Tensor> lengths_opt
) {
    validate_lev_scores_cuda(scores);
    validate_lev_backward_input_cuda(scores, grad_posteriors);
    D2P_CUDA_GUARD(scores);

    const int B = checked_lev_dim_to_int_cuda(scores.size(0), "scores.size(0)");
    const int max_L1 = checked_lev_dim_to_int_cuda(scores.size(1), "scores.size(1)");
    const int max_L2 = checked_lev_dim_to_int_cuda(scores.size(2), "scores.size(2)");
    const int64_t alpha_size = checked_lev_alpha_size_cuda(max_L1, max_L2);

    auto options = scores.options();
    torch::Tensor lengths = resolve_lev_lengths_cuda(lengths_opt, B, max_L1, max_L2, options);

    grad_posteriors = grad_posteriors.contiguous();
    if (grad_posteriors.dtype() != torch::kFloat32) {
        grad_posteriors = grad_posteriors.to(torch::kFloat32);
    }

    float ins_val = static_cast<float>(ins_cost);
    float del_val = static_cast<float>(del_cost);
    float temp_val = static_cast<float>(temperature);

    // Forward pass
    torch::Tensor alpha = torch::zeros({B, alpha_size}, options);
    torch::Tensor distance = torch::zeros({B}, options);

    d2p::common::record_streams_current({&scores, &alpha, &distance, &lengths});
    lev_forward(
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

    d2p::common::record_streams_current({&alpha, &scores, &distance, &grad_posteriors, &d_alpha, &d_distance, &beta, &d_beta, &grad_scores, &lengths});
    lev_hvp(
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

    d2p::common::record_streams_current({&alpha, &scores, &distance, &U_ins, &beta_ins, &W_ins, &dP_dIns, &lengths});
    lev_param_grad(
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
        LEV_PARAM_INS
    );
    torch::Tensor total_grad_ins = (grad_posteriors * dP_dIns).sum().reshape({1});

    // Param grad for del_cost
    torch::Tensor U_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_del = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dDel = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &scores, &distance, &U_del, &beta_del, &W_del, &dP_dDel, &lengths});
    lev_param_grad(
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
        LEV_PARAM_DEL
    );
    torch::Tensor total_grad_del = (grad_posteriors * dP_dDel).sum().reshape({1});

    // Param grad for temperature
    torch::Tensor U_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor beta_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor W_T = torch::zeros({B, alpha_size}, options);
    torch::Tensor dP_dT = torch::zeros({B, max_L1, max_L2}, options);

    d2p::common::record_streams_current({&alpha, &scores, &distance, &U_T, &beta_T, &W_T, &dP_dT, &lengths});
    lev_param_grad(
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
        LEV_PARAM_TEMPERATURE
    );
    torch::Tensor total_grad_T = (grad_posteriors * dP_dT).sum().reshape({1});

    return std::make_tuple(grad_scores, total_grad_ins, total_grad_del, total_grad_T);
}

}  // namespace lev
}  // namespace d2p

// ============================================================================
// Module Registration
// ============================================================================

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_levenshtein", d2p::lev::soft_levenshtein_cuda);
    m.impl("soft_levenshtein_float", d2p::lev::soft_levenshtein_cuda_float);
    m.impl("soft_levenshtein_with_grads", d2p::lev::soft_levenshtein_cuda_with_grads);
    m.impl("soft_levenshtein_hvp", d2p::lev::soft_levenshtein_hvp_cuda);
    m.impl("soft_levenshtein_param_jacobian", d2p::lev::soft_levenshtein_param_jacobian_cuda);
    m.impl("soft_levenshtein_backward_full", d2p::lev::soft_levenshtein_backward_full_cuda);
}

TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_levenshtein", d2p::lev::soft_levenshtein_cuda);
    m.impl("soft_levenshtein_float", d2p::lev::soft_levenshtein_cuda_float);
}

#endif // USE_TORCH_LIBRARY
