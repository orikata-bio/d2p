// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft Levenshtein (Edit Distance) CPU Kernel Implementations
 *
 * CPU implementations mirroring CUDA kernels for seamless dispatch.
 * Uses softmin for minimization (edit distance) rather than logsumexp.
 */

#include "kernels_cpu.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <algorithm>

namespace d2p {
namespace lev {

// ============================================================================
// Constants and Helper Functions
// ============================================================================

constexpr float PINF_CPU = 1e30f;  // Positive infinity for minimization

inline float safe_exp_cpu(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

// Kahan compensated summation
struct KahanAccumulator {
    float sum = 0.0f;
    float c = 0.0f;

    void add(float value) {
        float y = value - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    float result() const { return sum; }
};

// Softmin for 3 values
inline float softmin3_cpu(float a, float b, float c, float T) {
    float m = std::min({a, b, c});
    if (m >= PINF_CPU) return PINF_CPU;

    KahanAccumulator sum;
    if (a < PINF_CPU) sum.add(safe_exp_cpu(-(a - m) / T));
    if (b < PINF_CPU) sum.add(safe_exp_cpu(-(b - m) / T));
    if (c < PINF_CPU) sum.add(safe_exp_cpu(-(c - m) / T));

    float s = sum.result();
    if (s <= 0.0f) return PINF_CPU;
    return m - T * std::log(s);
}

// Softmin weights
inline void softmin3_weights_cpu(float a, float b, float c, float T,
                                  float& wa, float& wb, float& wc) {
    float m = std::min({a, b, c});
    if (m >= PINF_CPU) {
        wa = wb = wc = 0.0f;
        return;
    }

    float ea = (a < PINF_CPU) ? safe_exp_cpu(-(a - m) / T) : 0.0f;
    float eb = (b < PINF_CPU) ? safe_exp_cpu(-(b - m) / T) : 0.0f;
    float ec = (c < PINF_CPU) ? safe_exp_cpu(-(c - m) / T) : 0.0f;

    float total = ea + eb + ec;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
        wc = ec / total;
    } else {
        wa = wb = wc = 0.0f;
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

void lev_forward_cpu(
    const float* scores,
    float* alpha,
    float* distance,
    const int* lengths,
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
) {
    const size_t alpha_stride = static_cast<size_t>(max_L1 + 1) * static_cast<size_t>(max_L2 + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2 + 1);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const size_t b = static_cast<size_t>(batch);
            const float* s = scores + b * score_stride;
            float* a = alpha + b * alpha_stride;
            const int64_t L1 = lengths[b * 2];
            const int64_t L2 = lengths[b * 2 + 1];

            // Initialize alpha
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = PINF_CPU;
            }

            // Base cases
            a[0] = 0.0f;
            for (int64_t i = 1; i <= L1; i++) {
                a[static_cast<size_t>(i) * alpha_cols] = static_cast<float>(i) * del_cost;
            }
            for (int64_t j = 1; j <= L2; j++) {
                a[static_cast<size_t>(j)] = static_cast<float>(j) * ins_cost;
            }

            // Forward DP
            for (int64_t k = 2; k <= L1 + L2; k++) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sub_cost = s[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;

                    a[idx] = softmin3_cpu(v_sub, v_del, v_ins, T);
                }
            }

            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            distance[b] = a[final_idx];
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

void lev_backward_cpu(
    const float* alpha,
    const float* scores,
    const float* distance,
    float* beta,
    float* posteriors,
    float* grad_ins,
    float* grad_del,
    float* grad_T,
    const int* lengths,
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
) {
    const size_t alpha_stride = static_cast<size_t>(max_L1 + 1) * static_cast<size_t>(max_L2 + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2 + 1);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const size_t b = static_cast<size_t>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * score_stride;
            const int64_t L1 = lengths[b * 2];
            const int64_t L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = 0.0f;
            }

            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;

            float sum_ins_grad = 0.0f;
            float sum_del_grad = 0.0f;
            float sum_T_grad = 0.0f;

            // Backward DP
            for (int64_t k = L1 + L2; k >= 2; k--) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float beta_ij = be[idx];
                    if (beta_ij == 0.0f) continue;

                    float sub_cost = s[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;

                    float w_sub, w_del, w_ins;
                    softmin3_weights_cpu(v_sub, v_del, v_ins, T, w_sub, w_del, w_ins);

                    post[score_idx] = beta_ij * w_sub;

                    sum_del_grad += beta_ij * w_del;
                    sum_ins_grad += beta_ij * w_ins;

                    float alpha_ij = a[idx];
                    if (alpha_ij < PINF_CPU) {
                        float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins;
                        sum_T_grad += beta_ij * (alpha_ij - E_v) / T;
                    }

                    if (w_sub > 0.0f) be[idx_diag] += beta_ij * w_sub;
                    if (w_del > 0.0f) be[idx_up] += beta_ij * w_del;
                    if (w_ins > 0.0f) be[idx_left] += beta_ij * w_ins;
                }
            }

            // Boundary cells are initialized from closed-form gap costs rather than
            // DP recurrences, so their parameter contributions must be added here.
            for (int64_t i = 1; i <= L1; i++) {
                sum_del_grad += be[static_cast<size_t>(i) * alpha_cols] * static_cast<float>(i);
            }
            for (int64_t j = 1; j <= L2; j++) {
                sum_ins_grad += be[static_cast<size_t>(j)] * static_cast<float>(j);
            }

            grad_ins[b] = sum_ins_grad;
            grad_del[b] = sum_del_grad;
            grad_T[b] = sum_T_grad;
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT
// ============================================================================

void lev_hvp_cpu(
    const float* alpha,
    const float* scores,
    const float* distance,
    const float* V,
    float* d_alpha,
    float* d_distance,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T
) {
    const size_t alpha_stride = static_cast<size_t>(max_L1 + 1) * static_cast<size_t>(max_L2 + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2 + 1);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const size_t b = static_cast<size_t>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            const float* v = V + b * score_stride;
            float* da = d_alpha + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* dbe = d_beta + b * alpha_stride;
            float* H = H_scores + b * score_stride;
            const int64_t L1 = lengths[b * 2];
            const int64_t L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                da[idx] = 0.0f;
                be[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                H[idx] = 0.0f;
            }

            // Forward tangent
            da[0] = 0.0f;
            for (int64_t i = 1; i <= L1; i++) da[static_cast<size_t>(i) * alpha_cols] = 0.0f;
            for (int64_t j = 1; j <= L2; j++) da[static_cast<size_t>(j)] = 0.0f;

            for (int64_t k = 2; k <= L1 + L2; k++) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sub_cost = s[score_idx];
                    float v_ij = v[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;

                    float w_sub, w_del, w_ins;
                    softmin3_weights_cpu(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

                    float dv_sub = da[idx_diag] + v_ij;
                    float dv_del = da[idx_up];
                    float dv_ins = da[idx_left];

                    da[idx] = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;
                }
            }

            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            d_distance[b] = da[final_idx];

            // Initialize beta
            be[final_idx] = 1.0f;
            dbe[final_idx] = 0.0f;

            // Backward tangent
            for (int64_t k = L1 + L2; k >= 2; k--) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sub_cost = s[score_idx];
                    float v_ij = v[score_idx];
                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];

                    if (beta_ij == 0.0f && std::abs(dbeta_ij) < 1e-20f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;

                    float w_sub, w_del, w_ins;
                    softmin3_weights_cpu(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

                    float dv_sub = da[idx_diag] + v_ij;
                    float dv_del = da[idx_up];
                    float dv_ins = da[idx_left];

                    float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;

                    float dw_sub = w_sub * (-dv_sub + E_dv) / T;
                    float dw_del = w_del * (-dv_del + E_dv) / T;
                    float dw_ins = w_ins * (-dv_ins + E_dv) / T;

                    H[score_idx] = dbeta_ij * w_sub + beta_ij * dw_sub;

                    if (w_sub > 0.0f) {
                        be[idx_diag] += beta_ij * w_sub;
                        dbe[idx_diag] += dbeta_ij * w_sub + beta_ij * dw_sub;
                    }
                    if (w_del > 0.0f) {
                        be[idx_up] += beta_ij * w_del;
                        dbe[idx_up] += dbeta_ij * w_del + beta_ij * dw_del;
                    }
                    if (w_ins > 0.0f) {
                        be[idx_left] += beta_ij * w_ins;
                        dbe[idx_left] += dbeta_ij * w_ins + beta_ij * dw_ins;
                    }
                }
            }

        }
    });
}

// ============================================================================
// PARAMETER GRADIENT
// ============================================================================

void lev_param_grad_cpu(
    const float* alpha,
    const float* scores,
    const float* distance,
    float* U,
    float* beta,
    float* W,
    float* dP_dparam,
    const int* lengths,
    int64_t B, int64_t max_L1, int64_t max_L2,
    float ins_cost, float del_cost, float T,
    int param_type
) {
    const size_t alpha_stride = static_cast<size_t>(max_L1 + 1) * static_cast<size_t>(max_L2 + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2 + 1);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const size_t b = static_cast<size_t>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w_buf = W + b * alpha_stride;
            float* dP = dP_dparam + b * score_stride;
            const int64_t L1 = lengths[b * 2];
            const int64_t L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                u[idx] = 0.0f;
                be[idx] = 0.0f;
                w_buf[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                dP[idx] = 0.0f;
            }

            // Base case tangents
            u[0] = 0.0f;
            for (int64_t i = 1; i <= L1; i++) {
                u[static_cast<size_t>(i) * alpha_cols] = (param_type == LEV_PARAM_DEL_CPU) ? static_cast<float>(i) : 0.0f;
            }
            for (int64_t j = 1; j <= L2; j++) {
                u[static_cast<size_t>(j)] = (param_type == LEV_PARAM_INS_CPU) ? static_cast<float>(j) : 0.0f;
            }

            // Forward U
            for (int64_t k = 2; k <= L1 + L2; k++) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sub_cost = s[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;

                    float w_sub, w_del, w_ins;
                    softmin3_weights_cpu(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

                    float u_diag = u[idx_diag];
                    float u_up = u[idx_up];
                    float u_left = u[idx_left];

                    float du_sub = u_diag;
                    float du_del = u_up;
                    float du_ins = u_left;

                    if (param_type == LEV_PARAM_INS_CPU) du_ins += 1.0f;
                    else if (param_type == LEV_PARAM_DEL_CPU) du_del += 1.0f;

                    float U_val = w_sub * du_sub + w_del * du_del + w_ins * du_ins;

                    if (param_type == LEV_PARAM_TEMPERATURE_CPU) {
                        float alpha_ij = a[idx];
                        float E_v = w_sub * val_sub + w_del * val_del + w_ins * val_ins;
                        U_val += (alpha_ij - E_v) / T;
                    }

                    u[idx] = U_val;
                }
            }

            // Backward
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;
            w_buf[final_idx] = 0.0f;

            for (int64_t k = L1 + L2; k >= 2; k--) {
                int64_t i_start = std::max<int64_t>(1, k - L2);
                int64_t i_end = std::min<int64_t>(L1, k - 1);

                for (int64_t i = i_start; i <= i_end; i++) {
                    int64_t j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float beta_ij = be[idx];
                    float W_ij = w_buf[idx];
                    float sub_cost = s[score_idx];

                    if (beta_ij == 0.0f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;

                    float w_sub, w_del, w_ins;
                    softmin3_weights_cpu(val_sub, val_del, val_ins, T, w_sub, w_del, w_ins);

                    dP[score_idx] += W_ij * w_sub;

                    float u_diag = u[idx_diag];
                    float u_up = u[idx_up];
                    float u_left = u[idx_left];

                    float dv_sub = u_diag;
                    float dv_del = u_up;
                    float dv_ins = u_left;

                    if (param_type == LEV_PARAM_INS_CPU) dv_ins += 1.0f;
                    else if (param_type == LEV_PARAM_DEL_CPU) dv_del += 1.0f;

                    float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins;

                    float dw_sub = w_sub * (-dv_sub + E_dv) / T;
                    float dw_del = w_del * (-dv_del + E_dv) / T;
                    float dw_ins = w_ins * (-dv_ins + E_dv) / T;

                    if (param_type == LEV_PARAM_TEMPERATURE_CPU) {
                        float E_v = w_sub * val_sub + w_del * val_del + w_ins * val_ins;
                        float inv_T2 = 1.0f / (T * T);
                        dw_sub += w_sub * (val_sub - E_v) * inv_T2;
                        dw_del += w_del * (val_del - E_v) * inv_T2;
                        dw_ins += w_ins * (val_ins - E_v) * inv_T2;
                    }

                    dP[score_idx] += beta_ij * dw_sub;

                    if (w_sub > 0.0f) {
                        be[idx_diag] += beta_ij * w_sub;
                        w_buf[idx_diag] += W_ij * w_sub + beta_ij * dw_sub;
                    }
                    if (w_del > 0.0f) {
                        be[idx_up] += beta_ij * w_del;
                        w_buf[idx_up] += W_ij * w_del + beta_ij * dw_del;
                    }
                    if (w_ins > 0.0f) {
                        be[idx_left] += beta_ij * w_ins;
                        w_buf[idx_left] += W_ij * w_ins + beta_ij * dw_ins;
                    }
                }
            }
        }
    });
}

}  // namespace lev
}  // namespace d2p
