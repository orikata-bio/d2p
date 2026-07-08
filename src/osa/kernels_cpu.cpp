// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft OSA CPU Kernel Implementations
 *
 * CPU implementation of Soft OSA (Optimal String Alignment).
 * Mirrors the CUDA kernel interface exactly for seamless dispatch.
 */

#include <cmath>
#include <algorithm>
#include <vector>

#include <ATen/Parallel.h>

#include "kernels_cpu.h"

namespace d2p {
namespace osa {
namespace cpu {

// ============================================================================
// Helper Functions
// ============================================================================

inline float safe_exp(float x) {
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

// Softmin for 4 values
inline float softmin4(float a, float b, float c, float d, float T) {
    float m = std::min({a, b, c, d});
    if (m >= PINF) return PINF;

    KahanAccumulator sum;
    if (a < PINF) sum.add(safe_exp(-(a - m) / T));
    if (b < PINF) sum.add(safe_exp(-(b - m) / T));
    if (c < PINF) sum.add(safe_exp(-(c - m) / T));
    if (d < PINF) sum.add(safe_exp(-(d - m) / T));

    float s = sum.result();
    if (s <= 0.0f) return PINF;
    return m - T * std::log(s);
}

// Softmin weights for 4 values
inline void softmin4_weights(float a, float b, float c, float d, float T,
                             float& wa, float& wb, float& wc, float& wd) {
    float m = std::min({a, b, c, d});
    if (m >= PINF) {
        wa = wb = wc = wd = 0.0f;
        return;
    }

    float ea = (a < PINF) ? safe_exp(-(a - m) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp(-(b - m) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp(-(c - m) / T) : 0.0f;
    float ed = (d < PINF) ? safe_exp(-(d - m) / T) : 0.0f;

    float total = ea + eb + ec + ed;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
        wc = ec / total;
        wd = ed / total;
    } else {
        wa = wb = wc = wd = 0.0f;
    }
}

inline size_t alpha_index(int i, int j, size_t alpha_cols) {
    return static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
}

inline size_t score_index(int i, int j, int max_L2) {
    return static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);
}

inline int diag_i_start(int64_t k, int L2) {
    return static_cast<int>(std::max<int64_t>(1, k - static_cast<int64_t>(L2)));
}

inline int diag_i_end(int64_t k, int L1) {
    return static_cast<int>(std::min<int64_t>(static_cast<int64_t>(L1), k - 1));
}

inline float compute_cost_param_grad_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    int L1,
    int L2,
    int max_L2,
    float ins_cost,
    float del_cost,
    float trans_cost,
    float T,
    int param_type
) {
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    std::vector<float> u((static_cast<size_t>(L1) + 1) * alpha_cols, 0.0f);

    if (param_type == OSA_PARAM_DEL_CPU) {
        for (int i = 1; i <= L1; ++i) {
            u[alpha_index(i, 0, alpha_cols)] = (float)i;
        }
    } else if (param_type == OSA_PARAM_INS_CPU) {
        for (int j = 1; j <= L2; ++j) {
            u[alpha_index(0, j, alpha_cols)] = (float)j;
        }
    }

    const int64_t diag_end = static_cast<int64_t>(L1) + static_cast<int64_t>(L2);
    for (int64_t k = 2; k <= diag_end; ++k) {
        int i_start = diag_i_start(k, L2);
        int i_end = diag_i_end(k, L1);

        for (int i = i_start; i <= i_end; ++i) {
            int j = static_cast<int>(k - static_cast<int64_t>(i));
            if (j < 1 || j > L2) continue;

            size_t idx = alpha_index(i, j, alpha_cols);
            size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
            size_t idx_up = alpha_index(i - 1, j, alpha_cols);
            size_t idx_left = alpha_index(i, j - 1, alpha_cols);
            size_t score_idx = score_index(i, j, max_L2);

            float sub_cost_val = sub_costs[score_idx];
            float trans_valid = (i >= 2 && j >= 2) ? trans_mask[score_idx] : 0.0f;

            float v_sub = alpha[idx_diag] + sub_cost_val;
            float v_del = alpha[idx_up] + del_cost;
            float v_ins = alpha[idx_left] + ins_cost;
            float v_trans = PINF;
            float du_trans = 0.0f;
            if (trans_valid > 0.5f) {
                size_t idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                v_trans = alpha[idx_trans] + trans_cost;
                du_trans = u[idx_trans];
            }

            float w_sub, w_del, w_ins, w_trans;
            softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

            float du_sub = u[idx_diag];
            float du_del = u[idx_up];
            float du_ins = u[idx_left];

            if (param_type == OSA_PARAM_INS_CPU) du_ins += 1.0f;
            else if (param_type == OSA_PARAM_DEL_CPU) du_del += 1.0f;
            else if (param_type == OSA_PARAM_TRANS_CPU && trans_valid > 0.5f) du_trans += 1.0f;

            u[idx] = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;
        }
    }

    return u[alpha_index(L1, L2, alpha_cols)];
}

// ============================================================================
// FORWARD PASS
// ============================================================================

void osa_forward_cpu(
    const float* sub_costs,
    const float* trans_mask,
    float* alpha,
    float* osa_score,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = sub_costs + b * score_stride;
            const float* tm = trans_mask + b * score_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize alpha
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = PINF;
            }

            // Base cases
            a[0] = 0.0f;
            for (int i = 1; i <= L1; i++) {
                a[alpha_index(i, 0, alpha_cols)] = i * del_cost;
            }
            for (int j = 1; j <= L2; j++) {
                a[alpha_index(0, j, alpha_cols)] = j * ins_cost;
            }

            // Forward DP
            const int64_t diag_end = static_cast<int64_t>(L1) + static_cast<int64_t>(L2);
            for (int64_t k = 2; k <= diag_end; k++) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float sub_cost_val = s[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost_val;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;
                    float v_trans = PINF;
                    if (trans_valid > 0.5f) {
                        size_t idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        v_trans = a[idx_trans] + trans_cost;
                    }

                    a[idx] = softmin4(v_sub, v_del, v_ins, v_trans, T);
                }
            }

            size_t final_idx = alpha_index(L1, L2, alpha_cols);
            osa_score[b] = a[final_idx];
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

void osa_backward_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    const float* osa_score,
    float* beta,
    float* posteriors,
    float* grad_T,
    float* grad_ins,
    float* grad_del,
    float* grad_trans,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = sub_costs + b * score_stride;
            const float* tm = trans_mask + b * score_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = 0.0f;
            }

            size_t final_idx = alpha_index(L1, L2, alpha_cols);
            be[final_idx] = 1.0f;

            float sum_T_grad = 0.0f;
            // Backward DP
            const int64_t diag_end = static_cast<int64_t>(L1) + static_cast<int64_t>(L2);
            for (int64_t k = diag_end; k >= 2; k--) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float beta_ij = be[idx];
                    if (beta_ij == 0.0f) continue;

                    float sub_cost_val = s[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost_val;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;
                    float v_trans = PINF;
                    size_t idx_trans = 0;
                    bool has_trans = false;
                    if (trans_valid > 0.5f) {
                        idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        has_trans = true;
                        v_trans = a[idx_trans] + trans_cost;
                    }

                    float w_sub, w_del, w_ins, w_trans;
                    softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

                    // Posteriors
                    post[score_idx] = beta_ij * w_sub;

                    // Temperature gradient
                    float alpha_ij = a[idx];
                    if (alpha_ij < PINF) {
                        float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
                        sum_T_grad += beta_ij * (alpha_ij - E_v) / T;
                    }

                    // Propagate beta
                    if (w_sub > 0.0f) {
                        be[idx_diag] += beta_ij * w_sub;
                    }
                    if (w_del > 0.0f) {
                        be[idx_up] += beta_ij * w_del;
                    }
                    if (w_ins > 0.0f) {
                        be[idx_left] += beta_ij * w_ins;
                    }
                    if (w_trans > 0.0f && has_trans) {
                        be[idx_trans] += beta_ij * w_trans;
                    }
                }
            }

            grad_T[b] = sum_T_grad;
            grad_ins[b] = compute_cost_param_grad_cpu(
                a, s, tm, L1, L2, max_L2, ins_cost, del_cost, trans_cost, T, OSA_PARAM_INS_CPU
            );
            grad_del[b] = compute_cost_param_grad_cpu(
                a, s, tm, L1, L2, max_L2, ins_cost, del_cost, trans_cost, T, OSA_PARAM_DEL_CPU
            );
            grad_trans[b] = compute_cost_param_grad_cpu(
                a, s, tm, L1, L2, max_L2, ins_cost, del_cost, trans_cost, T, OSA_PARAM_TRANS_CPU
            );
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT (HVP)
// ============================================================================

void osa_hvp_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    const float* osa_score,
    const float* V,
    float* d_alpha,
    float* d_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    float ins_cost, float del_cost, float trans_cost,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = sub_costs + b * score_stride;
            const float* tm = trans_mask + b * score_stride;
            const float* v = V + b * score_stride;
            float* da = d_alpha + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* dbe = d_beta + b * alpha_stride;
            float* H = H_scores + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                da[idx] = 0.0f;
                be[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                H[idx] = 0.0f;
            }

            // Forward tangent pass
            const int64_t diag_end = static_cast<int64_t>(L1) + static_cast<int64_t>(L2);
            for (int64_t k = 2; k <= diag_end; k++) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float sub_cost_val = s[score_idx];
                    float v_ij = v[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost_val;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;
                    float val_trans = PINF;
                    float da_trans = 0.0f;
                    size_t idx_trans = 0;
                    bool has_trans = false;
                    if (trans_valid > 0.5f) {
                        idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        has_trans = true;
                        val_trans = a[idx_trans] + trans_cost;
                        da_trans = da[idx_trans];
                    }

                    float w_sub, w_del, w_ins, w_trans;
                    softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

                    float dv_sub = da[idx_diag] + v_ij;
                    float dv_del = da[idx_up];
                    float dv_ins = da[idx_left];
                    float dv_trans = has_trans ? da_trans : 0.0f;

                    da[idx] = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins + w_trans * dv_trans;
                }
            }

            size_t final_idx = alpha_index(L1, L2, alpha_cols);
            d_score[b] = da[final_idx];

            // Backward pass
            be[final_idx] = 1.0f;
            dbe[final_idx] = 0.0f;

            for (int64_t k = diag_end; k >= 2; k--) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];
                    float sub_cost_val = s[score_idx];
                    float v_ij = v[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    if (beta_ij == 0.0f && std::abs(dbeta_ij) < 1e-20f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_sub = a_diag + sub_cost_val;
                    float val_del = a_up + del_cost;
                    float val_ins = a_left + ins_cost;
                    float val_trans = PINF;
                    float da_trans = 0.0f;
                    size_t idx_trans = 0;
                    bool has_trans = false;
                    if (trans_valid > 0.5f) {
                        idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        has_trans = true;
                        val_trans = a[idx_trans] + trans_cost;
                        da_trans = da[idx_trans];
                    }

                    float w_sub, w_del, w_ins, w_trans;
                    softmin4_weights(val_sub, val_del, val_ins, val_trans, T, w_sub, w_del, w_ins, w_trans);

                    float dv_sub = da[idx_diag] + v_ij;
                    float dv_del = da[idx_up];
                    float dv_ins = da[idx_left];
                    float dv_trans = has_trans ? da_trans : 0.0f;

                    float E_dv = w_sub * dv_sub + w_del * dv_del + w_ins * dv_ins + w_trans * dv_trans;

                    // Weight tangents for softmin
                    float dw_sub = -w_sub * (dv_sub - E_dv) / T;
                    float dw_del = -w_del * (dv_del - E_dv) / T;
                    float dw_ins = -w_ins * (dv_ins - E_dv) / T;
                    float dw_trans = -w_trans * (dv_trans - E_dv) / T;

                    // HVP
                    H[score_idx] = dbeta_ij * w_sub + beta_ij * dw_sub;

                    // Propagate
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
                    if (w_trans > 0.0f && has_trans) {
                        be[idx_trans] += beta_ij * w_trans;
                        dbe[idx_trans] += dbeta_ij * w_trans + beta_ij * dw_trans;
                    }
                }
            }
        }
    });
}

// ============================================================================
// PARAMETER GRADIENT
// ============================================================================

void osa_param_grad_cpu(
    const float* alpha,
    const float* sub_costs,
    const float* trans_mask,
    const float* osa_score,
    float* U,
    float* beta,
    float* W,
    float* dP_dparam,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float ins_cost, float del_cost, float trans_cost, float T,
    int param_type
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = sub_costs + b * score_stride;
            const float* tm = trans_mask + b * score_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w_buf = W + b * alpha_stride;
            float* dP = dP_dparam + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

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
            for (int i = 1; i <= L1; i++) {
                u[alpha_index(i, 0, alpha_cols)] = (param_type == OSA_PARAM_DEL_CPU) ? (float)i : 0.0f;
            }
            for (int j = 1; j <= L2; j++) {
                u[alpha_index(0, j, alpha_cols)] = (param_type == OSA_PARAM_INS_CPU) ? (float)j : 0.0f;
            }

            // Forward U
            const int64_t diag_end = static_cast<int64_t>(L1) + static_cast<int64_t>(L2);
            for (int64_t k = 2; k <= diag_end; k++) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float sub_cost_val = s[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost_val;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;
                    float v_trans = PINF;
                    float u_trans_val = 0.0f;
                    if (trans_valid > 0.5f) {
                        size_t idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        v_trans = a[idx_trans] + trans_cost;
                        u_trans_val = u[idx_trans];
                    }

                    float w_sub, w_del, w_ins, w_trans;
                    softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

                    float du_sub = u[idx_diag];
                    float du_del = u[idx_up];
                    float du_ins = u[idx_left];
                    float du_trans = u_trans_val;

                    if (param_type == OSA_PARAM_INS_CPU) du_ins += 1.0f;
                    else if (param_type == OSA_PARAM_DEL_CPU) du_del += 1.0f;
                    else if (param_type == OSA_PARAM_TRANS_CPU && trans_valid > 0.5f) du_trans += 1.0f;

                    float U_val = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

                    if (param_type == OSA_PARAM_TEMPERATURE_CPU) {
                        float alpha_ij = a[idx];
                        if (alpha_ij < PINF) {
                            float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
                            U_val += (alpha_ij - E_v) / T;
                        }
                    }

                    u[idx] = U_val;
                }
            }

            // Backward
            size_t final_idx = alpha_index(L1, L2, alpha_cols);
            be[final_idx] = 1.0f;
            w_buf[final_idx] = 0.0f;

            for (int64_t k = diag_end; k >= 2; k--) {
                int i_start = diag_i_start(k, L2);
                int i_end = diag_i_end(k, L1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = static_cast<int>(k - static_cast<int64_t>(i));
                    if (j < 1 || j > L2) continue;

                    size_t idx = alpha_index(i, j, alpha_cols);
                    size_t idx_diag = alpha_index(i - 1, j - 1, alpha_cols);
                    size_t idx_up = alpha_index(i - 1, j, alpha_cols);
                    size_t idx_left = alpha_index(i, j - 1, alpha_cols);
                    size_t score_idx = score_index(i, j, max_L2);

                    float beta_ij = be[idx];
                    float W_ij = w_buf[idx];
                    float sub_cost_val = s[score_idx];
                    float trans_valid = (i >= 2 && j >= 2) ? tm[score_idx] : 0.0f;

                    if (beta_ij == 0.0f && std::abs(W_ij) < 1e-20f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_sub = a_diag + sub_cost_val;
                    float v_del = a_up + del_cost;
                    float v_ins = a_left + ins_cost;
                    float v_trans = PINF;
                    float u_trans_val = 0.0f;
                    size_t idx_trans = 0;
                    bool has_trans = false;
                    if (trans_valid > 0.5f) {
                        idx_trans = alpha_index(i - 2, j - 2, alpha_cols);
                        has_trans = true;
                        v_trans = a[idx_trans] + trans_cost;
                        u_trans_val = u[idx_trans];
                    }

                    float w_sub, w_del, w_ins, w_trans;
                    softmin4_weights(v_sub, v_del, v_ins, v_trans, T, w_sub, w_del, w_ins, w_trans);

                    // Accumulate dP from W path
                    dP[score_idx] += W_ij * w_sub;

                    float du_sub = u[idx_diag];
                    float du_del = u[idx_up];
                    float du_ins = u[idx_left];
                    float du_trans = u_trans_val;

                    if (param_type == OSA_PARAM_INS_CPU) du_ins += 1.0f;
                    else if (param_type == OSA_PARAM_DEL_CPU) du_del += 1.0f;
                    else if (param_type == OSA_PARAM_TRANS_CPU && trans_valid > 0.5f) du_trans += 1.0f;

                    float E_dv = w_sub * du_sub + w_del * du_del + w_ins * du_ins + w_trans * du_trans;

                    float dw_sub = w_sub * (-du_sub + E_dv) / T;
                    float dw_del = w_del * (-du_del + E_dv) / T;
                    float dw_ins = w_ins * (-du_ins + E_dv) / T;
                    float dw_trans = w_trans * (-du_trans + E_dv) / T;

                    if (param_type == OSA_PARAM_TEMPERATURE_CPU) {
                        float E_v = w_sub * v_sub + w_del * v_del + w_ins * v_ins + w_trans * v_trans;
                        float inv_T2 = 1.0f / (T * T);
                        dw_sub += w_sub * (v_sub - E_v) * inv_T2;
                        dw_del += w_del * (v_del - E_v) * inv_T2;
                        dw_ins += w_ins * (v_ins - E_v) * inv_T2;
                        dw_trans += w_trans * (v_trans - E_v) * inv_T2;
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
                    if (w_trans > 0.0f && has_trans) {
                        be[idx_trans] += beta_ij * w_trans;
                        w_buf[idx_trans] += W_ij * w_trans + beta_ij * dw_trans;
                    }
                }
            }
        }
    });
}

}  // namespace cpu
}  // namespace osa
}  // namespace d2p
