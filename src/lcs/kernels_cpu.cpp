// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft LCS CPU Kernel Implementations
 *
 * CPU implementation mirroring CUDA interface for seamless dispatch.
 * LCS uses SOFTMAX (maximization) - only temperature parameter.
 */

#include "kernels_cpu.h"
#include <cmath>
#include <algorithm>

#include <ATen/Parallel.h>

namespace d2p {
namespace lcs {
namespace cpu {

// ============================================================================
// Helper Functions
// ============================================================================

// Safe exponential with bounds for float32
inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

// Kahan compensated summation for numerical precision
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

// Softmax for 3 values: max + T * log(sum(exp((x-max)/T)))
inline float softmax3(float a, float b, float c, float T) {
    float m = std::max({a, b, c});
    if (m <= NINF) return NINF;

    KahanAccumulator sum;
    if (a > NINF) sum.add(safe_exp((a - m) / T));
    if (b > NINF) sum.add(safe_exp((b - m) / T));
    if (c > NINF) sum.add(safe_exp((c - m) / T));

    float s = sum.result();
    if (s <= 0.0f) return NINF;
    return m + T * std::log(s);
}

// Softmax weights: w_r = exp((a_r-m)/T) / sum exp((a_q-m)/T)
inline void softmax3_weights(float a, float b, float c, float T,
                             float& wa, float& wb, float& wc) {
    float m = std::max({a, b, c});
    if (m <= NINF) {
        wa = wb = wc = 0.0f;
        return;
    }

    float ea = (a > NINF) ? safe_exp((a - m) / T) : 0.0f;
    float eb = (b > NINF) ? safe_exp((b - m) / T) : 0.0f;
    float ec = (c > NINF) ? safe_exp((c - m) / T) : 0.0f;

    float total = ea + eb + ec;
    if (total > 0.0f) {
        wa = ea / total;
        wb = eb / total;
        wc = ec / total;
    } else {
        wa = wb = wc = 0.0f;
    }
}

inline size_t lcs_cell_index(int i, int j, size_t alpha_cols) {
    return static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
}

inline size_t lcs_score_index(int i, int j, int max_L2) {
    return static_cast<size_t>(i) * static_cast<size_t>(max_L2) + static_cast<size_t>(j);
}

// ============================================================================
// Forward Pass
// ============================================================================

void lcs_forward_cpu(
    const float* scores,
    float* alpha,
    float* lcs_score,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = scores + b * score_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize alpha with base cases
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = NINF;
            }

            // Base cases for LCS: all boundary cells are 0
            for (int i = 0; i <= L1; i++) {
                a[lcs_cell_index(i, 0, alpha_cols)] = 0.0f;
            }
            for (int j = 0; j <= L2; j++) {
                a[static_cast<size_t>(j)] = 0.0f;
            }

            // Forward DP: process anti-diagonals
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float match_score = s[score_idx];

                    // Three transitions
                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    // Option values
                    float v_match = a_diag + match_score;
                    float v_skip1 = a_up;
                    float v_skip2 = a_left;

                    a[idx] = softmax3(v_match, v_skip1, v_skip2, T);
                }
            }

            // Score is the final cell
            const size_t final_idx = lcs_cell_index(L1, L2, alpha_cols);
            lcs_score[b] = a[final_idx];
        }
    });
}

// ============================================================================
// Backward Pass
// ============================================================================

void lcs_backward_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    float* beta,
    float* posteriors,
    float* grad_T,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)lcs_score;  // unused but kept for interface consistency

    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize posteriors and beta to 0
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = 0.0f;
            }

            // Terminal condition: beta[L1, L2] = 1
            const size_t final_idx = lcs_cell_index(L1, L2, alpha_cols);
            be[final_idx] = 1.0f;

            // Initialize gradient
            float sum_T_grad = 0.0f;

            // Backward DP: process anti-diagonals in reverse
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float beta_ij = be[idx];
                    if (beta_ij == 0.0f) continue;

                    float match_score = s[score_idx];

                    // Compute option values
                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float v_match = a_diag + match_score;
                    float v_skip1 = a_up;
                    float v_skip2 = a_left;

                    float w_match, w_skip1, w_skip2;
                    softmax3_weights(v_match, v_skip1, v_skip2, T, w_match, w_skip1, w_skip2);

                    // Posteriors for match
                    post[score_idx] = beta_ij * w_match;

                    // Temperature gradient
                    float alpha_ij = a[idx];
                    if (alpha_ij > NINF) {
                        float E_v = w_match * v_match + w_skip1 * v_skip1 + w_skip2 * v_skip2;
                        sum_T_grad += beta_ij * (alpha_ij - E_v) / T;
                    }

                    // Propagate beta to predecessors
                    if (w_match > 0.0f) {
                        be[idx_diag] += beta_ij * w_match;
                    }
                    if (w_skip1 > 0.0f) {
                        be[idx_up] += beta_ij * w_skip1;
                    }
                    if (w_skip2 > 0.0f) {
                        be[idx_left] += beta_ij * w_skip2;
                    }
                }
            }

            grad_T[b] = sum_T_grad;
        }
    });
}

// ============================================================================
// Hessian-Vector Product
// ============================================================================

void lcs_hvp_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    const float* V,
    float* d_alpha,
    float* d_lcs_score,
    float* beta,
    float* d_beta,
    float* H_scores,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)lcs_score;  // unused but kept for interface consistency

    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            const float* v = V + b * score_stride;
            float* da = d_alpha + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* dbe = d_beta + b * alpha_stride;
            float* H = H_scores + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize workspaces
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                da[idx] = 0.0f;
                be[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                H[idx] = 0.0f;
            }

            // =========== Forward tangent pass ===========
            da[0] = 0.0f;
            for (int i = 1; i <= L1; i++) {
                da[lcs_cell_index(i, 0, alpha_cols)] = 0.0f;
            }
            for (int j = 1; j <= L2; j++) {
                da[static_cast<size_t>(j)] = 0.0f;
            }

            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float match_score = s[score_idx];
                    float v_ij = v[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_match = a_diag + match_score;
                    float val_skip1 = a_up;
                    float val_skip2 = a_left;

                    float w_match, w_skip1, w_skip2;
                    softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

                    float dv_match = da[idx_diag] + v_ij;
                    float dv_skip1 = da[idx_up];
                    float dv_skip2 = da[idx_left];

                    da[idx] = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;
                }
            }

            const size_t final_idx = lcs_cell_index(L1, L2, alpha_cols);
            d_lcs_score[b] = da[final_idx];

            // =========== Backward pass ===========
            be[final_idx] = 1.0f;
            dbe[final_idx] = 0.0f;

            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float match_score = s[score_idx];
                    float v_ij = v[score_idx];
                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];

                    if (beta_ij == 0.0f && std::abs(dbeta_ij) < 1e-20f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_match = a_diag + match_score;
                    float val_skip1 = a_up;
                    float val_skip2 = a_left;

                    float w_match, w_skip1, w_skip2;
                    softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

                    float dv_match = da[idx_diag] + v_ij;
                    float dv_skip1 = da[idx_up];
                    float dv_skip2 = da[idx_left];

                    float E_dv = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;

                    float dw_match = w_match * (dv_match - E_dv) / T;
                    float dw_skip1 = w_skip1 * (dv_skip1 - E_dv) / T;
                    float dw_skip2 = w_skip2 * (dv_skip2 - E_dv) / T;

                    H[score_idx] = dbeta_ij * w_match + beta_ij * dw_match;

                    if (w_match > 0.0f) {
                        be[idx_diag] += beta_ij * w_match;
                        dbe[idx_diag] += dbeta_ij * w_match + beta_ij * dw_match;
                    }
                    if (w_skip1 > 0.0f) {
                        be[idx_up] += beta_ij * w_skip1;
                        dbe[idx_up] += dbeta_ij * w_skip1 + beta_ij * dw_skip1;
                    }
                    if (w_skip2 > 0.0f) {
                        be[idx_left] += beta_ij * w_skip2;
                        dbe[idx_left] += dbeta_ij * w_skip2 + beta_ij * dw_skip2;
                    }
                }
            }
        }
    });
}

// ============================================================================
// Parameter Gradient (dP/dT)
// ============================================================================

void lcs_param_grad_cpu(
    const float* alpha,
    const float* scores,
    const float* lcs_score,
    float* U,
    float* beta,
    float* W,
    float* dP_dT,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T
) {
    (void)lcs_score;  // unused but kept for interface consistency

    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t score_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w_buf = W + b * alpha_stride;
            float* dP = dP_dT + b * score_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize workspaces
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                u[idx] = 0.0f;
                be[idx] = 0.0f;
                w_buf[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                dP[idx] = 0.0f;
            }

            // =========== Forward U pass: compute d(alpha)/dT ===========
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float match_score = s[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_match = a_diag + match_score;
                    float val_skip1 = a_up;
                    float val_skip2 = a_left;

                    float w_match, w_skip1, w_skip2;
                    softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

                    float u_diag = u[idx_diag];
                    float u_up = u[idx_up];
                    float u_left = u[idx_left];

                    float U_val = w_match * u_diag + w_skip1 * u_up + w_skip2 * u_left;

                    // Add direct temperature derivative
                    float alpha_ij = a[idx];
                    float E_v = w_match * val_match + w_skip1 * val_skip1 + w_skip2 * val_skip2;
                    U_val += (alpha_ij - E_v) / T;

                    u[idx] = U_val;
                }
            }

            // =========== Backward pass ===========
            const size_t final_idx = lcs_cell_index(L1, L2, alpha_cols);
            be[final_idx] = 1.0f;
            w_buf[final_idx] = 0.0f;

            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    const size_t idx = lcs_cell_index(i, j, alpha_cols);
                    const size_t idx_diag = lcs_cell_index(i - 1, j - 1, alpha_cols);
                    const size_t idx_up = lcs_cell_index(i - 1, j, alpha_cols);
                    const size_t idx_left = lcs_cell_index(i, j - 1, alpha_cols);
                    const size_t score_idx = lcs_score_index(i - 1, j - 1, max_L2);

                    float beta_ij = be[idx];
                    float W_ij = w_buf[idx];
                    float match_score = s[score_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    float val_match = a_diag + match_score;
                    float val_skip1 = a_up;
                    float val_skip2 = a_left;

                    float w_match, w_skip1, w_skip2;
                    softmax3_weights(val_match, val_skip1, val_skip2, T, w_match, w_skip1, w_skip2);

                    // Accumulate dP/dT = W * w_match
                    dP[score_idx] += W_ij * w_match;

                    if (beta_ij == 0.0f) continue;

                    float u_diag = u[idx_diag];
                    float u_up = u[idx_up];
                    float u_left = u[idx_left];

                    float dv_match = u_diag;
                    float dv_skip1 = u_up;
                    float dv_skip2 = u_left;

                    float E_dv = w_match * dv_match + w_skip1 * dv_skip1 + w_skip2 * dv_skip2;

                    float dw_match = w_match * (dv_match - E_dv) / T;
                    float dw_skip1 = w_skip1 * (dv_skip1 - E_dv) / T;
                    float dw_skip2 = w_skip2 * (dv_skip2 - E_dv) / T;

                    // Add direct temperature derivative: dw_k/dT = w_k * (E[v] - v_k) / T^2
                    float E_v = w_match * val_match + w_skip1 * val_skip1 + w_skip2 * val_skip2;
                    float inv_T2 = 1.0f / (T * T);
                    dw_match += w_match * (E_v - val_match) * inv_T2;
                    dw_skip1 += w_skip1 * (E_v - val_skip1) * inv_T2;
                    dw_skip2 += w_skip2 * (E_v - val_skip2) * inv_T2;

                    // Add beta * dw to dP
                    dP[score_idx] += beta_ij * dw_match;

                    // Propagate beta and W
                    if (w_match > 0.0f) {
                        be[idx_diag] += beta_ij * w_match;
                        w_buf[idx_diag] += W_ij * w_match + beta_ij * dw_match;
                    }
                    if (w_skip1 > 0.0f) {
                        be[idx_up] += beta_ij * w_skip1;
                        w_buf[idx_up] += W_ij * w_skip1 + beta_ij * dw_skip1;
                    }
                    if (w_skip2 > 0.0f) {
                        be[idx_left] += beta_ij * w_skip2;
                        w_buf[idx_left] += W_ij * w_skip2 + beta_ij * dw_skip2;
                    }
                }
            }
        }
    });
}

}  // namespace cpu
}  // namespace lcs
}  // namespace d2p
