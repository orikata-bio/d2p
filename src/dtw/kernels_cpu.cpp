// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft DTW CPU Kernel Implementations
 *
 * CPU implementation of DTW using softmin (minimization).
 * Uses Kahan summation for numerical precision.
 */

#include "kernels_cpu.h"
#include "common/numerics.h"
#include <cmath>
#include <algorithm>

#include <ATen/Parallel.h>

using namespace d2p::common;

// ============================================================================
// DTW-specific helpers
// ============================================================================

// Check if cell (i, j) is within Sakoe-Chiba band
inline bool in_band(int i, int j, int L1, int L2, int64_t bandwidth) {
    if (bandwidth < 0) return true;  // No band constraint
    if (L1 == 0 || L2 == 0) return true;
    float expected_j = static_cast<float>(static_cast<int64_t>(i) * L2) / L1;
    return std::abs(j - expected_j) <= bandwidth;
}

// Softmin weights for 3 values
inline void softmin3_weights_cpu(float a, float b, float c, float T,
                                  float& wa, float& wb, float& wc) {
    float m = std::min({a, b, c});
    if (m >= PINF) {
        wa = wb = wc = 0.0f;
        return;
    }

    float ea = (a < PINF) ? safe_exp((m - a) / T) : 0.0f;
    float eb = (b < PINF) ? safe_exp((m - b) / T) : 0.0f;
    float ec = (c < PINF) ? safe_exp((m - c) / T) : 0.0f;

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

extern "C" void dtw_forward_cpu(
    const float* costs,
    float* alpha,
    float* score,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T, int64_t bandwidth
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t cost_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* c = costs + b * cost_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize alpha: alpha[0,0] = 0, all others = +INF
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = PINF;
            }
            a[0] = 0.0f;

            // Forward DP: process anti-diagonals
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float cost = c[cost_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    // Check bandwidth for predecessors
                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    // DTW recurrence: alpha[i,j] = cost[i,j] + softmin(predecessors)
                    a[idx] = cost + softmin3(a_diag, a_up, a_left, T);
                }
            }

            // Score is just the final cell
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            score[b] = a[final_idx];
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

extern "C" void dtw_backward_cpu(
    const float* alpha,
    const float* costs,
    const float* score,
    float* beta,
    float* posteriors,
    float* grad_T,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T, int64_t bandwidth
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t cost_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* c = costs + b * cost_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * cost_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize posteriors and beta to 0
            for (size_t idx = 0; idx < cost_stride; idx++) {
                post[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = 0.0f;
            }

            // Terminal condition: beta[L1, L2] = 1
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;

            // Backward DP: process anti-diagonals in reverse
            float sum_T_grad = 0.0f;

            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float beta_ij = be[idx];
                    if (beta_ij == 0.0f) continue;

                    // For DTW, posteriors = beta
                    post[cost_idx] = beta_ij;

                    // Compute softmin weights
                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    float w_diag, w_up, w_left;
                    softmin3_weights_cpu(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

                    // Propagate beta to predecessors
                    if (in_band(i-1, j-1, L1, L2, bandwidth) && w_diag > 0.0f) {
                        be[idx_diag] += beta_ij * w_diag;
                    }
                    if (in_band(i-1, j, L1, L2, bandwidth) && w_up > 0.0f) {
                        be[idx_up] += beta_ij * w_up;
                    }
                    if (in_band(i, j-1, L1, L2, bandwidth) && w_left > 0.0f) {
                        be[idx_left] += beta_ij * w_left;
                    }

                    // Temperature gradient
                    float softmin_val = a[idx] - c[cost_idx];
                    if (softmin_val < PINF) {
                        float E_a = w_diag * a_diag + w_up * a_up + w_left * a_left;
                        sum_T_grad += beta_ij * (softmin_val - E_a) / T;
                    }
                }
            }

            grad_T[b] = sum_T_grad;
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT
// ============================================================================

extern "C" void dtw_hvp_cpu(
    const float* alpha,
    const float* costs,
    const float* score,
    const float* V,
    float* d_alpha,
    float* d_score,
    float* beta,
    float* d_beta,
    float* H_costs,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T, int64_t bandwidth
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t cost_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* v = V + b * cost_stride;
            float* da = d_alpha + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* dbe = d_beta + b * alpha_stride;
            float* H = H_costs + b * cost_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize workspaces
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                da[idx] = 0.0f;
                be[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < cost_stride; idx++) {
                H[idx] = 0.0f;
            }

            // Forward tangent pass
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float v_ij = v[cost_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    float w_diag, w_up, w_left;
                    softmin3_weights_cpu(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

                    float da_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? da[idx_diag] : 0.0f;
                    float da_up = in_band(i-1, j, L1, L2, bandwidth) ? da[idx_up] : 0.0f;
                    float da_left = in_band(i, j-1, L1, L2, bandwidth) ? da[idx_left] : 0.0f;

                    da[idx] = v_ij + w_diag * da_diag + w_up * da_up + w_left * da_left;
                }
            }

            // d_score = d_alpha[L1, L2]
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            d_score[b] = da[final_idx];

            // Initialize beta
            be[final_idx] = 1.0f;
            dbe[final_idx] = 0.0f;

            // Backward tangent pass
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];

                    if (beta_ij == 0.0f) continue;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    float w_diag, w_up, w_left;
                    softmin3_weights_cpu(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

                    // Weight tangents
                    float da_diag_v = in_band(i-1, j-1, L1, L2, bandwidth) ? da[idx_diag] : 0.0f;
                    float da_up_v = in_band(i-1, j, L1, L2, bandwidth) ? da[idx_up] : 0.0f;
                    float da_left_v = in_band(i, j-1, L1, L2, bandwidth) ? da[idx_left] : 0.0f;

                    float E_da = w_diag * da_diag_v + w_up * da_up_v + w_left * da_left_v;

                    float dw_diag = w_diag * (-da_diag_v + E_da) / T;
                    float dw_up = w_up * (-da_up_v + E_da) / T;
                    float dw_left = w_left * (-da_left_v + E_da) / T;

                    // HVP output
                    H[cost_idx] = dbeta_ij;

                    // Propagate beta and dbeta
                    if (in_band(i-1, j-1, L1, L2, bandwidth) && w_diag > 0.0f) {
                        be[idx_diag] += beta_ij * w_diag;
                        dbe[idx_diag] += dbeta_ij * w_diag + beta_ij * dw_diag;
                    }
                    if (in_band(i-1, j, L1, L2, bandwidth) && w_up > 0.0f) {
                        be[idx_up] += beta_ij * w_up;
                        dbe[idx_up] += dbeta_ij * w_up + beta_ij * dw_up;
                    }
                    if (in_band(i, j-1, L1, L2, bandwidth) && w_left > 0.0f) {
                        be[idx_left] += beta_ij * w_left;
                        dbe[idx_left] += dbeta_ij * w_left + beta_ij * dw_left;
                    }
                }
            }
        }
    });
}

// ============================================================================
// PARAMETER GRADIENT
// ============================================================================

extern "C" void dtw_param_grad_cpu(
    const float* alpha,
    const float* costs,
    const float* score,
    float* U,
    float* beta,
    float* W,
    float* dP_dT,
    const int* lengths,
    int B, int max_L1, int max_L2,
    float T, int64_t bandwidth
) {
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t cost_stride = static_cast<size_t>(max_L1) * static_cast<size_t>(max_L2);
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* c = costs + b * cost_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w = W + b * alpha_stride;
            float* dPdT = dP_dT + b * cost_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize workspaces
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                u[idx] = 0.0f;
                be[idx] = 0.0f;
                w[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < cost_stride; idx++) {
                dPdT[idx] = 0.0f;
            }

            // Forward U pass
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float cost = c[cost_idx];

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    float w_diag, w_up, w_left;
                    softmin3_weights_cpu(a_diag, a_up, a_left, T, w_diag, w_up, w_left);

                    float u_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? u[idx_diag] : 0.0f;
                    float u_up = in_band(i-1, j, L1, L2, bandwidth) ? u[idx_up] : 0.0f;
                    float u_left = in_band(i, j-1, L1, L2, bandwidth) ? u[idx_left] : 0.0f;

                    float softmin_val = a[idx] - cost;
                    float E_a = w_diag * a_diag + w_up * a_up + w_left * a_left;
                    float dSoftmin_dT = (softmin_val - E_a) / T + w_diag * u_diag + w_up * u_up + w_left * u_left;

                    u[idx] = dSoftmin_dT;
                }
            }

            // Initialize beta
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;
            w[final_idx] = 0.0f;

            // Backward W pass
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;
                    if (!in_band(i, j, L1, L2, bandwidth)) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cost_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float beta_ij = be[idx];
                    float W_ij = w[idx];

                    if (beta_ij == 0.0f && std::abs(W_ij) < 1e-20f) continue;

                    // Posteriors = beta for DTW
                    dPdT[cost_idx] = W_ij;

                    float a_diag = a[idx_diag];
                    float a_up = a[idx_up];
                    float a_left = a[idx_left];

                    if (!in_band(i-1, j-1, L1, L2, bandwidth)) a_diag = PINF;
                    if (!in_band(i-1, j, L1, L2, bandwidth)) a_up = PINF;
                    if (!in_band(i, j-1, L1, L2, bandwidth)) a_left = PINF;

                    float wt_diag, wt_up, wt_left;
                    softmin3_weights_cpu(a_diag, a_up, a_left, T, wt_diag, wt_up, wt_left);

                    float u_diag = in_band(i-1, j-1, L1, L2, bandwidth) ? u[idx_diag] : 0.0f;
                    float u_up = in_band(i-1, j, L1, L2, bandwidth) ? u[idx_up] : 0.0f;
                    float u_left = in_band(i, j-1, L1, L2, bandwidth) ? u[idx_left] : 0.0f;

                    // Total derivative of the softmin weights w.r.t. temperature:
                    // the predecessor alphas depend on T through U, and the
                    // exp(-a / T) scaling contributes the explicit T term.
                    float E_a = wt_diag * a_diag + wt_up * a_up + wt_left * a_left;
                    float E_u = wt_diag * u_diag + wt_up * u_up + wt_left * u_left;
                    float inv_T = 1.0f / T;
                    float inv_T2 = inv_T * inv_T;
                    float dw_diag_dT = wt_diag * ((E_u - u_diag) * inv_T + (a_diag - E_a) * inv_T2);
                    float dw_up_dT = wt_up * ((E_u - u_up) * inv_T + (a_up - E_a) * inv_T2);
                    float dw_left_dT = wt_left * ((E_u - u_left) * inv_T + (a_left - E_a) * inv_T2);

                    // Propagate beta and W
                    if (in_band(i-1, j-1, L1, L2, bandwidth) && wt_diag > 0.0f) {
                        be[idx_diag] += beta_ij * wt_diag;
                        w[idx_diag] += W_ij * wt_diag + beta_ij * dw_diag_dT;
                    }
                    if (in_band(i-1, j, L1, L2, bandwidth) && wt_up > 0.0f) {
                        be[idx_up] += beta_ij * wt_up;
                        w[idx_up] += W_ij * wt_up + beta_ij * dw_up_dT;
                    }
                    if (in_band(i, j-1, L1, L2, bandwidth) && wt_left > 0.0f) {
                        be[idx_left] += beta_ij * wt_left;
                        w[idx_left] += W_ij * wt_left + beta_ij * dw_left_dT;
                    }
                }
            }
        }
    });
}
