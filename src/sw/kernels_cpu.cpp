// SPDX-License-Identifier: Apache-2.0
// soft_sw_regular_cpu.cpp
//
// CPU implementation of Soft Smith-Waterman (regular/linear gap penalty)
// Mirrors the CUDA kernel interface exactly for seamless dispatch.
//
// Operations:
//   - Forward:  Compute partition function S = logsumexp over all alignments
//   - Backward: Compute all gradients (dS/dscores, dS/dgap, dS/dT)
//   - HVP:      Hessian-vector product d^2S/dscores^2 * V
//   - ParamGrad: Cross-derivatives dP/dtheta for gap, temperature
//
// Shapes:
//   scores:      [B, L1, L2]       - similarity scores
//   alpha:       [B, (L1+1)*(L2+1)] - DP table (row-major, 1-indexed logic)
//   partition:   [B]               - partition function values
//   posteriors:  [B, L1, L2]       - alignment marginals (dS/dscores)
//   grad_gap:    [B]               - expected number of gap steps
//   grad_T:      [B]               - temperature gradient

#include <cmath>
#include <algorithm>
#include <limits>

#include <ATen/Parallel.h>

// Shared utilities
#include "common/numerics.h"

using namespace d2p::common;

// ============================================================================
// SW-Specific Helper Functions
// ============================================================================

// Clamp log-posterior to ensure posteriors <= 1
inline float clamp_log_posterior(float log_post) {
    return std::min(log_post, 0.0f);
}

// Safe softmax weight computation for 4 values
// Sets weight to 0 for -inf options to avoid 0/0
inline void softmax4_weights_cpu(float a, float b, float c, float d, float T,
                                  float& wa, float& wb, float& wc, float& wd) {
    float max_v = std::max({a, b, c, d});
    if (max_v <= NINF) {
        wa = wb = wc = wd = 0.0f;
        return;
    }

    // Use Kahan summation for precision
    KahanSum sum;

    // Set weight to 0 for -inf options before normalization
    if (a > NINF) { wa = safe_exp((a - max_v) / T); sum.add(wa); } else { wa = 0.0f; }
    if (b > NINF) { wb = safe_exp((b - max_v) / T); sum.add(wb); } else { wb = 0.0f; }
    if (c > NINF) { wc = safe_exp((c - max_v) / T); sum.add(wc); } else { wc = 0.0f; }
    if (d > NINF) { wd = safe_exp((d - max_v) / T); sum.add(wd); } else { wd = 0.0f; }

    float total = sum.result();
    if (total > 0) {
        wa /= total; wb /= total; wc /= total; wd /= total;
    }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

extern "C" void sw_regular_forward_cpu(
    const float* scores,      // [B, L1, L2]
    float* alpha,             // [B, (L1+1)*(L2+1)]
    float* partition,         // [B]
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = scores + b * score_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize alpha: alpha[0,0] = 0, all others = -inf
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = NINF;
            }
            a[0] = 0.0f;

            // Forward DP: process anti-diagonals
            // k = i + j ranges from 2 to L1 + L2
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float score = s[score_idx];

                    // Four transitions
                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;  // Local alignment: can start fresh

                    a[idx] = logsumexp4(v_align, v_up, v_left, v_sky, T);
                }
            }

            // Compute partition function: logsumexp over all alpha values
            // Fix #3: LogSumExp with max-subtraction trick
            float max_alpha = NINF;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    max_alpha = std::max(max_alpha, a[idx]);
                }
            }

            // Fix #2: Use Kahan summation for partition function computation
            KahanSum sum_exp;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    if (a[idx] > NINF) {  // Skip -inf values
                        sum_exp.add(safe_exp((a[idx] - max_alpha) / T));
                    }
                }
            }
            partition[b] = max_alpha + T * std::log(sum_exp.result());
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

extern "C" void sw_regular_backward_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* posteriors,        // [B, L1, L2] output
    float* grad_gap,          // [B] output
    float* grad_T,            // [B] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * score_stride;
            float S = partition[b];
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize posteriors to 0
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }

            // Initialize beta: beta[i,j] = exp((alpha[i,j] - S) / T) initially
            // Then propagate backwards
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = NINF;
            }

            // Terminal condition: beta[i,j] = exp((alpha[i,j] - S) / T)
            // In log space: log_beta[i,j] = (alpha[i,j] - S) / T ... but we work in prob space
            // Actually, we compute: beta[i,j] = dS/dalpha[i,j]

            // For partition = logsumexp(alpha), dS/dalpha[i,j] = exp((alpha[i,j] - S) / T)
            // Fix #10: Clamp log-posterior to ensure posteriors <= 1
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    float log_post = (a[idx] - S) / T;
                    log_post = clamp_log_posterior(log_post);  // Ensure <= 0
                    be[idx] = safe_exp(log_post);
                }
            }

            // Backward DP: process anti-diagonals in reverse
            float sum_gap_grad = 0.0f;
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float score = s[score_idx];
                    float beta_ij = be[idx];

                    // Fix #11: Skip invalid gradients (inf/0 positions)
                    if (std::isinf(beta_ij) || std::isnan(beta_ij) || beta_ij == 0.0f) {
                        continue;
                    }

                    // Compute softmax weights for the 4 transitions
                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;

                    float w_align, w_up, w_left, w_sky;
                    softmax4_weights_cpu(v_align, v_up, v_left, v_sky, T, w_align, w_up, w_left, w_sky);

                    // Accumulate posteriors (gradient w.r.t. score)
                    // score contributes via align and sky paths
                    post[score_idx] += beta_ij * (w_align + w_sky);

                    // Propagate beta to predecessors
                    if (i > 1 && j > 1 && w_align > 0.0f) {
                        be[idx_diag] += beta_ij * w_align;
                    }
                    if (i > 1 && w_up > 0.0f) {
                        be[idx_up] += beta_ij * w_up;
                        sum_gap_grad += beta_ij * w_up;  // gap used
                    }
                    if (j > 1 && w_left > 0.0f) {
                        be[idx_left] += beta_ij * w_left;
                        sum_gap_grad += beta_ij * w_left;  // gap used
                    }

                }
            }

            KahanSum match_score_sum;
            for (size_t idx = 0; idx < score_stride; ++idx) {
                match_score_sum.add(post[idx] * s[idx]);
            }

            float expected_total = match_score_sum.result() + sum_gap_grad * gap;
            grad_gap[b] = sum_gap_grad;
            grad_T[b] = (S - expected_total) / T;
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT (HVP)
//
// Computes d^2S/dscores^2 * V using forward-mode autodiff through the backward pass.
// ============================================================================

extern "C" void sw_regular_hvp_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    const float* V,           // [B, L1, L2] tangent vector
    float* d_alpha,           // [B, (L1+1)*(L2+1)] workspace
    float* d_partition,       // [B] workspace
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* d_beta,            // [B, (L1+1)*(L2+1)] workspace
    float* H_scores,          // [B, L1, L2] output: Hessian-vector product
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);

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
            float S = partition[b];
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
            // Compute d_alpha: tangent of alpha w.r.t. scores in direction V

            // d_alpha[0,0] = 0 (constant)
            da[0] = 0.0f;

            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float score = s[score_idx];
                    float v_ij = v[score_idx];

                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;

                    float w_align, w_up, w_left, w_sky;
                    softmax4_weights_cpu(v_align, v_up, v_left, v_sky, T, w_align, w_up, w_left, w_sky);

                    // Tangent contributions
                    float dv_align = (i > 1 && j > 1) ? da[idx_diag] + v_ij : 0.0f;
                    float dv_up = (i > 1) ? da[idx_up] : 0.0f;
                    float dv_left = (j > 1) ? da[idx_left] : 0.0f;
                    float dv_sky = v_ij;

                    da[idx] = w_align * dv_align + w_up * dv_up + w_left * dv_left + w_sky * dv_sky;
                }
            }

            // Compute d_partition using Kahan summation
            KahanSum dS_acc;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    // Fix #10: Clamp log-posterior
                    float log_post = clamp_log_posterior((a[idx] - S) / T);
                    float w = safe_exp(log_post);
                    dS_acc.add(w * da[idx]);
                }
            }
            float dS = dS_acc.result();
            d_partition[b] = dS;

            // =========== Backward pass (primal) ===========
            // Compute beta = dS/dalpha
            // Fix #10: Clamp log-posterior to ensure posteriors <= 1
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    float log_post = clamp_log_posterior((a[idx] - S) / T);
                    be[idx] = safe_exp(log_post);
                }
            }

            // =========== Backward tangent pass ===========
            // Compute d_beta: tangent of beta w.r.t. scores in direction V

            // Terminal condition: d_beta[i,j] = d/dV [exp((alpha[i,j] - S) / T)]
            //                                 = exp(...) * (d_alpha[i,j] - dS) / T
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    dbe[idx] = be[idx] * (da[idx] - dS) / T;
                }
            }

            // Backward through the DP in reverse order
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float score = s[score_idx];
                    float v_ij = v[score_idx];
                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];

                    // Fix #11: Skip invalid gradients
                    if (std::isinf(beta_ij) || std::isnan(beta_ij) || beta_ij == 0.0f) {
                        continue;
                    }

                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;

                    float w_align, w_up, w_left, w_sky;
                    softmax4_weights_cpu(v_align, v_up, v_left, v_sky, T, w_align, w_up, w_left, w_sky);

                    // Tangent of weights (Jacobian of softmax)
                    float dv_align = (i > 1 && j > 1) ? da[idx_diag] + v_ij : 0.0f;
                    float dv_up = (i > 1) ? da[idx_up] : 0.0f;
                    float dv_left = (j > 1) ? da[idx_left] : 0.0f;
                    float dv_sky = v_ij;

                    float sum_w_dv = w_align * dv_align + w_up * dv_up + w_left * dv_left + w_sky * dv_sky;

                    float dw_align = w_align * (dv_align - sum_w_dv) / T;
                    float dw_up = w_up * (dv_up - sum_w_dv) / T;
                    float dw_left = w_left * (dv_left - sum_w_dv) / T;
                    float dw_sky = w_sky * (dv_sky - sum_w_dv) / T;

                    // HVP contribution: d(posteriors)/dV
                    // posteriors[score_idx] += beta_ij * (w_align + w_sky)
                    // d(posteriors) = dbeta_ij * (w_align + w_sky) + beta_ij * (dw_align + dw_sky)
                    H[score_idx] += dbeta_ij * (w_align + w_sky) + beta_ij * (dw_align + dw_sky);

                    // Propagate beta to predecessors (same as regular backward)
                    // This was missing - needed for correct HVP computation
                    if (i > 1 && j > 1 && w_align > 0.0f) {
                        be[idx_diag] += beta_ij * w_align;
                    }
                    if (i > 1 && w_up > 0.0f) {
                        be[idx_up] += beta_ij * w_up;
                    }
                    if (j > 1 && w_left > 0.0f) {
                        be[idx_left] += beta_ij * w_left;
                    }

                    // Propagate d_beta to predecessors
                    if (i > 1 && j > 1 && w_align > 0.0f) {
                        dbe[idx_diag] += dbeta_ij * w_align + beta_ij * dw_align;
                    }
                    if (i > 1 && w_up > 0.0f) {
                        dbe[idx_up] += dbeta_ij * w_up + beta_ij * dw_up;
                    }
                    if (j > 1 && w_left > 0.0f) {
                        dbe[idx_left] += dbeta_ij * w_left + beta_ij * dw_left;
                    }
                }
            }
        }
    });
}

// ============================================================================
// PARAMETER GRADIENT
//
// Computes dP/dtheta where P = posteriors and theta  in  {gap, temperature}
// ============================================================================

// Parameter type enum (matches CUDA)
enum ParamType {
    PARAM_GAP = 0,
    PARAM_TEMPERATURE = 1
};

extern "C" void sw_regular_param_grad_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    const float* dS_dtheta,   // [B] pre-computed dS/dtheta from backward
    float* U,                 // [B, (L1+1)*(L2+1)] workspace
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* W,                 // [B, (L1+1)*(L2+1)] workspace
    float* dP_dtheta,         // [B, L1, L2] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T,
    int param_type
) {
    const size_t alpha_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w = W + b * alpha_stride;
            float* dP = dP_dtheta + b * score_stride;
            float S = partition[b];
            float dS_dt = dS_dtheta[b];
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                u[idx] = 0.0f;
                be[idx] = 0.0f;
                w[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                dP[idx] = 0.0f;
            }

            // =========== Forward pass for U: dalpha/dtheta ===========
            // U[0,0] = 0 (constant)
            u[0] = 0.0f;

            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float score = s[score_idx];

                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;

                    float max_v = std::max({v_align, v_up, v_left, v_sky});
                    if (max_v <= NINF) {
                        u[idx] = 0.0f;
                        continue;
                    }

                    float w_align = safe_exp((v_align - max_v) / T);
                    float w_up = safe_exp((v_up - max_v) / T);
                    float w_left = safe_exp((v_left - max_v) / T);
                    float w_sky = safe_exp((v_sky - max_v) / T);
                    float sum_w = w_align + w_up + w_left + w_sky;
                    float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

                    w_align *= inv_sum;
                    w_up *= inv_sum;
                    w_left *= inv_sum;
                    w_sky *= inv_sum;

                    float du_align = (i > 1 && j > 1) ? u[idx_diag] : 0.0f;
                    float du_up = (i > 1) ? u[idx_up] : 0.0f;
                    float du_left = (j > 1) ? u[idx_left] : 0.0f;
                    float du_sky = 0.0f;

                    if (param_type == PARAM_GAP) {
                        du_up += 1.0f;
                        du_left += 1.0f;
                    }

                    float u_val =
                        w_align * du_align + w_up * du_up + w_left * du_left + w_sky * du_sky;

                    if (param_type == PARAM_TEMPERATURE) {
                        float alpha_ij = a[idx];
                        float mean_v = w_align * v_align + w_up * v_up + w_left * v_left + w_sky * v_sky;
                        u_val += (alpha_ij - mean_v) / T;
                    }

                    u[idx] = u_val;
                }
            }

            // =========== Compute beta ===========
            // Fix #10: Clamp log-posterior to ensure posteriors <= 1
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    float log_post = clamp_log_posterior((a[idx] - S) / T);
                    be[idx] = safe_exp(log_post);
                }
            }

            // =========== Backward pass for dP/dtheta ===========
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t idx_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t idx_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * score_cols + static_cast<size_t>(j - 1);

                    float beta_curr = be[idx];
                    float beta_init = 0.0f;
                    if (a[idx] > NINF) {
                        float log_post = clamp_log_posterior((a[idx] - S) / T);
                        beta_init = safe_exp(log_post);
                    }

                    float dW_init = beta_init * (u[idx] - dS_dt) / T;
                    if (param_type == PARAM_TEMPERATURE && a[idx] > NINF) {
                        dW_init -= beta_init * (a[idx] - S) / (T * T);
                    }

                    float w_curr = w[idx] + dW_init;

                    if (beta_curr <= 1e-20f && std::abs(w_curr) <= 1e-20f) {
                        continue;
                    }

                    float score = s[score_idx];
                    float v_align = (i > 1 && j > 1) ? a[idx_diag] + score : NINF;
                    float v_up = (i > 1) ? a[idx_up] + gap : NINF;
                    float v_left = (j > 1) ? a[idx_left] + gap : NINF;
                    float v_sky = score;

                    float max_v = std::max({v_align, v_up, v_left, v_sky});
                    if (max_v <= NINF) {
                        continue;
                    }

                    float w_align = safe_exp((v_align - max_v) / T);
                    float w_up = safe_exp((v_up - max_v) / T);
                    float w_left = safe_exp((v_left - max_v) / T);
                    float w_sky = safe_exp((v_sky - max_v) / T);
                    float sum_w = w_align + w_up + w_left + w_sky;
                    float inv_sum = (sum_w > 1e-20f) ? 1.0f / sum_w : 0.0f;

                    w_align *= inv_sum;
                    w_up *= inv_sum;
                    w_left *= inv_sum;
                    w_sky *= inv_sum;

                    float du_align = (i > 1 && j > 1) ? u[idx_diag] : 0.0f;
                    float du_up = (i > 1) ? u[idx_up] : 0.0f;
                    float du_left = (j > 1) ? u[idx_left] : 0.0f;
                    float du_sky = 0.0f;

                    if (param_type == PARAM_GAP) {
                        du_up += 1.0f;
                        du_left += 1.0f;
                    }

                    float mean_du = w_align * du_align + w_up * du_up + w_left * du_left + w_sky * du_sky;

                    float dw_align = w_align * (du_align - mean_du) / T;
                    float dw_up = w_up * (du_up - mean_du) / T;
                    float dw_left = w_left * (du_left - mean_du) / T;
                    float dw_sky = w_sky * (du_sky - mean_du) / T;

                    if (param_type == PARAM_TEMPERATURE) {
                        float mean_v = w_align * v_align + w_up * v_up + w_left * v_left + w_sky * v_sky;
                        float inv_T2 = 1.0f / (T * T);
                        dw_align += w_align * (mean_v - v_align) * inv_T2;
                        dw_up += w_up * (mean_v - v_up) * inv_T2;
                        dw_left += w_left * (mean_v - v_left) * inv_T2;
                        dw_sky += w_sky * (mean_v - v_sky) * inv_T2;
                    }

                    dP[score_idx] += w_curr * (w_align + w_sky) + beta_curr * (dw_align + dw_sky);

                    if (v_align > NINF) {
                        be[idx_diag] += beta_curr * w_align;
                        w[idx_diag] += w_curr * w_align + beta_curr * dw_align;
                    }
                    if (v_up > NINF) {
                        be[idx_up] += beta_curr * w_up;
                        w[idx_up] += w_curr * w_up + beta_curr * dw_up;
                    }
                    if (v_left > NINF) {
                        be[idx_left] += beta_curr * w_left;
                        w[idx_left] += w_curr * w_left + beta_curr * dw_left;
                    }
                }
            }
        }
    });
}
