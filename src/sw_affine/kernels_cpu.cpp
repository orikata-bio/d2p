// SPDX-License-Identifier: Apache-2.0
// soft_sw_affine_cpu.cpp
//
// CPU implementation of Soft Smith-Waterman with affine gap penalties
// Mirrors the CUDA kernel interface exactly for seamless dispatch.
//
// Three-state DP: M (match), I (gap in seq1), D (gap in seq2)
//
// Operations:
//   - Forward:  Compute partition function S = logsumexp over all alignments
//   - Backward: Compute all gradients (dS/dscores, dS/dgap_open, dS/dgap_ext, dS/dT)
//   - HVP:      Hessian-vector product d^2S/dscores^2 * V
//   - ParamGrad: Cross-derivatives dP/dtheta for gap_open, gap_ext, temperature
//
// Shapes:
//   scores:      [B, L1, L2]           - similarity scores
//   alpha:       [B, 3*(L1+1)*(L2+1)]  - DP table (3 states: M, I, D)
//   partition:   [B]                   - partition function values
//   posteriors:  [B, L1, L2]           - alignment marginals (dS/dscores)

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

#include <ATen/Parallel.h>

// Shared utilities
#include "common/numerics.h"

using namespace d2p::common;

// ============================================================================
// SW-Specific Constants and Helper Functions
// ============================================================================

// State indices
constexpr int STATE_M = 0;  // Match
constexpr int STATE_I = 1;  // Insert (gap in seq2, moving down)
constexpr int STATE_D = 2;  // Delete (gap in seq1, moving right)

// Clamp log-posterior to ensure posteriors <= 1
inline float clamp_log_posterior(float log_post) {
    return std::min(log_post, 0.0f);
}

inline float clamped_posterior_weight(float alpha, float partition, float T) {
    if (alpha <= NINF) {
        return 0.0f;
    }
    return safe_exp(clamp_log_posterior((alpha - partition) / T));
}

// Safe softmax weight computation for 2 values
// Sets weight to 0 for -inf options to avoid 0/0
inline void softmax2_weights_cpu(float a, float b, float T, float& wa, float& wb) {
    float max_v = std::max(a, b);
    if (max_v <= NINF) { wa = wb = 0.0f; return; }
    KahanSum sum;
    if (a > NINF) { wa = safe_exp((a - max_v) / T); sum.add(wa); } else { wa = 0.0f; }
    if (b > NINF) { wb = safe_exp((b - max_v) / T); sum.add(wb); } else { wb = 0.0f; }
    float total = sum.result();
    if (total > 0) { wa /= total; wb /= total; }
}

inline void softmax3_weights_cpu(float a, float b, float c, float T, float& wa, float& wb, float& wc) {
    float max_v = std::max({a, b, c});
    if (max_v <= NINF) { wa = wb = wc = 0.0f; return; }
    KahanSum sum;
    if (a > NINF) { wa = safe_exp((a - max_v) / T); sum.add(wa); } else { wa = 0.0f; }
    if (b > NINF) { wb = safe_exp((b - max_v) / T); sum.add(wb); } else { wb = 0.0f; }
    if (c > NINF) { wc = safe_exp((c - max_v) / T); sum.add(wc); } else { wc = 0.0f; }
    float total = sum.result();
    if (total > 0) { wa /= total; wb /= total; wc /= total; }
}

inline void softmax4_weights_cpu(float a, float b, float c, float d, float T,
                                  float& wa, float& wb, float& wc, float& wd) {
    float max_v = std::max({a, b, c, d});
    if (max_v <= NINF) { wa = wb = wc = wd = 0.0f; return; }
    KahanSum sum;
    if (a > NINF) { wa = safe_exp((a - max_v) / T); sum.add(wa); } else { wa = 0.0f; }
    if (b > NINF) { wb = safe_exp((b - max_v) / T); sum.add(wb); } else { wb = 0.0f; }
    if (c > NINF) { wc = safe_exp((c - max_v) / T); sum.add(wc); } else { wc = 0.0f; }
    if (d > NINF) { wd = safe_exp((d - max_v) / T); sum.add(wd); } else { wd = 0.0f; }
    float total = sum.result();
    if (total > 0) { wa /= total; wb /= total; wc /= total; wd /= total; }
}

// ============================================================================
// FORWARD PASS
// ============================================================================

extern "C" void sw_affine_forward_cpu(
    const float* scores,      // [B, L1, L2]
    float* alpha,             // [B, 3*(L1+1)*(L2+1)]
    float* partition,         // [B]
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t cell_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t alpha_stride = 3 * cell_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = scores + b * score_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize all states to -inf
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = NINF;
            }

            // Alpha[0,0,M] = 0 (can start from here)
            a[0 * cell_stride + 0] = 0.0f;

            // Forward DP: process anti-diagonals
            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float score = s[score_idx];

                    // M state: match/mismatch
                    // Can come from: M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], or fresh start (sky)
                    float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                    float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                    float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                    float m_sky = score;  // Local alignment: can start fresh

                    a[STATE_M * cell_stride + cell_idx] = logsumexp4(m_from_M, m_from_I, m_from_D, m_sky, T);

                    // I state: gap in seq2 (moving down, i increases)
                    // Can come from: M[i-1,j] + gap_open, I[i-1,j] + gap_ext
                    float i_from_M = (i > 1) ? a[STATE_M * cell_stride + cell_up] + gap_open : NINF;
                    float i_from_I = (i > 1) ? a[STATE_I * cell_stride + cell_up] + gap_ext : NINF;

                    a[STATE_I * cell_stride + cell_idx] = logsumexp2(i_from_M, i_from_I, T);

                    // D state: gap in seq1 (moving right, j increases)
                    // Can come from: M[i,j-1] + gap_open, D[i,j-1] + gap_ext
                    float d_from_M = (j > 1) ? a[STATE_M * cell_stride + cell_left] + gap_open : NINF;
                    float d_from_D = (j > 1) ? a[STATE_D * cell_stride + cell_left] + gap_ext : NINF;

                    a[STATE_D * cell_stride + cell_idx] = logsumexp2(d_from_M, d_from_D, T);
                }
            }

            // Compute partition function: logsumexp over all M state values
            // (alignments end in match state)
            // Fix #3: LogSumExp with max-subtraction trick
            float max_alpha = NINF;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    // Sum over all states for local alignment
                    for (int state = 0; state < 3; state++) {
                        max_alpha = std::max(max_alpha, a[state * cell_stride + cell_idx]);
                    }
                }
            }

            // Fix #2: Use Kahan summation for partition function computation
            KahanSum sum_exp;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        float val = a[state * cell_stride + cell_idx];
                        if (val > NINF) {  // Skip -inf values
                            sum_exp.add(safe_exp((val - max_alpha) / T));
                        }
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

extern "C" void sw_affine_backward_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    float* beta,              // [B, 3*(L1+1)*(L2+1)] workspace
    float* posteriors,        // [B, L1, L2] output
    float* grad_open,         // [B] output
    float* grad_ext,          // [B] output
    float* grad_T,            // [B] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t cell_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t alpha_stride = 3 * cell_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

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

            // Initialize
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                be[idx] = 0.0f;
            }

            // Terminal condition: beta[i,j,state] = exp((alpha[i,j,state] - S) / T)
            // Fix #10: Clamp log-posterior to ensure posteriors <= 1
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        float log_post = (a[state * cell_stride + cell_idx] - S) / T;
                        log_post = clamp_log_posterior(log_post);  // Ensure <= 0
                        be[state * cell_stride + cell_idx] = safe_exp(log_post);
                    }
                }
            }

            // Backward DP
            float sum_open_grad = 0.0f;
            float sum_ext_grad = 0.0f;

            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float score = s[score_idx];

                    // ========== M state backward ==========
                    float beta_M = be[STATE_M * cell_stride + cell_idx];

                    // Fix #11: Skip invalid gradients (inf/0 positions)
                    if (!std::isinf(beta_M) && !std::isnan(beta_M) && beta_M != 0.0f) {
                        float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                        float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                        float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                        float m_sky = score;

                        float wM_M, wM_I, wM_D, wM_sky;
                        softmax4_weights_cpu(m_from_M, m_from_I, m_from_D, m_sky, T, wM_M, wM_I, wM_D, wM_sky);

                        // Posteriors: score contributes via all paths to M
                        post[score_idx] += beta_M * (wM_M + wM_I + wM_D + wM_sky);

                        // Propagate beta
                        if (i > 1 && j > 1) {
                            if (wM_M > 0.0f) be[STATE_M * cell_stride + cell_diag] += beta_M * wM_M;
                            if (wM_I > 0.0f) be[STATE_I * cell_stride + cell_diag] += beta_M * wM_I;
                            if (wM_D > 0.0f) be[STATE_D * cell_stride + cell_diag] += beta_M * wM_D;
                        }
                    }

                    // ========== I state backward ==========
                    float beta_I = be[STATE_I * cell_stride + cell_idx];

                    // Fix #11: Skip invalid gradients
                    if (!std::isinf(beta_I) && !std::isnan(beta_I) && beta_I != 0.0f) {
                        float i_from_M = (i > 1) ? a[STATE_M * cell_stride + cell_up] + gap_open : NINF;
                        float i_from_I = (i > 1) ? a[STATE_I * cell_stride + cell_up] + gap_ext : NINF;

                        float wI_M, wI_I;
                        softmax2_weights_cpu(i_from_M, i_from_I, T, wI_M, wI_I);

                        if (i > 1) {
                            if (wI_M > 0.0f) {
                                be[STATE_M * cell_stride + cell_up] += beta_I * wI_M;
                                sum_open_grad += beta_I * wI_M;
                            }
                            if (wI_I > 0.0f) {
                                be[STATE_I * cell_stride + cell_up] += beta_I * wI_I;
                                sum_ext_grad += beta_I * wI_I;
                            }
                        }
                    }

                    // ========== D state backward ==========
                    float beta_D = be[STATE_D * cell_stride + cell_idx];

                    // Fix #11: Skip invalid gradients
                    if (!std::isinf(beta_D) && !std::isnan(beta_D) && beta_D != 0.0f) {
                        float d_from_M = (j > 1) ? a[STATE_M * cell_stride + cell_left] + gap_open : NINF;
                        float d_from_D = (j > 1) ? a[STATE_D * cell_stride + cell_left] + gap_ext : NINF;

                        float wD_M, wD_D;
                        softmax2_weights_cpu(d_from_M, d_from_D, T, wD_M, wD_D);

                        if (j > 1) {
                            if (wD_M > 0.0f) {
                                be[STATE_M * cell_stride + cell_left] += beta_D * wD_M;
                                sum_open_grad += beta_D * wD_M;
                            }
                            if (wD_D > 0.0f) {
                                be[STATE_D * cell_stride + cell_left] += beta_D * wD_D;
                                sum_ext_grad += beta_D * wD_D;
                            }
                        }
                    }
                }
            }

            KahanSum match_score_sum;
            for (size_t idx = 0; idx < score_stride; ++idx) {
                match_score_sum.add(post[idx] * s[idx]);
            }

            float expected_gap_cost = sum_open_grad * gap_open + sum_ext_grad * gap_ext;
            float expected_total = match_score_sum.result() + expected_gap_cost;

            grad_open[b] = sum_open_grad;
            grad_ext[b] = sum_ext_grad;
            grad_T[b] = (S - expected_total) / T;
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT (HVP)
// ============================================================================

extern "C" void sw_affine_hvp_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    const float* V,           // [B, L1, L2] tangent vector
    float* d_alpha,           // [B, 3*(L1+1)*(L2+1)] workspace
    float* d_partition,       // [B] workspace
    float* d_beta,            // [B, 3*(L1+1)*(L2+1)] workspace
    float* H_scores,          // [B, L1, L2] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t cell_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t alpha_stride = 3 * cell_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            const float* v = V + b * score_stride;
            float* da = d_alpha + b * alpha_stride;
            float* dbe = d_beta + b * alpha_stride;
            float* H = H_scores + b * score_stride;
            float S = partition[b];
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                da[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                H[idx] = 0.0f;
            }

            // =========== Forward tangent pass ===========
            da[STATE_M * cell_stride + 0] = 0.0f;

            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float score = s[score_idx];
                    float v_ij = v[score_idx];

                    // M state tangent
                    float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                    float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                    float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                    float m_sky = score;

                    float wM_M, wM_I, wM_D, wM_sky;
                    softmax4_weights_cpu(m_from_M, m_from_I, m_from_D, m_sky, T, wM_M, wM_I, wM_D, wM_sky);

                    float dm_from_M = (i > 1 && j > 1) ? da[STATE_M * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_from_I = (i > 1 && j > 1) ? da[STATE_I * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_from_D = (i > 1 && j > 1) ? da[STATE_D * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_sky = v_ij;

                    da[STATE_M * cell_stride + cell_idx] = wM_M * dm_from_M + wM_I * dm_from_I +
                                                            wM_D * dm_from_D + wM_sky * dm_sky;

                    // I state tangent
                    float i_from_M = (i > 1) ? a[STATE_M * cell_stride + cell_up] + gap_open : NINF;
                    float i_from_I = (i > 1) ? a[STATE_I * cell_stride + cell_up] + gap_ext : NINF;

                    float wI_M, wI_I;
                    softmax2_weights_cpu(i_from_M, i_from_I, T, wI_M, wI_I);

                    float di_from_M = (i > 1) ? da[STATE_M * cell_stride + cell_up] : 0.0f;
                    float di_from_I = (i > 1) ? da[STATE_I * cell_stride + cell_up] : 0.0f;

                    da[STATE_I * cell_stride + cell_idx] = wI_M * di_from_M + wI_I * di_from_I;

                    // D state tangent
                    float d_from_M = (j > 1) ? a[STATE_M * cell_stride + cell_left] + gap_open : NINF;
                    float d_from_D = (j > 1) ? a[STATE_D * cell_stride + cell_left] + gap_ext : NINF;

                    float wD_M, wD_D;
                    softmax2_weights_cpu(d_from_M, d_from_D, T, wD_M, wD_D);

                    float dd_from_M = (j > 1) ? da[STATE_M * cell_stride + cell_left] : 0.0f;
                    float dd_from_D = (j > 1) ? da[STATE_D * cell_stride + cell_left] : 0.0f;

                    da[STATE_D * cell_stride + cell_idx] = wD_M * dd_from_M + wD_D * dd_from_D;
                }
            }

            // Compute d_partition using Kahan summation
            KahanSum dS_acc;
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        // Fix #10: Clamp log-posterior
                        float log_post = clamp_log_posterior((a[state * cell_stride + cell_idx] - S) / T);
                        float w = safe_exp(log_post);
                        dS_acc.add(w * da[state * cell_stride + cell_idx]);
                    }
                }
            }
            float dS = dS_acc.result();
            d_partition[b] = dS;

            // Batch-local scratch keeps the HVP beta pass disjoint across threads.
            std::vector<float> beta_storage(alpha_stride, 0.0f);
            float* be = beta_storage.data();

            // First, initialize beta with terminal condition
            // Fix #10: Clamp log-posterior to ensure posteriors <= 1
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        size_t idx = state * cell_stride + cell_idx;
                        float log_post = clamp_log_posterior((a[idx] - S) / T);
                        be[idx] = safe_exp(log_post);
                    }
                }
            }

            // Initialize d_beta from terminal condition
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        size_t idx = state * cell_stride + cell_idx;
                        dbe[idx] = be[idx] * (da[idx] - dS) / T;
                    }
                }
            }

            // =========== Backward tangent pass ===========
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float score = s[score_idx];
                    float v_ij = v[score_idx];

                    // ========== M state HVP ==========
                    float beta_M = be[STATE_M * cell_stride + cell_idx];
                    float dbeta_M = dbe[STATE_M * cell_stride + cell_idx];

                    float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                    float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                    float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                    float m_sky = score;

                    float wM_M, wM_I, wM_D, wM_sky;
                    softmax4_weights_cpu(m_from_M, m_from_I, m_from_D, m_sky, T, wM_M, wM_I, wM_D, wM_sky);

                    // Tangent of M state inputs
                    float dm_from_M = (i > 1 && j > 1) ? da[STATE_M * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_from_I = (i > 1 && j > 1) ? da[STATE_I * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_from_D = (i > 1 && j > 1) ? da[STATE_D * cell_stride + cell_diag] + v_ij : 0.0f;
                    float dm_sky = v_ij;

                    float sum_w_dm = wM_M * dm_from_M + wM_I * dm_from_I + wM_D * dm_from_D + wM_sky * dm_sky;

                    float dwM_M = wM_M * (dm_from_M - sum_w_dm) / T;
                    float dwM_I = wM_I * (dm_from_I - sum_w_dm) / T;
                    float dwM_D = wM_D * (dm_from_D - sum_w_dm) / T;
                    float dwM_sky = wM_sky * (dm_sky - sum_w_dm) / T;

                    // HVP contribution from M state
                    H[score_idx] += dbeta_M * (wM_M + wM_I + wM_D + wM_sky) +
                                    beta_M * (dwM_M + dwM_I + dwM_D + dwM_sky);

                    // Propagate beta to predecessors (was missing!)
                    if (i > 1 && j > 1) {
                        be[STATE_M * cell_stride + cell_diag] += beta_M * wM_M;
                        be[STATE_I * cell_stride + cell_diag] += beta_M * wM_I;
                        be[STATE_D * cell_stride + cell_diag] += beta_M * wM_D;
                    }

                    // Propagate d_beta to predecessors
                    if (i > 1 && j > 1) {
                        dbe[STATE_M * cell_stride + cell_diag] += dbeta_M * wM_M + beta_M * dwM_M;
                        dbe[STATE_I * cell_stride + cell_diag] += dbeta_M * wM_I + beta_M * dwM_I;
                        dbe[STATE_D * cell_stride + cell_diag] += dbeta_M * wM_D + beta_M * dwM_D;
                    }

                    // ========== I state HVP ==========
                    float beta_I = be[STATE_I * cell_stride + cell_idx];
                    float dbeta_I = dbe[STATE_I * cell_stride + cell_idx];

                    float i_from_M = (i > 1) ? a[STATE_M * cell_stride + cell_up] + gap_open : NINF;
                    float i_from_I = (i > 1) ? a[STATE_I * cell_stride + cell_up] + gap_ext : NINF;

                    float wI_M, wI_I;
                    softmax2_weights_cpu(i_from_M, i_from_I, T, wI_M, wI_I);

                    float di_from_M = (i > 1) ? da[STATE_M * cell_stride + cell_up] : 0.0f;
                    float di_from_I = (i > 1) ? da[STATE_I * cell_stride + cell_up] : 0.0f;

                    float sum_wI_di = wI_M * di_from_M + wI_I * di_from_I;
                    float dwI_M = wI_M * (di_from_M - sum_wI_di) / T;
                    float dwI_I = wI_I * (di_from_I - sum_wI_di) / T;

                    // Propagate beta (was missing!)
                    if (i > 1) {
                        be[STATE_M * cell_stride + cell_up] += beta_I * wI_M;
                        be[STATE_I * cell_stride + cell_up] += beta_I * wI_I;
                    }

                    // Propagate d_beta
                    if (i > 1) {
                        dbe[STATE_M * cell_stride + cell_up] += dbeta_I * wI_M + beta_I * dwI_M;
                        dbe[STATE_I * cell_stride + cell_up] += dbeta_I * wI_I + beta_I * dwI_I;
                    }

                    // ========== D state HVP ==========
                    float beta_D = be[STATE_D * cell_stride + cell_idx];
                    float dbeta_D = dbe[STATE_D * cell_stride + cell_idx];

                    float d_from_M = (j > 1) ? a[STATE_M * cell_stride + cell_left] + gap_open : NINF;
                    float d_from_D = (j > 1) ? a[STATE_D * cell_stride + cell_left] + gap_ext : NINF;

                    float wD_M, wD_D;
                    softmax2_weights_cpu(d_from_M, d_from_D, T, wD_M, wD_D);

                    float dd_from_M = (j > 1) ? da[STATE_M * cell_stride + cell_left] : 0.0f;
                    float dd_from_D = (j > 1) ? da[STATE_D * cell_stride + cell_left] : 0.0f;

                    float sum_wD_dd = wD_M * dd_from_M + wD_D * dd_from_D;
                    float dwD_M = wD_M * (dd_from_M - sum_wD_dd) / T;
                    float dwD_D = wD_D * (dd_from_D - sum_wD_dd) / T;

                    // Propagate beta (was missing!)
                    if (j > 1) {
                        be[STATE_M * cell_stride + cell_left] += beta_D * wD_M;
                        be[STATE_D * cell_stride + cell_left] += beta_D * wD_D;
                    }

                    // Propagate d_beta
                    if (j > 1) {
                        dbe[STATE_M * cell_stride + cell_left] += dbeta_D * wD_M + beta_D * dwD_M;
                        dbe[STATE_D * cell_stride + cell_left] += dbeta_D * wD_D + beta_D * dwD_D;
                    }
                }
            }
        }
    });
}

// ============================================================================
// PARAMETER GRADIENT
// ============================================================================

enum AffineParamType {
    PARAM_GAP_OPEN = 0,
    PARAM_GAP_EXT = 1,
    PARAM_TEMPERATURE = 2
};

extern "C" void sw_affine_param_grad_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* partition,   // [B]
    const float* dS_dtheta,   // [B] pre-computed dS/dtheta
    float* U,                 // [B, 3*(L1+1)*(L2+1)] workspace
    float* beta,              // [B, 3*(L1+1)*(L2+1)] workspace
    float* W,                 // [B, 3*(L1+1)*(L2+1)] workspace
    float* dP_dtheta,         // [B, L1, L2] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T,
    int param_type
) {
    const size_t cell_stride = (size_t)(max_L1 + 1) * (max_L2 + 1);
    const size_t alpha_stride = 3 * cell_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

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
            u[STATE_M * cell_stride + 0] = 0.0f;

            for (int k = 2; k <= L1 + L2; k++) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);

                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);
                    float score = s[score_idx];

                    // M state
                    float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                    float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                    float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                    float m_sky = score;

                    float wM_M, wM_I, wM_D, wM_sky;
                    softmax4_weights_cpu(m_from_M, m_from_I, m_from_D, m_sky, T, wM_M, wM_I, wM_D, wM_sky);

                    float du_M_M = (i > 1 && j > 1) ? u[STATE_M * cell_stride + cell_diag] : 0.0f;
                    float du_M_I = (i > 1 && j > 1) ? u[STATE_I * cell_stride + cell_diag] : 0.0f;
                    float du_M_D = (i > 1 && j > 1) ? u[STATE_D * cell_stride + cell_diag] : 0.0f;
                    float dU_M = wM_M * du_M_M + wM_I * du_M_I + wM_D * du_M_D;

                    if (param_type == PARAM_TEMPERATURE) {
                        float alpha_M = a[STATE_M * cell_stride + cell_idx];
                        float E_val = wM_M * m_from_M + wM_I * m_from_I + wM_D * m_from_D + wM_sky * m_sky;
                        dU_M += (alpha_M - E_val) / T;
                    }
                    u[STATE_M * cell_stride + cell_idx] = dU_M;

                    // I state
                    float i_from_M = (i > 1) ? a[STATE_M * cell_stride + cell_up] + gap_open : NINF;
                    float i_from_I = (i > 1) ? a[STATE_I * cell_stride + cell_up] + gap_ext : NINF;

                    float wI_M, wI_I;
                    softmax2_weights_cpu(i_from_M, i_from_I, T, wI_M, wI_I);

                    float du_I_M = (i > 1) ? u[STATE_M * cell_stride + cell_up] : 0.0f;
                    float du_I_I = (i > 1) ? u[STATE_I * cell_stride + cell_up] : 0.0f;
                    if (param_type == PARAM_GAP_OPEN) du_I_M += 1.0f;
                    if (param_type == PARAM_GAP_EXT) du_I_I += 1.0f;

                    float dU_I = wI_M * du_I_M + wI_I * du_I_I;
                    if (param_type == PARAM_TEMPERATURE) {
                        float alpha_I = a[STATE_I * cell_stride + cell_idx];
                        float E_val = wI_M * i_from_M + wI_I * i_from_I;
                        dU_I += (alpha_I - E_val) / T;
                    }
                    u[STATE_I * cell_stride + cell_idx] = dU_I;

                    // D state
                    float d_from_M = (j > 1) ? a[STATE_M * cell_stride + cell_left] + gap_open : NINF;
                    float d_from_D = (j > 1) ? a[STATE_D * cell_stride + cell_left] + gap_ext : NINF;

                    float wD_M, wD_D;
                    softmax2_weights_cpu(d_from_M, d_from_D, T, wD_M, wD_D);

                    float du_D_M = (j > 1) ? u[STATE_M * cell_stride + cell_left] : 0.0f;
                    float du_D_D = (j > 1) ? u[STATE_D * cell_stride + cell_left] : 0.0f;
                    if (param_type == PARAM_GAP_OPEN) du_D_M += 1.0f;
                    if (param_type == PARAM_GAP_EXT) du_D_D += 1.0f;

                    float dU_D = wD_M * du_D_M + wD_D * du_D_D;
                    if (param_type == PARAM_TEMPERATURE) {
                        float alpha_D = a[STATE_D * cell_stride + cell_idx];
                        float E_val = wD_M * d_from_M + wD_D * d_from_D;
                        dU_D += (alpha_D - E_val) / T;
                    }
                    u[STATE_D * cell_stride + cell_idx] = dU_D;
                }
            }

            // =========== Initialize beta ===========
            for (int i = 0; i <= L1; i++) {
                for (int j = 0; j <= L2; j++) {
                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    for (int state = 0; state < 3; state++) {
                        size_t idx = state * cell_stride + cell_idx;
                        be[idx] = clamped_posterior_weight(a[idx], S, T);
                    }
                }
            }

            // =========== Backward pass for dP/dtheta ===========
            for (int k = L1 + L2; k >= 2; k--) {
                int i_start = std::max(1, k - L2);
                int i_end = std::min(L1, k - 1);

                for (int i = i_start; i <= i_end; i++) {
                    int j = k - i;
                    if (j < 1 || j > L2) continue;

                    size_t cell_idx = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_diag = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t cell_up = static_cast<size_t>(i - 1) * alpha_cols + static_cast<size_t>(j);
                    size_t cell_left = static_cast<size_t>(i) * alpha_cols + static_cast<size_t>(j - 1);
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    size_t idx_M = STATE_M * cell_stride + cell_idx;
                    size_t idx_I = STATE_I * cell_stride + cell_idx;
                    size_t idx_D = STATE_D * cell_stride + cell_idx;
                    float score = s[score_idx];

                    // M state
                    float beta_M = be[idx_M];
                    float beta_init_M = clamped_posterior_weight(a[idx_M], S, T);
                    float W_M = w[idx_M] + beta_init_M * (u[idx_M] - dS_dt) / T;
                    if (param_type == PARAM_TEMPERATURE && a[idx_M] > NINF) {
                        W_M -= beta_init_M * (a[idx_M] - S) / (T * T);
                    }

                    float m_from_M = (i > 1 && j > 1) ? a[STATE_M * cell_stride + cell_diag] + score : NINF;
                    float m_from_I = (i > 1 && j > 1) ? a[STATE_I * cell_stride + cell_diag] + score : NINF;
                    float m_from_D = (i > 1 && j > 1) ? a[STATE_D * cell_stride + cell_diag] + score : NINF;
                    float m_sky = score;

                    float wM_M, wM_I, wM_D, wM_sky;
                    softmax4_weights_cpu(m_from_M, m_from_I, m_from_D, m_sky, T, wM_M, wM_I, wM_D, wM_sky);

                    float du_M_M = (i > 1 && j > 1) ? u[STATE_M * cell_stride + cell_diag] : 0.0f;
                    float du_M_I = (i > 1 && j > 1) ? u[STATE_I * cell_stride + cell_diag] : 0.0f;
                    float du_M_D = (i > 1 && j > 1) ? u[STATE_D * cell_stride + cell_diag] : 0.0f;
                    float du_M_sky = 0.0f;
                    float sum_w_du = wM_M * du_M_M + wM_I * du_M_I + wM_D * du_M_D + wM_sky * du_M_sky;

                    float dwM_M = wM_M * (du_M_M - sum_w_du) / T;
                    float dwM_I = wM_I * (du_M_I - sum_w_du) / T;
                    float dwM_D = wM_D * (du_M_D - sum_w_du) / T;
                    float dwM_sky = wM_sky * (du_M_sky - sum_w_du) / T;

                    if (param_type == PARAM_TEMPERATURE) {
                        float E_val = wM_M * m_from_M + wM_I * m_from_I + wM_D * m_from_D + wM_sky * m_sky;
                        float inv_T2 = 1.0f / (T * T);
                        dwM_M += wM_M * (E_val - m_from_M) * inv_T2;
                        dwM_I += wM_I * (E_val - m_from_I) * inv_T2;
                        dwM_D += wM_D * (E_val - m_from_D) * inv_T2;
                        dwM_sky += wM_sky * (E_val - m_sky) * inv_T2;
                    }

                    dP[score_idx] += W_M;

                    if (i > 1 && j > 1) {
                        be[STATE_M * cell_stride + cell_diag] += beta_M * wM_M;
                        be[STATE_I * cell_stride + cell_diag] += beta_M * wM_I;
                        be[STATE_D * cell_stride + cell_diag] += beta_M * wM_D;
                        w[STATE_M * cell_stride + cell_diag] += W_M * wM_M + beta_M * dwM_M;
                        w[STATE_I * cell_stride + cell_diag] += W_M * wM_I + beta_M * dwM_I;
                        w[STATE_D * cell_stride + cell_diag] += W_M * wM_D + beta_M * dwM_D;
                    }

                    // I state
                    float beta_I = be[idx_I];
                    float beta_init_I = clamped_posterior_weight(a[idx_I], S, T);
                    float W_I = w[idx_I] + beta_init_I * (u[idx_I] - dS_dt) / T;
                    if (param_type == PARAM_TEMPERATURE && a[idx_I] > NINF) {
                        W_I -= beta_init_I * (a[idx_I] - S) / (T * T);
                    }

                    if (i > 1) {
                        float i_from_M = a[STATE_M * cell_stride + cell_up] + gap_open;
                        float i_from_I = a[STATE_I * cell_stride + cell_up] + gap_ext;
                        float wI_M, wI_I;
                        softmax2_weights_cpu(i_from_M, i_from_I, T, wI_M, wI_I);

                        float du_I_M = u[STATE_M * cell_stride + cell_up];
                        float du_I_I = u[STATE_I * cell_stride + cell_up];
                        if (param_type == PARAM_GAP_OPEN) du_I_M += 1.0f;
                        if (param_type == PARAM_GAP_EXT) du_I_I += 1.0f;

                        float sum_wI = wI_M * du_I_M + wI_I * du_I_I;
                        float dwI_M = wI_M * (du_I_M - sum_wI) / T;
                        float dwI_I = wI_I * (du_I_I - sum_wI) / T;

                        if (param_type == PARAM_TEMPERATURE) {
                            float E_val = wI_M * i_from_M + wI_I * i_from_I;
                            float inv_T2 = 1.0f / (T * T);
                            dwI_M += wI_M * (E_val - i_from_M) * inv_T2;
                            dwI_I += wI_I * (E_val - i_from_I) * inv_T2;
                        }

                        be[STATE_M * cell_stride + cell_up] += beta_I * wI_M;
                        be[STATE_I * cell_stride + cell_up] += beta_I * wI_I;
                        w[STATE_M * cell_stride + cell_up] += W_I * wI_M + beta_I * dwI_M;
                        w[STATE_I * cell_stride + cell_up] += W_I * wI_I + beta_I * dwI_I;
                    }

                    // D state
                    float beta_D = be[idx_D];
                    float beta_init_D = clamped_posterior_weight(a[idx_D], S, T);
                    float W_D = w[idx_D] + beta_init_D * (u[idx_D] - dS_dt) / T;
                    if (param_type == PARAM_TEMPERATURE && a[idx_D] > NINF) {
                        W_D -= beta_init_D * (a[idx_D] - S) / (T * T);
                    }

                    if (j > 1) {
                        float d_from_M = a[STATE_M * cell_stride + cell_left] + gap_open;
                        float d_from_D = a[STATE_D * cell_stride + cell_left] + gap_ext;
                        float wD_M, wD_D;
                        softmax2_weights_cpu(d_from_M, d_from_D, T, wD_M, wD_D);

                        float du_D_M = u[STATE_M * cell_stride + cell_left];
                        float du_D_D = u[STATE_D * cell_stride + cell_left];
                        if (param_type == PARAM_GAP_OPEN) du_D_M += 1.0f;
                        if (param_type == PARAM_GAP_EXT) du_D_D += 1.0f;

                        float sum_wD = wD_M * du_D_M + wD_D * du_D_D;
                        float dwD_M = wD_M * (du_D_M - sum_wD) / T;
                        float dwD_D = wD_D * (du_D_D - sum_wD) / T;

                        if (param_type == PARAM_TEMPERATURE) {
                            float E_val = wD_M * d_from_M + wD_D * d_from_D;
                            float inv_T2 = 1.0f / (T * T);
                            dwD_M += wD_M * (E_val - d_from_M) * inv_T2;
                            dwD_D += wD_D * (E_val - d_from_D) * inv_T2;
                        }

                        be[STATE_M * cell_stride + cell_left] += beta_D * wD_M;
                        be[STATE_D * cell_stride + cell_left] += beta_D * wD_D;
                        w[STATE_M * cell_stride + cell_left] += W_D * wD_M + beta_D * dwD_M;
                        w[STATE_D * cell_stride + cell_left] += W_D * wD_D + beta_D * dwD_D;
                    }
                }
            }
        }
    });
}
