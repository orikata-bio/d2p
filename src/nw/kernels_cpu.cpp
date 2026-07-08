// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft Needleman-Wunsch CPU Kernels (Linear Gap Penalty)
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Needleman-Wunsch finds optimal GLOBAL alignments between two sequences.
 * This CPU implementation is functionally identical to the CUDA version
 * but uses sequential processing with enhanced numerical precision.
 *
 * Key properties:
 *   - Global alignment: aligns full sequences end-to-end
 *   - Linear gap model: each gap costs a fixed penalty
 *   - Soft version: uses temperature-scaled logsumexp instead of max
 *
 * Key differences from Smith-Waterman:
 *   - 3 transitions (diagonal+score, up+gap, left+gap) - no "sky" restart
 *   - Base cases: alpha[0,0]=0, alpha[i,0]=i*gap, alpha[0,j]=j*gap
 *   - Score = alpha[L1, L2], not logsumexp over all cells
 *   - Posteriors = beta * w_diag (option-additive)
 *
 * ============================================================================
 */

#include <cmath>
#include <algorithm>

#include <ATen/Parallel.h>

// Shared utilities
#include "common/numerics.h"

using namespace d2p::common;

// ============================================================================
// NW-Specific Helper Functions
// ============================================================================

// Safe softmax weight computation for 3 values
// Sets weight to 0 for -inf options to avoid 0/0
inline void softmax3_weights_cpu(float a, float b, float c, float T,
                                  float& wa, float& wb, float& wc) {
    float max_v = std::max({a, b, c});
    if (max_v <= NINF) {
        wa = wb = wc = 0.0f;
        return;
    }

    // Use Kahan summation for precision
    KahanSum sum;

    // Set weight to 0 for -inf options before normalization
    if (a > NINF) { wa = safe_exp((a - max_v) / T); sum.add(wa); } else { wa = 0.0f; }
    if (b > NINF) { wb = safe_exp((b - max_v) / T); sum.add(wb); } else { wb = 0.0f; }
    if (c > NINF) { wc = safe_exp((c - max_v) / T); sum.add(wc); } else { wc = 0.0f; }

    float total = sum.result();
    if (total > 0) {
        wa /= total; wb /= total; wc /= total;
    }
}

// Parameter type enum (must match CUDA)
enum NWParamType {
    NW_PARAM_GAP = 0,
    NW_PARAM_TEMPERATURE = 1
};

// ============================================================================
// FORWARD PASS
// ============================================================================

extern "C" void nw_forward_cpu(
    const float* scores,      // [B, L1, L2]
    float* alpha,             // [B, (L1+1)*(L2+1)]
    float* score,             // [B]
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    const size_t score_stride = static_cast<size_t>(max_L1) * score_cols;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = scores + b * score_stride;
            float* a = alpha + b * alpha_stride;
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize alpha with NW base cases:
            // alpha[0,0] = 0
            // alpha[i,0] = i * gap for i > 0
            // alpha[0,j] = j * gap for j > 0
            // All other cells = NINF (will be computed)
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                a[idx] = NINF;
            }

            a[0] = 0.0f;  // alpha[0,0] = 0

            // alpha[i,0] = i * gap
            for (int i = 1; i <= L1; i++) {
                size_t idx = static_cast<size_t>(i) * alpha_cols;
                a[idx] = i * gap;
            }

            // alpha[0,j] = j * gap
            for (int j = 1; j <= L2; j++) {
                size_t idx = static_cast<size_t>(j);
                a[idx] = j * gap;
            }

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

                    float sc = s[score_idx];

                    // Three logits (option-additive: score/gap added to options)
                    float v_diag = a[idx_diag] + sc;    // diagonal + match score
                    float v_up   = a[idx_up]   + gap;   // up + gap penalty
                    float v_left = a[idx_left] + gap;   // left + gap penalty

                    // NW recurrence: alpha[i,j] = logsumexp(v_diag, v_up, v_left)
                    a[idx] = logsumexp3(v_diag, v_up, v_left, T);
                }
            }

            // Score is just the final cell (global alignment)
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            score[b] = a[final_idx];
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

extern "C" void nw_backward_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* posteriors,        // [B, L1, L2] output
    float* grad_gap,          // [B] output
    float* grad_T,            // [B] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    const size_t score_stride = static_cast<size_t>(max_L1) * score_cols;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* be = beta + b * alpha_stride;
            float* post = posteriors + b * score_stride;
            float S = score[b];
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
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;

            // Backward DP: process anti-diagonals in reverse
            KahanSum gap_grad_sum;
            KahanSum match_score_sum;

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

                    float beta_ij = be[idx];

                    // Skip if beta is 0 (unreachable cell)
                    if (beta_ij == 0.0f) continue;

                    float sc = s[score_idx];

                    // Recompute logits (must match forward exactly)
                    float v_diag = a[idx_diag] + sc;
                    float v_up   = a[idx_up]   + gap;
                    float v_left = a[idx_left] + gap;

                    // Compute softmax weights
                    float w_diag, w_up, w_left;
                    softmax3_weights_cpu(v_diag, v_up, v_left, T, w_diag, w_up, w_left);

                    // Posteriors: P[i,j] = beta * w_diag (option-additive)
                    float post_ij = beta_ij * w_diag;
                    post[score_idx] = post_ij;

                    // Accumulate for temperature gradient
                    match_score_sum.add(post_ij * sc);

                    // Gap gradient: sum beta * (w_up + w_left)
                    gap_grad_sum.add(beta_ij * (w_up + w_left));

                    // Propagate beta to predecessors
                    if (w_diag > 0.0f) be[idx_diag] += beta_ij * w_diag;
                    if (w_up > 0.0f)   be[idx_up]   += beta_ij * w_up;
                    if (w_left > 0.0f) be[idx_left] += beta_ij * w_left;
                }
            }

            // NW boundary cells already encode gap costs via alpha[i,0] = i*gap
            // and alpha[0,j] = j*gap, so their beta mass contributes to dS/dgap too.
            for (int i = 1; i <= L1; i++) {
                gap_grad_sum.add(be[static_cast<size_t>(i) * alpha_cols] * static_cast<float>(i));
            }
            for (int j = 1; j <= L2; j++) {
                gap_grad_sum.add(be[static_cast<size_t>(j)] * static_cast<float>(j));
            }

            float gap_grad = gap_grad_sum.result();
            grad_gap[b] = gap_grad;

            // Temperature gradient: dS/dT = (S - E[total_score]) / T
            float expected_total = match_score_sum.result() + gap_grad * gap;
            grad_T[b] = (S - expected_total) / T;
        }
    });
}

// ============================================================================
// HESSIAN-VECTOR PRODUCT (HVP)
//
// Computes d^2S/dscores^2 * V using forward-mode autodiff through the backward pass.
// ============================================================================

extern "C" void nw_hvp_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
    const float* V,           // [B, L1, L2] tangent vector
    float* d_alpha,           // [B, (L1+1)*(L2+1)] workspace
    float* d_score,           // [B] workspace
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* d_beta,            // [B, (L1+1)*(L2+1)] workspace
    float* H_scores,          // [B, L1, L2] output: Hessian-vector product
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T
) {
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    const size_t score_stride = static_cast<size_t>(max_L1) * score_cols;

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
            // Compute d_alpha: tangent of alpha w.r.t. scores in direction V
            da[0] = 0.0f;  // d_alpha[0,0] = 0 (constant)

            // Tangent of boundary conditions (they don't depend on scores)
            for (int i = 1; i <= L1; i++) {
                size_t idx = static_cast<size_t>(i) * alpha_cols;
                da[idx] = 0.0f;  // d(i*gap)/dscores = 0
            }
            for (int j = 1; j <= L2; j++) {
                size_t idx = static_cast<size_t>(j);
                da[idx] = 0.0f;  // d(j*gap)/dscores = 0
            }

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

                    float sc = s[score_idx];
                    float v_ij = v[score_idx];

                    // Recompute logits
                    float log_diag = a[idx_diag] + sc;
                    float log_up   = a[idx_up]   + gap;
                    float log_left = a[idx_left] + gap;

                    float w_diag, w_up, w_left;
                    softmax3_weights_cpu(log_diag, log_up, log_left, T, w_diag, w_up, w_left);

                    // Tangent of logits
                    float dv_diag = da[idx_diag] + v_ij;  // d(alpha_diag + score) = da_diag + V
                    float dv_up   = da[idx_up];           // d(alpha_up + gap) = da_up
                    float dv_left = da[idx_left];         // d(alpha_left + gap) = da_left

                    // d(alpha[i,j]) = sum w_k * dv_k
                    da[idx] = w_diag * dv_diag + w_up * dv_up + w_left * dv_left;
                }
            }

            // d_score = d_alpha[L1, L2]
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            float dS = da[final_idx];
            d_score[b] = dS;

            // =========== Backward pass (primal) ===========
            be[final_idx] = 1.0f;

            // =========== Backward tangent pass ===========
            dbe[final_idx] = 0.0f;  // beta is constant 1 at terminal

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

                    float sc = s[score_idx];
                    float v_ij = v[score_idx];
                    float beta_ij = be[idx];
                    float dbeta_ij = dbe[idx];

                    if (beta_ij == 0.0f && std::abs(dbeta_ij) < 1e-20f) continue;

                    // Recompute logits and weights
                    float log_diag = a[idx_diag] + sc;
                    float log_up   = a[idx_up]   + gap;
                    float log_left = a[idx_left] + gap;

                    float w_diag, w_up, w_left;
                    softmax3_weights_cpu(log_diag, log_up, log_left, T, w_diag, w_up, w_left);

                    // Tangent of logits
                    float dv_diag = da[idx_diag] + v_ij;
                    float dv_up   = da[idx_up];
                    float dv_left = da[idx_left];

                    // Compute weight tangents for softmax
                    // dw_k = w_k * (dv_k - E[dv]) / T
                    float E_dv = w_diag * dv_diag + w_up * dv_up + w_left * dv_left;

                    float dw_diag = w_diag * (dv_diag - E_dv) / T;
                    float dw_up   = w_up   * (dv_up   - E_dv) / T;
                    float dw_left = w_left * (dv_left - E_dv) / T;

                    // HVP: d(posteriors) = dbeta * w_diag + beta * dw_diag
                    H[score_idx] = dbeta_ij * w_diag + beta_ij * dw_diag;

                    // Propagate beta and dbeta
                    if (w_diag > 0.0f) {
                        be[idx_diag] += beta_ij * w_diag;
                        dbe[idx_diag] += dbeta_ij * w_diag + beta_ij * dw_diag;
                    }
                    if (w_up > 0.0f) {
                        be[idx_up] += beta_ij * w_up;
                        dbe[idx_up] += dbeta_ij * w_up + beta_ij * dw_up;
                    }
                    if (w_left > 0.0f) {
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
//
// Computes dP/d{gap, T} (derivative of posteriors w.r.t. parameters)
// ============================================================================

extern "C" void nw_param_grad_cpu(
    const float* alpha,       // [B, (L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
    const float* dS_dtheta,   // [B] pre-computed dS/dtheta
    float* U,                 // [B, (L1+1)*(L2+1)] workspace: dalpha/dtheta
    float* beta,              // [B, (L1+1)*(L2+1)] workspace
    float* W,                 // [B, (L1+1)*(L2+1)] workspace: dbeta/dtheta
    float* dP_dtheta,         // [B, L1, L2] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap, float T,
    int param_type
) {
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;
    const size_t score_cols = static_cast<size_t>(max_L2);
    const size_t alpha_stride = (static_cast<size_t>(max_L1) + 1) * alpha_cols;
    const size_t score_stride = static_cast<size_t>(max_L1) * score_cols;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* a = alpha + b * alpha_stride;
            const float* s = scores + b * score_stride;
            float* u = U + b * alpha_stride;
            float* be = beta + b * alpha_stride;
            float* w = W + b * alpha_stride;
            float* dPdT = dP_dtheta + b * score_stride;
            float S = score[b];
            float dS = dS_dtheta[b];
            int L1 = lengths[b * 2];
            int L2 = lengths[b * 2 + 1];

            // Initialize workspaces
            for (size_t idx = 0; idx < alpha_stride; idx++) {
                u[idx] = 0.0f;
                be[idx] = 0.0f;
                w[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                dPdT[idx] = 0.0f;
            }

            // =========== Forward U pass: compute dalpha/dtheta ===========
            u[0] = 0.0f;  // dalpha[0,0]/dtheta = 0

            // Boundary conditions for gap parameter
            if (param_type == NW_PARAM_GAP) {
                // d(i*gap)/dgap = i
                for (int i = 1; i <= L1; i++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols;
                    u[idx] = static_cast<float>(i);
                }
                for (int j = 1; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(j);
                    u[idx] = static_cast<float>(j);
                }
            }
            // For temperature, boundaries don't depend on T

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

                    float sc = s[score_idx];

                    // Recompute logits
                    float v_diag = a[idx_diag] + sc;
                    float v_up   = a[idx_up]   + gap;
                    float v_left = a[idx_left] + gap;

                    float w_diag, w_up, w_left;
                    softmax3_weights_cpu(v_diag, v_up, v_left, T, w_diag, w_up, w_left);

                    // Get predecessor U values
                    float u_diag = u[idx_diag];
                    float u_up   = u[idx_up];
                    float u_left = u[idx_left];

                    // For gap parameter: add +1 tangent to gap transitions
                    if (param_type == NW_PARAM_GAP) {
                        u_up   += 1.0f;
                        u_left += 1.0f;
                    }

                    // Propagate tangent: U[idx] = sum w_k * u_k
                    float U_val = w_diag * u_diag + w_up * u_up + w_left * u_left;

                    // For temperature: add (alpha - E[v]) / T term
                    if (param_type == NW_PARAM_TEMPERATURE) {
                        float alpha_ij = a[idx];
                        float E_v = w_diag * v_diag + w_up * v_up + w_left * v_left;
                        U_val += (alpha_ij - E_v) / T;
                    }

                    u[idx] = U_val;
                }
            }

            // =========== Backward pass (primal) ===========
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            be[final_idx] = 1.0f;

            // =========== Backward W pass: compute dbeta/dtheta ===========
            w[final_idx] = 0.0f;  // terminal beta is constant

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

                    float sc = s[score_idx];
                    float beta_curr = be[idx];
                    float W_ij = w[idx];

                    if (beta_curr == 0.0f && std::abs(W_ij) < 1e-20f) continue;

                    // Recompute logits and weights
                    float v_diag = a[idx_diag] + sc;
                    float v_up   = a[idx_up]   + gap;
                    float v_left = a[idx_left] + gap;

                    float wt_diag, wt_up, wt_left;
                    softmax3_weights_cpu(v_diag, v_up, v_left, T, wt_diag, wt_up, wt_left);

                    // Get predecessor U values
                    float u_diag = u[idx_diag];
                    float u_up   = u[idx_up];
                    float u_left = u[idx_left];

                    // Add gap tangent
                    if (param_type == NW_PARAM_GAP) {
                        u_up   += 1.0f;
                        u_left += 1.0f;
                    }

                    // Compute weight tangents via softmax Jacobian
                    // dw_k = w_k * (u_k - E[u]) / T
                    float E_u = wt_diag * u_diag + wt_up * u_up + wt_left * u_left;
                    float dw_diag = wt_diag * (u_diag - E_u) / T;
                    float dw_up   = wt_up   * (u_up   - E_u) / T;
                    float dw_left = wt_left * (u_left - E_u) / T;

                    // For temperature: add direct dw/dT = w_k * (E[v] - v_k) / T^2
                    if (param_type == NW_PARAM_TEMPERATURE) {
                        float E_v = wt_diag * v_diag + wt_up * v_up + wt_left * v_left;
                        float inv_T2 = 1.0f / (T * T);
                        dw_diag += wt_diag * (E_v - v_diag) * inv_T2;
                        dw_up   += wt_up   * (E_v - v_up)   * inv_T2;
                        dw_left += wt_left * (E_v - v_left) * inv_T2;
                    }

                    // Accumulate dP/dtheta: posteriors = beta * w_diag
                    // d(posteriors) = W_ij * w_diag + beta_curr * dw_diag
                    dPdT[score_idx] = W_ij * wt_diag + beta_curr * dw_diag;

                    // Propagate beta and W to predecessors
                    if (wt_diag > 0.0f) {
                        be[idx_diag] += beta_curr * wt_diag;
                        w[idx_diag] += W_ij * wt_diag + beta_curr * dw_diag;
                    }
                    if (wt_up > 0.0f) {
                        be[idx_up] += beta_curr * wt_up;
                        w[idx_up] += W_ij * wt_up + beta_curr * dw_up;
                    }
                    if (wt_left > 0.0f) {
                        be[idx_left] += beta_curr * wt_left;
                        w[idx_left] += W_ij * wt_left + beta_curr * dw_left;
                    }
                }
            }
        }
    });
}
