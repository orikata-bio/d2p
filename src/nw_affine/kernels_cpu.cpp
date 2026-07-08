// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft Needleman-Wunsch Affine Gap CPU Kernels
 *
 * CPU implementation mirroring the CUDA kernels for seamless device dispatch.
 * Uses sequential wavefront iteration with Kahan summation for precision.
 *
 * Three-state DP: M (Match), I (Insert/gap in seq2), D (Delete/gap in seq1)
 *
 * Key differences from SW affine:
 *   - No "sky" restart transition in M state (global alignment)
 *   - Base cases: M(0,0)=0, I(i,0)=g_o+(i-1)*g_e, D(0,j)=g_o+(j-1)*g_e
 *   - Score = LSE(M(L1,L2), I(L1,L2), D(L1,L2)), not over all cells
 *   - Beta initialized at terminal only via final LSE weights
 */

#include <cmath>
#include <algorithm>
#include <limits>

#include <ATen/Parallel.h>

// ============================================================================
// Constants and Helper Functions
// ============================================================================

constexpr float NINF = -1e30f;

// Parameter types (must match CUDA)
enum NWAffineParamType {
    NW_AFFINE_PARAM_GAP_OPEN = 0,
    NW_AFFINE_PARAM_GAP_EXT = 1,
    NW_AFFINE_PARAM_TEMPERATURE = 2
};

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

// Softmax for 3 values
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

// Softmax weights for 3 values
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

// ============================================================================
// FORWARD PASS
// ============================================================================

extern "C" void nw_affine_forward_cpu(
    const float* scores,      // [B, L1, L2]
    float* alpha,             // [B, 3*(L1+1)*(L2+1)]
    float* score,             // [B]
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t state_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t alpha_stride = 3 * state_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* s = scores + b * score_stride;
            float* M = alpha + b * alpha_stride;
            float* I = M + state_stride;
            float* D = I + state_stride;
            int L1 = lengths ? lengths[b * 2] : max_L1;
            int L2 = lengths ? lengths[b * 2 + 1] : max_L2;

            // Initialize all states to -inf
            for (size_t idx = 0; idx < state_stride; idx++) {
                M[idx] = NINF;
                I[idx] = NINF;
                D[idx] = NINF;
            }

            // Base cases:
            // M(0,0) = 0
            M[0] = 0.0f;

            // I(i,0) = gap_open + (i-1)*gap_ext for i > 0
            for (int i = 1; i <= L1; i++) {
                size_t idx = static_cast<size_t>(i) * alpha_cols;
                I[idx] = gap_open + (i - 1) * gap_ext;
            }

            // D(0,j) = gap_open + (j-1)*gap_ext for j > 0
            for (int j = 1; j <= L2; j++) {
                size_t idx = static_cast<size_t>(j);
                D[idx] = gap_open + (j - 1) * gap_ext;
            }

            // Forward DP: process anti-diagonals
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
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sc = s[score_idx];

                    // M[i,j] = score + LSE(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1])
                    M[idx] = sc + softmax3(M[idx_diag], I[idx_diag], D[idx_diag], T);

                    // I[i,j] = LSE(M[i-1,j]+gap_open, I[i-1,j]+gap_ext, D[i-1,j]+gap_open)
                    I[idx] = softmax3(
                        M[idx_up] + gap_open,
                        I[idx_up] + gap_ext,
                        D[idx_up] + gap_open, T
                    );

                    // D[i,j] = LSE(M[i,j-1]+gap_open, I[i,j-1]+gap_open, D[i,j-1]+gap_ext)
                    D[idx] = softmax3(
                        M[idx_left] + gap_open,
                        I[idx_left] + gap_open,
                        D[idx_left] + gap_ext, T
                    );
                }
            }

            // Score = LSE(M[L1,L2], I[L1,L2], D[L1,L2])
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            score[b] = softmax3(M[final_idx], I[final_idx], D[final_idx], T);
        }
    });
}

// ============================================================================
// BACKWARD PASS
// ============================================================================

extern "C" void nw_affine_backward_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
    float* beta,              // [B, 3*(L1+1)*(L2+1)] workspace
    float* posteriors,        // [B, L1, L2] output
    float* grad_open,         // [B] output
    float* grad_ext,          // [B] output
    float* grad_T,            // [B] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t state_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t alpha_stride = 3 * state_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* A_M = alpha + b * alpha_stride;
            const float* A_I = A_M + state_stride;
            const float* A_D = A_I + state_stride;
            const float* s = scores + b * score_stride;

            float* B_M = beta + b * alpha_stride;
            float* B_I = B_M + state_stride;
            float* B_D = B_I + state_stride;

            float* post = posteriors + b * score_stride;
            float S = score[b];
            int L1 = lengths ? lengths[b * 2] : max_L1;
            int L2 = lengths ? lengths[b * 2 + 1] : max_L2;

            // Initialize beta and posteriors
            for (size_t idx = 0; idx < state_stride; idx++) {
                B_M[idx] = 0.0f;
                B_I[idx] = 0.0f;
                B_D[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                post[idx] = 0.0f;
            }

            // Initialize beta at terminal: weights from final LSE
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            float w_m, w_i, w_d;
            softmax3_weights(A_M[final_idx], A_I[final_idx], A_D[final_idx], T, w_m, w_i, w_d);
            B_M[final_idx] = w_m;
            B_I[final_idx] = w_i;
            B_D[final_idx] = w_d;

            // Backward DP
            float sum_open = 0.0f;
            float sum_ext = 0.0f;
            KahanAccumulator match_score_sum;

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
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float sc = s[score_idx];
                    float betaM = B_M[idx];
                    float betaI = B_I[idx];
                    float betaD = B_D[idx];

                    // M state backprop
                    if (betaM > 1e-20f) {
                        float m1 = A_M[idx_diag];
                        float m2 = A_I[idx_diag];
                        float m3 = A_D[idx_diag];
                        float w1, w2, w3;
                        if (softmax3(m1, m2, m3, T) > NINF) {
                            softmax3_weights(m1, m2, m3, T, w1, w2, w3);
                            if (m1 > NINF) B_M[idx_diag] += betaM * w1;
                            if (m2 > NINF) B_I[idx_diag] += betaM * w2;
                            if (m3 > NINF) B_D[idx_diag] += betaM * w3;
                            post[score_idx] += betaM;
                            match_score_sum.add(betaM * sc);
                        }
                    }

                    // I state backprop
                    if (betaI > 1e-20f) {
                        float i1 = A_M[idx_up] + gap_open;
                        float i2 = A_I[idx_up] + gap_ext;
                        float i3 = A_D[idx_up] + gap_open;
                        float w1, w2, w3;
                        if (softmax3(i1, i2, i3, T) > NINF) {
                            softmax3_weights(i1, i2, i3, T, w1, w2, w3);
                            B_M[idx_up] += betaI * w1;
                            B_I[idx_up] += betaI * w2;
                            B_D[idx_up] += betaI * w3;
                            sum_open += betaI * (w1 + w3);  // M->I, D->I
                            sum_ext += betaI * w2;          // I->I
                        }
                    }

                    // D state backprop
                    if (betaD > 1e-20f) {
                        float d1 = A_M[idx_left] + gap_open;
                        float d2 = A_I[idx_left] + gap_open;
                        float d3 = A_D[idx_left] + gap_ext;
                        float w1, w2, w3;
                        if (softmax3(d1, d2, d3, T) > NINF) {
                            softmax3_weights(d1, d2, d3, T, w1, w2, w3);
                            B_M[idx_left] += betaD * w1;
                            B_I[idx_left] += betaD * w2;
                            B_D[idx_left] += betaD * w3;
                            sum_open += betaD * (w1 + w2);  // M->D, I->D
                            sum_ext += betaD * w3;          // D->D
                        }
                    }
                }
            }

            // Account for leading-gap boundary states, which are initialized
            // directly from gap_open / gap_ext rather than via a DP transition.
            for (int i = 1; i <= L1; ++i) {
                size_t idx = static_cast<size_t>(i) * alpha_cols;
                float beta_boundary = B_I[idx];
                if (beta_boundary > 1e-20f) {
                    sum_open += beta_boundary;
                    sum_ext += beta_boundary * static_cast<float>(i - 1);
                }
            }
            for (int j = 1; j <= L2; ++j) {
                size_t idx = static_cast<size_t>(j);
                float beta_boundary = B_D[idx];
                if (beta_boundary > 1e-20f) {
                    sum_open += beta_boundary;
                    sum_ext += beta_boundary * static_cast<float>(j - 1);
                }
            }

            grad_open[b] = sum_open;
            grad_ext[b] = sum_ext;

            // Temperature gradient
            float expected_total = match_score_sum.result() + sum_open * gap_open + sum_ext * gap_ext;
            grad_T[b] = (S - expected_total) / T;
        }
    });
}

// ============================================================================
// HVP (Hessian-Vector Product)
// ============================================================================

extern "C" void nw_affine_hvp_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
    const float* V,           // [B, L1, L2] tangent vector
    float* d_alpha,           // [B, 3*(L1+1)*(L2+1)] workspace
    float* d_score,           // [B] workspace
    float* beta,              // [B, 3*(L1+1)*(L2+1)] workspace
    float* d_beta,            // [B, 3*(L1+1)*(L2+1)] workspace
    float* H_scores,          // [B, L1, L2] output
    const int* lengths,       // [B, 2]
    int B, int max_L1, int max_L2,
    float gap_open, float gap_ext, float T
) {
    const size_t state_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t alpha_stride = 3 * state_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* A_M = alpha + b * alpha_stride;
            const float* A_I = A_M + state_stride;
            const float* A_D = A_I + state_stride;
            const float* s = scores + b * score_stride;
            const float* v = V + b * score_stride;

            float* dA_M = d_alpha + b * alpha_stride;
            float* dA_I = dA_M + state_stride;
            float* dA_D = dA_I + state_stride;

            float* B_M = beta + b * alpha_stride;
            float* B_I = B_M + state_stride;
            float* B_D = B_I + state_stride;

            float* dB_M = d_beta + b * alpha_stride;
            float* dB_I = dB_M + state_stride;
            float* dB_D = dB_I + state_stride;

            float* H = H_scores + b * score_stride;
            int L1 = lengths ? lengths[b * 2] : max_L1;
            int L2 = lengths ? lengths[b * 2 + 1] : max_L2;

            // Initialize workspaces
            for (size_t idx = 0; idx < state_stride; idx++) {
                dA_M[idx] = 0.0f;
                dA_I[idx] = 0.0f;
                dA_D[idx] = 0.0f;
                B_M[idx] = 0.0f;
                B_I[idx] = 0.0f;
                B_D[idx] = 0.0f;
                dB_M[idx] = 0.0f;
                dB_I[idx] = 0.0f;
                dB_D[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                H[idx] = 0.0f;
            }

            // =========== Forward tangent pass ===========
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
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float V_s = v[score_idx];

                    // M state tangent
                    {
                        float m1 = A_M[idx_diag];
                        float m2 = A_I[idx_diag];
                        float m3 = A_D[idx_diag];
                        float w1, w2, w3;
                        if (softmax3(m1, m2, m3, T) > NINF) {
                            softmax3_weights(m1, m2, m3, T, w1, w2, w3);
                            float dm1 = dA_M[idx_diag];
                            float dm2 = dA_I[idx_diag];
                            float dm3 = dA_D[idx_diag];
                            dA_M[idx] = V_s + w1 * dm1 + w2 * dm2 + w3 * dm3;
                        } else {
                            dA_M[idx] = 0.0f;
                        }
                    }

                    // I state tangent
                    {
                        float i1 = A_M[idx_up] + gap_open;
                        float i2 = A_I[idx_up] + gap_ext;
                        float i3 = A_D[idx_up] + gap_open;
                        float w1, w2, w3;
                        if (softmax3(i1, i2, i3, T) > NINF) {
                            softmax3_weights(i1, i2, i3, T, w1, w2, w3);
                            dA_I[idx] = w1 * dA_M[idx_up] + w2 * dA_I[idx_up] + w3 * dA_D[idx_up];
                        } else {
                            dA_I[idx] = 0.0f;
                        }
                    }

                    // D state tangent
                    {
                        float d1 = A_M[idx_left] + gap_open;
                        float d2 = A_I[idx_left] + gap_open;
                        float d3 = A_D[idx_left] + gap_ext;
                        float w1, w2, w3;
                        if (softmax3(d1, d2, d3, T) > NINF) {
                            softmax3_weights(d1, d2, d3, T, w1, w2, w3);
                            dA_D[idx] = w1 * dA_M[idx_left] + w2 * dA_I[idx_left] + w3 * dA_D[idx_left];
                        } else {
                            dA_D[idx] = 0.0f;
                        }
                    }
                }
            }

            // Compute d_score
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            float w_m, w_i, w_d;
            softmax3_weights(A_M[final_idx], A_I[final_idx], A_D[final_idx], T, w_m, w_i, w_d);
            d_score[b] = w_m * dA_M[final_idx] + w_i * dA_I[final_idx] + w_d * dA_D[final_idx];

            // Initialize beta at terminal
            B_M[final_idx] = w_m;
            B_I[final_idx] = w_i;
            B_D[final_idx] = w_d;

            // Initialize d_beta at terminal (tangent of beta init)
            {
                float dm = dA_M[final_idx];
                float di = dA_I[final_idx];
                float dd = dA_D[final_idx];
                float E_dv = w_m * dm + w_i * di + w_d * dd;
                dB_M[final_idx] = w_m * (dm - E_dv) / T;
                dB_I[final_idx] = w_i * (di - E_dv) / T;
                dB_D[final_idx] = w_d * (dd - E_dv) / T;
            }

            // =========== Backward tangent pass ===========
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
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float betaM = B_M[idx];
                    float betaI = B_I[idx];
                    float betaD = B_D[idx];
                    float dbetaM = dB_M[idx];
                    float dbetaI = dB_I[idx];
                    float dbetaD = dB_D[idx];

                    // M state
                    if (betaM > 1e-20f || std::abs(dbetaM) > 1e-20f) {
                        float m1 = A_M[idx_diag];
                        float m2 = A_I[idx_diag];
                        float m3 = A_D[idx_diag];
                        float w1, w2, w3;

                        if (softmax3(m1, m2, m3, T) > NINF) {
                            softmax3_weights(m1, m2, m3, T, w1, w2, w3);

                            float dm1 = dA_M[idx_diag];
                            float dm2 = dA_I[idx_diag];
                            float dm3 = dA_D[idx_diag];
                            float E_dv = w1 * dm1 + w2 * dm2 + w3 * dm3;
                            float dw1 = w1 * (dm1 - E_dv) / T;
                            float dw2 = w2 * (dm2 - E_dv) / T;
                            float dw3 = w3 * (dm3 - E_dv) / T;

                            if (m1 > NINF) {
                                B_M[idx_diag] += betaM * w1;
                                dB_M[idx_diag] += dbetaM * w1 + betaM * dw1;
                            }
                            if (m2 > NINF) {
                                B_I[idx_diag] += betaM * w2;
                                dB_I[idx_diag] += dbetaM * w2 + betaM * dw2;
                            }
                            if (m3 > NINF) {
                                B_D[idx_diag] += betaM * w3;
                                dB_D[idx_diag] += dbetaM * w3 + betaM * dw3;
                            }

                            H[score_idx] += dbetaM;
                        }
                    }

                    // I state
                    if (betaI > 1e-20f || std::abs(dbetaI) > 1e-20f) {
                        float i1 = A_M[idx_up] + gap_open;
                        float i2 = A_I[idx_up] + gap_ext;
                        float i3 = A_D[idx_up] + gap_open;
                        float w1, w2, w3;

                        if (softmax3(i1, i2, i3, T) > NINF) {
                            softmax3_weights(i1, i2, i3, T, w1, w2, w3);

                            float di1 = dA_M[idx_up];
                            float di2 = dA_I[idx_up];
                            float di3 = dA_D[idx_up];
                            float E_dv = w1 * di1 + w2 * di2 + w3 * di3;
                            float dw1 = w1 * (di1 - E_dv) / T;
                            float dw2 = w2 * (di2 - E_dv) / T;
                            float dw3 = w3 * (di3 - E_dv) / T;

                            B_M[idx_up] += betaI * w1;
                            dB_M[idx_up] += dbetaI * w1 + betaI * dw1;
                            B_I[idx_up] += betaI * w2;
                            dB_I[idx_up] += dbetaI * w2 + betaI * dw2;
                            B_D[idx_up] += betaI * w3;
                            dB_D[idx_up] += dbetaI * w3 + betaI * dw3;
                        }
                    }

                    // D state
                    if (betaD > 1e-20f || std::abs(dbetaD) > 1e-20f) {
                        float d1 = A_M[idx_left] + gap_open;
                        float d2 = A_I[idx_left] + gap_open;
                        float d3 = A_D[idx_left] + gap_ext;
                        float w1, w2, w3;

                        if (softmax3(d1, d2, d3, T) > NINF) {
                            softmax3_weights(d1, d2, d3, T, w1, w2, w3);

                            float dd1 = dA_M[idx_left];
                            float dd2 = dA_I[idx_left];
                            float dd3 = dA_D[idx_left];
                            float E_dv = w1 * dd1 + w2 * dd2 + w3 * dd3;
                            float dw1 = w1 * (dd1 - E_dv) / T;
                            float dw2 = w2 * (dd2 - E_dv) / T;
                            float dw3 = w3 * (dd3 - E_dv) / T;

                            B_M[idx_left] += betaD * w1;
                            dB_M[idx_left] += dbetaD * w1 + betaD * dw1;
                            B_I[idx_left] += betaD * w2;
                            dB_I[idx_left] += dbetaD * w2 + betaD * dw2;
                            B_D[idx_left] += betaD * w3;
                            dB_D[idx_left] += dbetaD * w3 + betaD * dw3;
                        }
                    }
                }
            }
        }
    });
}

// ============================================================================
// PARAMETER GRADIENT (dP/dtheta)
// ============================================================================

extern "C" void nw_affine_param_grad_cpu(
    const float* alpha,       // [B, 3*(L1+1)*(L2+1)]
    const float* scores,      // [B, L1, L2]
    const float* score,       // [B]
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
    const size_t state_stride = (static_cast<size_t>(max_L1) + 1) * (static_cast<size_t>(max_L2) + 1);
    const size_t alpha_stride = 3 * state_stride;
    const size_t score_stride = (size_t)max_L1 * max_L2;
    const size_t alpha_cols = static_cast<size_t>(max_L2) + 1;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* A_M = alpha + b * alpha_stride;
            const float* A_I = A_M + state_stride;
            const float* A_D = A_I + state_stride;
            const float* s = scores + b * score_stride;

            float* U_M = U + b * alpha_stride;
            float* U_I = U_M + state_stride;
            float* U_D = U_I + state_stride;

            float* B_M = beta + b * alpha_stride;
            float* B_I = B_M + state_stride;
            float* B_D = B_I + state_stride;

            float* W_M = W + b * alpha_stride;
            float* W_I = W_M + state_stride;
            float* W_D = W_I + state_stride;

            float* dP = dP_dtheta + b * score_stride;
            float S = score[b];
            float dS = dS_dtheta[b];
            (void)S;
            (void)dS;
            int L1 = lengths ? lengths[b * 2] : max_L1;
            int L2 = lengths ? lengths[b * 2 + 1] : max_L2;

            // Initialize workspaces
            for (size_t idx = 0; idx < state_stride; idx++) {
                U_M[idx] = 0.0f;
                U_I[idx] = 0.0f;
                U_D[idx] = 0.0f;
                B_M[idx] = 0.0f;
                B_I[idx] = 0.0f;
                B_D[idx] = 0.0f;
                W_M[idx] = 0.0f;
                W_I[idx] = 0.0f;
                W_D[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < score_stride; idx++) {
                dP[idx] = 0.0f;
            }

            // Initialize U for boundary conditions
            if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                // dI(i,0)/dgap_open = 1 (the gap_open term)
                for (int i = 1; i <= L1; i++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols;
                    U_I[idx] = 1.0f;
                }
                // dD(0,j)/dgap_open = 1
                for (int j = 1; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(j);
                    U_D[idx] = 1.0f;
                }
            } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                // dI(i,0)/dgap_ext = i-1
                for (int i = 1; i <= L1; i++) {
                    size_t idx = static_cast<size_t>(i) * alpha_cols;
                    U_I[idx] = static_cast<float>(i - 1);
                }
                // dD(0,j)/dgap_ext = j-1
                for (int j = 1; j <= L2; j++) {
                    size_t idx = static_cast<size_t>(j);
                    U_D[idx] = static_cast<float>(j - 1);
                }
            }

            // =========== Forward U pass ===========
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

                    // M state U
                    {
                        float m1 = A_M[idx_diag];
                        float m2 = A_I[idx_diag];
                        float m3 = A_D[idx_diag];
                        float w1, w2, w3;

                        if (softmax3(m1, m2, m3, T) > NINF) {
                            softmax3_weights(m1, m2, m3, T, w1, w2, w3);
                            float u1 = U_M[idx_diag];
                            float u2 = U_I[idx_diag];
                            float u3 = U_D[idx_diag];

                            float dU_M = w1 * u1 + w2 * u2 + w3 * u3;

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float alpha_M = A_M[idx];
                                float E_v = w1 * m1 + w2 * m2 + w3 * m3;
                                size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);
                                float sc = s[score_idx];
                                dU_M += (alpha_M - (E_v + sc)) / T;
                            }

                            U_M[idx] = dU_M;
                        }
                    }

                    // I state U
                    {
                        float i1 = A_M[idx_up] + gap_open;
                        float i2 = A_I[idx_up] + gap_ext;
                        float i3 = A_D[idx_up] + gap_open;
                        float w1, w2, w3;

                        if (softmax3(i1, i2, i3, T) > NINF) {
                            softmax3_weights(i1, i2, i3, T, w1, w2, w3);
                            float u1 = U_M[idx_up];
                            float u2 = U_I[idx_up];
                            float u3 = U_D[idx_up];

                            float direct1 = 0.0f, direct2 = 0.0f, direct3 = 0.0f;
                            if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                                direct1 = 1.0f;
                                direct3 = 1.0f;
                            } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                                direct2 = 1.0f;
                            }

                            float dU_I = w1 * (u1 + direct1) + w2 * (u2 + direct2) + w3 * (u3 + direct3);

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float alpha_I = A_I[idx];
                                float E_v = w1 * i1 + w2 * i2 + w3 * i3;
                                dU_I += (alpha_I - E_v) / T;
                            }

                            U_I[idx] = dU_I;
                        }
                    }

                    // D state U
                    {
                        float d1 = A_M[idx_left] + gap_open;
                        float d2 = A_I[idx_left] + gap_open;
                        float d3 = A_D[idx_left] + gap_ext;
                        float w1, w2, w3;

                        if (softmax3(d1, d2, d3, T) > NINF) {
                            softmax3_weights(d1, d2, d3, T, w1, w2, w3);
                            float u1 = U_M[idx_left];
                            float u2 = U_I[idx_left];
                            float u3 = U_D[idx_left];

                            float direct1 = 0.0f, direct2 = 0.0f, direct3 = 0.0f;
                            if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                                direct1 = 1.0f;
                                direct2 = 1.0f;
                            } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                                direct3 = 1.0f;
                            }

                            float dU_D = w1 * (u1 + direct1) + w2 * (u2 + direct2) + w3 * (u3 + direct3);

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float alpha_D = A_D[idx];
                                float E_v = w1 * d1 + w2 * d2 + w3 * d3;
                                dU_D += (alpha_D - E_v) / T;
                            }

                            U_D[idx] = dU_D;
                        }
                    }
                }
            }

            // Initialize beta at terminal
            size_t final_idx = static_cast<size_t>(L1) * alpha_cols + static_cast<size_t>(L2);
            float w_m, w_i, w_d;
            softmax3_weights(A_M[final_idx], A_I[final_idx], A_D[final_idx], T, w_m, w_i, w_d);
            B_M[final_idx] = w_m;
            B_I[final_idx] = w_i;
            B_D[final_idx] = w_d;

            // Initialize W at terminal
            {
                float um = U_M[final_idx];
                float ui = U_I[final_idx];
                float ud = U_D[final_idx];
                float E_u = w_m * um + w_i * ui + w_d * ud;
                float dw_m = w_m * (um - E_u) / T;
                float dw_i = w_i * (ui - E_u) / T;
                float dw_d = w_d * (ud - E_u) / T;

                if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                    float E_v = w_m * A_M[final_idx] + w_i * A_I[final_idx] + w_d * A_D[final_idx];
                    float inv_T2 = 1.0f / (T * T);
                    dw_m += w_m * (E_v - A_M[final_idx]) * inv_T2;
                    dw_i += w_i * (E_v - A_I[final_idx]) * inv_T2;
                    dw_d += w_d * (E_v - A_D[final_idx]) * inv_T2;
                }

                W_M[final_idx] = dw_m;
                W_I[final_idx] = dw_i;
                W_D[final_idx] = dw_d;
            }

            // =========== Backward W pass ===========
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
                    size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2) + static_cast<size_t>(j - 1);

                    float betaM = B_M[idx];
                    float betaI = B_I[idx];
                    float betaD = B_D[idx];
                    float wM = W_M[idx];
                    float wI = W_I[idx];
                    float wD = W_D[idx];

                    // M state
                    if (betaM > 1e-20f || std::abs(wM) > 1e-20f) {
                        float m1 = A_M[idx_diag];
                        float m2 = A_I[idx_diag];
                        float m3 = A_D[idx_diag];
                        float w1, w2, w3;

                        if (softmax3(m1, m2, m3, T) > NINF) {
                            softmax3_weights(m1, m2, m3, T, w1, w2, w3);

                            float u1 = U_M[idx_diag];
                            float u2 = U_I[idx_diag];
                            float u3 = U_D[idx_diag];
                            float E_u = w1 * u1 + w2 * u2 + w3 * u3;
                            float dw1 = w1 * (u1 - E_u) / T;
                            float dw2 = w2 * (u2 - E_u) / T;
                            float dw3 = w3 * (u3 - E_u) / T;

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float E_v = w1 * m1 + w2 * m2 + w3 * m3;
                                float inv_T2 = 1.0f / (T * T);
                                dw1 += w1 * (E_v - m1) * inv_T2;
                                dw2 += w2 * (E_v - m2) * inv_T2;
                                dw3 += w3 * (E_v - m3) * inv_T2;
                            }

                            if (m1 > NINF) {
                                B_M[idx_diag] += betaM * w1;
                                W_M[idx_diag] += wM * w1 + betaM * dw1;
                            }
                            if (m2 > NINF) {
                                B_I[idx_diag] += betaM * w2;
                                W_I[idx_diag] += wM * w2 + betaM * dw2;
                            }
                            if (m3 > NINF) {
                                B_D[idx_diag] += betaM * w3;
                                W_D[idx_diag] += wM * w3 + betaM * dw3;
                            }

                            dP[score_idx] += wM;
                        }
                    }

                    // I state
                    if (betaI > 1e-20f || std::abs(wI) > 1e-20f) {
                        float i1 = A_M[idx_up] + gap_open;
                        float i2 = A_I[idx_up] + gap_ext;
                        float i3 = A_D[idx_up] + gap_open;
                        float w1, w2, w3;

                        if (softmax3(i1, i2, i3, T) > NINF) {
                            softmax3_weights(i1, i2, i3, T, w1, w2, w3);

                            float u1 = U_M[idx_up];
                            float u2 = U_I[idx_up];
                            float u3 = U_D[idx_up];

                            if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                                u1 += 1.0f;
                                u3 += 1.0f;
                            } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                                u2 += 1.0f;
                            }

                            float E_u = w1 * u1 + w2 * u2 + w3 * u3;
                            float dw1 = w1 * (u1 - E_u) / T;
                            float dw2 = w2 * (u2 - E_u) / T;
                            float dw3 = w3 * (u3 - E_u) / T;

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float E_v = w1 * i1 + w2 * i2 + w3 * i3;
                                float inv_T2 = 1.0f / (T * T);
                                dw1 += w1 * (E_v - i1) * inv_T2;
                                dw2 += w2 * (E_v - i2) * inv_T2;
                                dw3 += w3 * (E_v - i3) * inv_T2;
                            }

                            B_M[idx_up] += betaI * w1;
                            W_M[idx_up] += wI * w1 + betaI * dw1;
                            B_I[idx_up] += betaI * w2;
                            W_I[idx_up] += wI * w2 + betaI * dw2;
                            B_D[idx_up] += betaI * w3;
                            W_D[idx_up] += wI * w3 + betaI * dw3;
                        }
                    }

                    // D state
                    if (betaD > 1e-20f || std::abs(wD) > 1e-20f) {
                        float d1 = A_M[idx_left] + gap_open;
                        float d2 = A_I[idx_left] + gap_open;
                        float d3 = A_D[idx_left] + gap_ext;
                        float w1, w2, w3;

                        if (softmax3(d1, d2, d3, T) > NINF) {
                            softmax3_weights(d1, d2, d3, T, w1, w2, w3);

                            float u1 = U_M[idx_left];
                            float u2 = U_I[idx_left];
                            float u3 = U_D[idx_left];

                            if (param_type == NW_AFFINE_PARAM_GAP_OPEN) {
                                u1 += 1.0f;
                                u2 += 1.0f;
                            } else if (param_type == NW_AFFINE_PARAM_GAP_EXT) {
                                u3 += 1.0f;
                            }

                            float E_u = w1 * u1 + w2 * u2 + w3 * u3;
                            float dw1 = w1 * (u1 - E_u) / T;
                            float dw2 = w2 * (u2 - E_u) / T;
                            float dw3 = w3 * (u3 - E_u) / T;

                            if (param_type == NW_AFFINE_PARAM_TEMPERATURE) {
                                float E_v = w1 * d1 + w2 * d2 + w3 * d3;
                                float inv_T2 = 1.0f / (T * T);
                                dw1 += w1 * (E_v - d1) * inv_T2;
                                dw2 += w2 * (E_v - d2) * inv_T2;
                                dw3 += w3 * (E_v - d3) * inv_T2;
                            }

                            B_M[idx_left] += betaD * w1;
                            W_M[idx_left] += wD * w1 + betaD * dw1;
                            B_I[idx_left] += betaD * w2;
                            W_I[idx_left] += wD * w2 + betaD * dw2;
                            B_D[idx_left] += betaD * w3;
                            W_D[idx_left] += wD * w3 + betaD * dw3;
                        }
                    }
                }
            }

        }
    });
}
