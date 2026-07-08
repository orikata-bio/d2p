// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft CKY CPU Implementation
 *
 * Pure CPU kernels for differentiable CKY parsing.
 * Mirrors the CUDA kernel interface for seamless dispatch.
 *
 * Operations:
 *   - Forward:    Inside algorithm (partition function over all trees)
 *   - Backward:   Outside algorithm (span/split marginals, all gradients)
 *   - HVP:        Hessian-vector product for second-order optimization
 *   - ParamGrad:  Cross-derivatives dP/dT (posteriors w.r.t. temperature)
 *
 * Shapes:
 *   merge_scores: [B, n, n, n]    - merge score for combining [i,k] and [k+1,j]
 *   leaf_scores:  [B, n]          - score for leaf span [i,i]
 *   Z:            [B, n, n]       - inside values (upper triangular)
 *   beta:         [B, n, n]       - outside values / span marginals
 *   Pcond:        [B, n, n, n]    - conditional split posteriors P(k|i,j)
 *   Pjoint:       [B, n, n, n]    - joint split posteriors beta[i,j] * P(k|i,j)
 */

#include "kernels_cpu.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

// =============================================================================
// Constants and Helper Functions
// =============================================================================

constexpr float NINF = -1e30f;

// Fix #1: Safe exponential with correct bounds for float32
// exp(88) ~= 1.65e38 (within float32 max ~3.4e38)
// exp(-88) ~= 6e-39 (underflows gracefully to 0)
inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

// Fix #2: Kahan compensated summation
// Reduces O(n*epsilon) error to O(epsilon) when summing many floats
struct KahanAccumulator {
    float sum = 0.0f;
    float c = 0.0f;  // Compensation for lost low-order bits

    void add(float value) {
        float y = value - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    float result() const { return sum; }
    void reset() { sum = 0.0f; c = 0.0f; }
};

// Temperature-scaled logsumexp for variable number of terms
// Uses Kahan summation for better precision
inline float logsumexp_T(const std::vector<float>& vals, float T) {
    if (vals.empty()) return NINF;
    float max_v = *std::max_element(vals.begin(), vals.end());
    if (max_v <= NINF) return NINF;

    KahanAccumulator sum;
    for (float v : vals) {
        if (v > NINF) {  // Skip -inf values
            sum.add(safe_exp((v - max_v) / T));
        }
    }
    return max_v + T * std::log(sum.result());
}

// Softmax weights for variable number of terms
// Uses Kahan summation and handles -inf values properly
inline void softmax_T(const std::vector<float>& vals, float T, std::vector<float>& weights) {
    weights.resize(vals.size());
    if (vals.empty()) return;

    float max_v = *std::max_element(vals.begin(), vals.end());
    if (max_v <= NINF) {
        std::fill(weights.begin(), weights.end(), 0.0f);
        return;
    }

    KahanAccumulator sum;
    for (size_t i = 0; i < vals.size(); i++) {
        if (vals[i] > NINF) {
            weights[i] = safe_exp((vals[i] - max_v) / T);
            sum.add(weights[i]);
        } else {
            weights[i] = 0.0f;
        }
    }

    float total = sum.result();
    if (total > 0) {
        for (size_t i = 0; i < vals.size(); i++) {
            weights[i] /= total;
        }
    }
}

inline float cky_temperature_value(
    const float* T,
    int b,
    int n,
    int i,
    int j,
    bool T_is_scalar
) {
    if (T_is_scalar) {
        return T[0];
    }
    return T[(size_t)b * n * n + (size_t)i * n + j];
}

// =============================================================================
// FORWARD PASS (Inside Algorithm)
// =============================================================================

void cky_forward_cpu(
    const float* merge_scores,  // [B, n, n, n]
    const float* leaf_scores,   // [B, n]
    float* Z,                   // [B, n, n] output: inside values
    float* partition,           // [B] output: log partition function
    const float* T,
    int B, int n,
    bool T_is_scalar
) {
    const size_t Z_stride = (size_t)n * n;
    const size_t merge_stride = (size_t)n * n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* leaf = leaf_scores + b * n;
            const float* merge = merge_scores + b * merge_stride;
            float* z = Z + b * Z_stride;

            // Initialize Z to -inf
            for (size_t idx = 0; idx < Z_stride; idx++) {
                z[idx] = NINF;
            }

            // Base case: leaf spans [i,i] for i = 0..n-1
            for (int i = 0; i < n; i++) {
                z[i * n + i] = leaf[i];
            }

            // Bottom-up: increasing span length
            std::vector<float> terms;
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;
                    float T_ij = cky_temperature_value(T, b, n, i, j, T_is_scalar);
                    terms.clear();

                    // Sum over all split points k = i..j-1
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }

                    z[i * n + j] = logsumexp_T(terms, T_ij);
                }
            }

            // Partition function is Z[0, n-1]
            partition[b] = z[0 * n + (n - 1)];
        }
    });
}

// =============================================================================
// BACKWARD PASS (Outside Algorithm + Gradients)
// =============================================================================

void cky_backward_cpu(
    const float* Z,             // [B, n, n] inside values
    const float* merge_scores,  // [B, n, n, n]
    const float* leaf_scores,   // [B, n] (unused but kept for interface)
    const float* partition,     // [B] (unused but kept for interface)
    float* beta,                // [B, n, n] output: outside values
    float* Pcond,               // [B, n, n, n] output: P(k|i,j)
    float* Pjoint,              // [B, n, n, n] output: beta[i,j] * P(k|i,j)
    float* grad_merge,          // [B, n, n, n] output: dZ/d(merge_scores)
    float* grad_leaf,           // [B, n] output: dZ/d(leaf_scores)
    float* grad_T,              // [B] or [B, n, n] output: dZ/dT
    const float* T,
    int B, int n,
    bool T_is_scalar
) {
    const size_t Z_stride = (size_t)n * n;
    const size_t merge_stride = (size_t)n * n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* z = Z + b * Z_stride;
            const float* merge = merge_scores + b * merge_stride;
            float* be = beta + b * Z_stride;
            float* pc = Pcond + b * merge_stride;
            float* pj = Pjoint + b * merge_stride;
            float* gm = grad_merge + b * merge_stride;
            float* gl = grad_leaf + b * n;

            // Initialize
            for (size_t idx = 0; idx < Z_stride; idx++) {
                be[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < merge_stride; idx++) {
                pc[idx] = 0.0f;
                pj[idx] = 0.0f;
                gm[idx] = 0.0f;
            }
            for (int i = 0; i < n; i++) {
                gl[i] = 0.0f;
            }
            if (T_is_scalar) {
                grad_T[b] = 0.0f;
            } else {
                std::fill(
                    grad_T + (size_t)b * n * n,
                    grad_T + (size_t)(b + 1) * n * n,
                    0.0f
                );
            }

            // Root span [0, n-1] has beta = 1
            be[0 * n + (n - 1)] = 1.0f;

            // Compute Pcond for all spans first (need Z values)
            std::vector<float> terms, weights;
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;
                    float T_ij = cky_temperature_value(T, b, n, i, j, T_is_scalar);
                    terms.clear();

                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }

                    softmax_T(terms, T_ij, weights);

                    for (int k = i; k < j; k++) {
                        pc[(size_t)i * n * n + (size_t)k * n + j] = weights[k - i];
                    }
                }
            }

            // Top-down: decreasing span length
            for (int len = n; len >= 2; len--) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;
                    float beta_ij = be[i * n + j];
                    float T_ij = cky_temperature_value(T, b, n, i, j, T_is_scalar);

                    if (beta_ij == 0.0f) continue;

                    // Compute Pjoint and propagate beta to children
                    for (int k = i; k < j; k++) {
                        float p = pc[(size_t)i * n * n + (size_t)k * n + j];
                        float mass = beta_ij * p;

                        pj[(size_t)i * n * n + (size_t)k * n + j] = mass;

                        // Gradient w.r.t. merge_scores
                        gm[(size_t)i * n * n + (size_t)k * n + j] = mass;

                        // Propagate beta to children
                        be[i * n + k] += mass;
                        be[(k + 1) * n + j] += mass;
                    }

                    // Temperature gradient contribution
                    float Zij = z[i * n + j];
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }

                    float E_term = 0.0f;
                    for (int k = i; k < j; k++) {
                        float p = pc[(size_t)i * n * n + (size_t)k * n + j];
                        E_term += p * terms[k - i];
                    }

                    float grad_T_ij = beta_ij * (Zij - E_term) / T_ij;
                    if (T_is_scalar) {
                        grad_T[b] += grad_T_ij;
                    } else {
                        grad_T[(size_t)b * n * n + (size_t)i * n + j] = grad_T_ij;
                    }
                }
            }

            // Leaf gradients: beta[i,i] flows to leaf_scores[i]
            for (int i = 0; i < n; i++) {
                gl[i] = be[i * n + i];
            }
        }
    });
}

// =============================================================================
// HESSIAN-VECTOR PRODUCT (HVP)
// =============================================================================

void cky_hvp_cpu(
    const float* Z,             // [B, n, n]
    const float* merge_scores,  // [B, n, n, n]
    const float* leaf_scores,   // [B, n] (unused)
    const float* partition,     // [B] (unused)
    const float* V_merge,       // [B, n, n, n] tangent for merge_scores
    const float* V_leaf,        // [B, n] tangent for leaf_scores
    float* d_Z,                 // [B, n, n] workspace: tangent of Z
    float* d_partition,         // [B] workspace
    float* beta,                // [B, n, n] workspace
    float* d_beta,              // [B, n, n] workspace: tangent of beta
    float* HVP_merge,           // [B, n, n, n] output: H * V for merge
    float* HVP_leaf,            // [B, n] output: H * V for leaf
    int B, int n, float T
) {
    const size_t Z_stride = (size_t)n * n;
    const size_t merge_stride = (size_t)n * n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* z = Z + b * Z_stride;
            const float* merge = merge_scores + b * merge_stride;
            const float* v_merge = V_merge + b * merge_stride;
            const float* v_leaf = V_leaf + b * n;
            float* dz = d_Z + b * Z_stride;
            float* be = beta + b * Z_stride;
            float* dbe = d_beta + b * Z_stride;
            float* hm = HVP_merge + b * merge_stride;
            float* hl = HVP_leaf + b * n;

            // Initialize
            for (size_t idx = 0; idx < Z_stride; idx++) {
                dz[idx] = 0.0f;
                be[idx] = 0.0f;
                dbe[idx] = 0.0f;
            }
            for (size_t idx = 0; idx < merge_stride; idx++) {
                hm[idx] = 0.0f;
            }
            for (int i = 0; i < n; i++) {
                hl[i] = 0.0f;
            }

            // ========== Forward pass for d_Z (tangent of inside values) ==========

            // Base case: d_Z[i,i] = V_leaf[i]
            for (int i = 0; i < n; i++) {
                dz[i * n + i] = v_leaf[i];
            }

            // Bottom-up
            std::vector<float> terms, weights;
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;

                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }
                    softmax_T(terms, T, weights);

                    float d_zij = 0.0f;
                    for (int k = i; k < j; k++) {
                        float d_left = dz[i * n + k];
                        float d_right = dz[(k + 1) * n + j];
                        float d_ms = v_merge[(size_t)i * n * n + (size_t)k * n + j];
                        d_zij += weights[k - i] * (d_left + d_right + d_ms);
                    }
                    dz[i * n + j] = d_zij;
                }
            }

            d_partition[b] = dz[0 * n + (n - 1)];

            // ========== Backward pass for beta and d_beta ==========

            be[0 * n + (n - 1)] = 1.0f;
            dbe[0 * n + (n - 1)] = 0.0f;

            // Top-down
            for (int len = n; len >= 2; len--) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;
                    float beta_ij = be[i * n + j];
                    float d_beta_ij = dbe[i * n + j];

                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }
                    softmax_T(terms, T, weights);

                    // Compute d_weights (derivative of softmax)
                    std::vector<float> d_terms(j - i);
                    float weighted_d_sum = 0.0f;
                    for (int k = i; k < j; k++) {
                        float d_left = dz[i * n + k];
                        float d_right = dz[(k + 1) * n + j];
                        float d_ms = v_merge[(size_t)i * n * n + (size_t)k * n + j];
                        d_terms[k - i] = d_left + d_right + d_ms;
                        weighted_d_sum += weights[k - i] * d_terms[k - i];
                    }

                    std::vector<float> d_weights(j - i);
                    for (int k = i; k < j; k++) {
                        d_weights[k - i] = weights[k - i] * (d_terms[k - i] - weighted_d_sum) / T;
                    }

                    // Propagate to children and accumulate HVP
                    for (int k = i; k < j; k++) {
                        float w = weights[k - i];
                        float dw = d_weights[k - i];

                        float mass = beta_ij * w;
                        float d_mass = d_beta_ij * w + beta_ij * dw;

                        hm[(size_t)i * n * n + (size_t)k * n + j] = d_mass;

                        be[i * n + k] += mass;
                        be[(k + 1) * n + j] += mass;
                        dbe[i * n + k] += d_mass;
                        dbe[(k + 1) * n + j] += d_mass;
                    }
                }
            }

            // HVP for leaf_scores
            for (int i = 0; i < n; i++) {
                hl[i] = dbe[i * n + i];
            }
        }
    });
}

// =============================================================================
// PARAMETER GRADIENT (dP/dT)
// =============================================================================

void cky_param_grad_cpu(
    const float* Z,             // [B, n, n]
    const float* merge_scores,  // [B, n, n, n]
    const float* leaf_scores,   // [B, n] (unused)
    const float* partition,     // [B] (unused)
    float* dP_dT_merge,         // [B, n, n, n] output
    float* dP_dT_leaf,          // [B, n] output
    int B, int n, float T
) {
    const size_t Z_stride = (size_t)n * n;
    const size_t merge_stride = (size_t)n * n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* z = Z + b * Z_stride;
            const float* merge = merge_scores + b * merge_stride;
            float* dpm = dP_dT_merge + b * merge_stride;
            float* dpl = dP_dT_leaf + b * n;

            // Workspace
            std::vector<float> U(Z_stride, 0.0f);
            std::vector<float> beta(Z_stride, 0.0f);
            std::vector<float> W(Z_stride, 0.0f);

            // Initialize outputs
            for (size_t idx = 0; idx < merge_stride; idx++) {
                dpm[idx] = 0.0f;
            }
            for (int i = 0; i < n; i++) {
                dpl[i] = 0.0f;
            }

            // ========== Forward: compute U = dZ/dT ==========
            std::vector<float> terms, weights;
            for (int len = 2; len <= n; len++) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;

                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }
                    softmax_T(terms, T, weights);

                    float Zij = z[i * n + j];
                    float E_term = 0.0f;
                    float E_U = 0.0f;
                    for (int k = i; k < j; k++) {
                        float u_left = U[i * n + k];
                        float u_right = U[(k + 1) * n + j];
                        E_term += weights[k - i] * terms[k - i];
                        E_U += weights[k - i] * (u_left + u_right);
                    }

                    U[i * n + j] = (Zij - E_term) / T + E_U;
                }
            }

            // ========== Backward: compute W = d_beta/dT ==========
            beta[0 * n + (n - 1)] = 1.0f;
            W[0 * n + (n - 1)] = 0.0f;

            for (int len = n; len >= 2; len--) {
                for (int i = 0; i + len - 1 < n; i++) {
                    int j = i + len - 1;
                    float beta_ij = beta[i * n + j];
                    float w_ij = W[i * n + j];

                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float left = z[i * n + k];
                        float right = z[(k + 1) * n + j];
                        float ms = merge[(size_t)i * n * n + (size_t)k * n + j];
                        terms.push_back(left + right + ms);
                    }
                    softmax_T(terms, T, weights);

                    // Compute d_weights/dT
                    std::vector<float> d_weights_dT(j - i);
                    float E_term = 0.0f;
                    for (int k = i; k < j; k++) {
                        E_term += weights[k - i] * terms[k - i];
                    }

                    for (int k = i; k < j; k++) {
                        float u_left = U[i * n + k];
                        float u_right = U[(k + 1) * n + j];
                        float d_term_dT = u_left + u_right;

                        float E_d_term = 0.0f;
                        for (int kk = i; kk < j; kk++) {
                            float ul = U[i * n + kk];
                            float ur = U[(kk + 1) * n + j];
                            E_d_term += weights[kk - i] * (ul + ur);
                        }

                        float diff = terms[k - i] - E_term;
                        d_weights_dT[k - i] = weights[k - i] * (
                            -diff / (T * T) + (d_term_dT - E_d_term) / T
                        );
                    }

                    // Propagate
                    for (int k = i; k < j; k++) {
                        float w = weights[k - i];
                        float dw_dT = d_weights_dT[k - i];

                        dpm[(size_t)i * n * n + (size_t)k * n + j] = w_ij * w + beta_ij * dw_dT;

                        float mass = beta_ij * w;
                        float d_mass = w_ij * w + beta_ij * dw_dT;

                        beta[i * n + k] += mass;
                        beta[(k + 1) * n + j] += mass;
                        W[i * n + k] += d_mass;
                        W[(k + 1) * n + j] += d_mass;
                    }
                }
            }

            // Leaf gradients
            for (int i = 0; i < n; i++) {
                dpl[i] = W[i * n + i];
            }
        }
    });
}
