// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernels_cpu.cpp
 * @brief Soft Eisner CPU Kernel Implementations
 *
 * Pure CPU kernels for differentiable projective dependency parsing.
 * Mirrors the CUDA kernel interface for seamless dispatch.
 */

#include "kernels_cpu.h"
#include <cmath>
#include <algorithm>
#include <vector>

#include <ATen/Parallel.h>

namespace d2p {
namespace eisner {
namespace cpu {

// =============================================================================
// Helper Functions
// =============================================================================

inline float safe_exp(float x) {
    if (x < -88.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    return std::exp(x);
}

inline size_t cell_offset(int row, int col, int n) {
    return static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
}

// Kahan compensated summation for better numerical precision
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
    void reset() { sum = 0.0f; c = 0.0f; }
};

// Temperature-scaled logsumexp
inline float logsumexp_T(const std::vector<float>& vals, float T) {
    if (vals.empty()) return NINF;
    float max_v = *std::max_element(vals.begin(), vals.end());
    if (max_v <= NINF) return NINF;

    KahanAccumulator sum;
    for (float v : vals) {
        if (v > NINF) {
            sum.add(safe_exp((v - max_v) / T));
        }
    }
    return max_v + T * std::log(sum.result());
}

// Compute softmax weights
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

// =============================================================================
// Forward Pass
// =============================================================================

void forward(
    const float* arc_scores,
    float* C_R,
    float* C_L,
    float* I_R,
    float* I_L,
    float* partition,
    const int* lengths,
    int B, int n, float T
) {
    const size_t stride = (size_t)n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* arc = arc_scores + b * stride;
            float* cr = C_R + b * stride;
            float* cl = C_L + b * stride;
            float* ir = I_R + b * stride;
            float* il = I_L + b * stride;

            int seq_len = lengths ? lengths[b] : n;

            // Initialize tables
            for (size_t idx = 0; idx < stride; idx++) {
                cr[idx] = NINF;
                cl[idx] = NINF;
                ir[idx] = NINF;
                il[idx] = NINF;
            }

            // Base case: C[i,i] = 0
            for (int i = 0; i < seq_len; i++) {
                cr[cell_offset(i, i, n)] = 0.0f;
                cl[cell_offset(i, i, n)] = 0.0f;
            }

            // Process spans by increasing length
            std::vector<float> terms;
            for (int len = 1; len < seq_len; len++) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Incomplete spans: I_R[i,j] and I_L[i,j]
                    // I_R[i,j] = arc[i,j] + LSE_k{ C_R[i,k] + C_L[k+1,j] }
                    // I_L[i,j] = arc[j,i] + LSE_k{ C_R[i,k] + C_L[k+1,j] }
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float v = cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)];
                        terms.push_back(v);
                    }
                    float lse = logsumexp_T(terms, T);
                    ir[cell_offset(i, j, n)] = arc[cell_offset(i, j, n)] + lse;
                    il[cell_offset(i, j, n)] = arc[cell_offset(j, i, n)] + lse;

                    // Complete spans: C_R[i,j] and C_L[i,j]
                    // C_R[i,j] = LSE_k{ C_R[i,k] + I_R[k,j] }  for k in [i, j)
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float v = cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)];
                        terms.push_back(v);
                    }
                    cr[cell_offset(i, j, n)] = logsumexp_T(terms, T);

                    // C_L[i,j] = LSE_k{ I_L[i,k] + C_L[k,j] }  for k in (i, j]
                    terms.clear();
                    for (int k = i + 1; k <= j; k++) {
                        float v = il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)];
                        terms.push_back(v);
                    }
                    cl[cell_offset(i, j, n)] = logsumexp_T(terms, T);
                }
            }

            // Partition = C_R[0, seq_len-1]
            partition[b] = cr[cell_offset(0, seq_len - 1, n)];
        }
    });
}

// =============================================================================
// Backward Pass
// =============================================================================

void backward(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* marginals,
    float* grad_T,
    const int* lengths,
    int B, int n, float T
) {
    const size_t stride = (size_t)n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* arc = arc_scores + b * stride;
            const float* cr = C_R + b * stride;
            const float* cl = C_L + b * stride;
            const float* ir = I_R + b * stride;
            const float* il = I_L + b * stride;
            float* bcr = beta_C_R + b * stride;
            float* bcl = beta_C_L + b * stride;
            float* bir = beta_I_R + b * stride;
            float* bil = beta_I_L + b * stride;
            float* marg = marginals + b * stride;

            int seq_len = lengths ? lengths[b] : n;

            // Initialize
            for (size_t idx = 0; idx < stride; idx++) {
                bcr[idx] = 0.0f;
                bcl[idx] = 0.0f;
                bir[idx] = 0.0f;
                bil[idx] = 0.0f;
                marg[idx] = 0.0f;
            }

            // Root span beta = 1
            bcr[cell_offset(0, seq_len - 1, n)] = 1.0f;

            float sum_grad_T = 0.0f;
            std::vector<float> terms, weights;

            // Top-down by decreasing span length
            for (int len = seq_len - 1; len >= 1; len--) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Backward for C_R[i,j] = LSE_k{ C_R[i,k] + I_R[k,j] }
                    float beta_cr_ij = bcr[cell_offset(i, j, n)];
                    if (beta_cr_ij != 0.0f) {
                        terms.clear();
                        for (int k = i; k < j; k++) {
                            float v = cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)];
                            terms.push_back(v);
                        }
                        softmax_T(terms, T, weights);

                        float Zij = cr[cell_offset(i, j, n)];
                        float E_term = 0.0f;
                        for (int k = i; k < j; k++) {
                            float mass = beta_cr_ij * weights[k - i];
                            bcr[cell_offset(i, k, n)] += mass;
                            bir[cell_offset(k, j, n)] += mass;
                            E_term += weights[k - i] * terms[k - i];
                        }
                        sum_grad_T += beta_cr_ij * (Zij - E_term) / T;
                    }

                    // Backward for C_L[i,j] = LSE_k{ I_L[i,k] + C_L[k,j] }
                    float beta_cl_ij = bcl[cell_offset(i, j, n)];
                    if (beta_cl_ij != 0.0f) {
                        terms.clear();
                        for (int k = i + 1; k <= j; k++) {
                            float v = il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)];
                            terms.push_back(v);
                        }
                        softmax_T(terms, T, weights);

                        float Zij = cl[cell_offset(i, j, n)];
                        float E_term = 0.0f;
                        for (int k = i + 1; k <= j; k++) {
                            float mass = beta_cl_ij * weights[k - i - 1];
                            bil[cell_offset(i, k, n)] += mass;
                            bcl[cell_offset(k, j, n)] += mass;
                            E_term += weights[k - i - 1] * terms[k - i - 1];
                        }
                        sum_grad_T += beta_cl_ij * (Zij - E_term) / T;
                    }

                    // Backward for incomplete spans
                    float beta_ir_ij = bir[cell_offset(i, j, n)];
                    float beta_il_ij = bil[cell_offset(i, j, n)];

                    // Arc marginals
                    marg[cell_offset(i, j, n)] = beta_ir_ij;
                    marg[cell_offset(j, i, n)] = beta_il_ij;

                    float beta_combined = beta_ir_ij + beta_il_ij;
                    if (beta_combined != 0.0f) {
                        terms.clear();
                        for (int k = i; k < j; k++) {
                            float v = cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)];
                            terms.push_back(v);
                        }
                        softmax_T(terms, T, weights);

                        float lse = logsumexp_T(terms, T);
                        float E_term = 0.0f;
                        for (int k = i; k < j; k++) {
                            float mass = beta_combined * weights[k - i];
                            bcr[cell_offset(i, k, n)] += mass;
                            bcl[cell_offset(k + 1, j, n)] += mass;
                            E_term += weights[k - i] * terms[k - i];
                        }
                        sum_grad_T += beta_combined * (lse - E_term) / T;
                    }
                }
            }

            grad_T[b] = sum_grad_T;
        }
    });
}

// =============================================================================
// Hessian-Vector Product (HVP)
// =============================================================================

void hvp(
    const float* arc_scores,
    const float* V,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* d_C_R,
    float* d_C_L,
    float* d_I_R,
    float* d_I_L,
    float* beta_C_R,
    float* beta_C_L,
    float* beta_I_R,
    float* beta_I_L,
    float* d_beta_C_R,
    float* d_beta_C_L,
    float* d_beta_I_R,
    float* d_beta_I_L,
    float* HVP,
    const int* lengths,
    int B, int n, float T
) {
    const size_t stride = (size_t)n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* arc = arc_scores + b * stride;
            const float* v = V + b * stride;
            const float* cr = C_R + b * stride;
            const float* cl = C_L + b * stride;
            const float* ir = I_R + b * stride;
            const float* il = I_L + b * stride;
            float* dcr = d_C_R + b * stride;
            float* dcl = d_C_L + b * stride;
            float* dir = d_I_R + b * stride;
            float* dil = d_I_L + b * stride;
            float* bcr = beta_C_R + b * stride;
            float* bcl = beta_C_L + b * stride;
            float* bir = beta_I_R + b * stride;
            float* bil = beta_I_L + b * stride;
            float* dbcr = d_beta_C_R + b * stride;
            float* dbcl = d_beta_C_L + b * stride;
            float* dbir = d_beta_I_R + b * stride;
            float* dbil = d_beta_I_L + b * stride;
            float* hvp_out = HVP + b * stride;

            int seq_len = lengths ? lengths[b] : n;

            // Initialize
            for (size_t idx = 0; idx < stride; idx++) {
                dcr[idx] = 0.0f;
                dcl[idx] = 0.0f;
                dir[idx] = 0.0f;
                dil[idx] = 0.0f;
                bcr[idx] = 0.0f;
                bcl[idx] = 0.0f;
                bir[idx] = 0.0f;
                bil[idx] = 0.0f;
                dbcr[idx] = 0.0f;
                dbcl[idx] = 0.0f;
                dbir[idx] = 0.0f;
                dbil[idx] = 0.0f;
                hvp_out[idx] = 0.0f;
            }

            // Base case
            for (int i = 0; i < seq_len; i++) {
                dcr[cell_offset(i, i, n)] = 0.0f;
                dcl[cell_offset(i, i, n)] = 0.0f;
            }
            bcr[cell_offset(0, seq_len - 1, n)] = 1.0f;

            std::vector<float> terms, weights, d_terms;

            // Forward pass for tangents
            for (int len = 1; len < seq_len; len++) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Tangent for incomplete spans
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float val = cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)];
                        terms.push_back(val);
                    }
                    softmax_T(terms, T, weights);

                    float d_lse = 0.0f;
                    for (int k = i; k < j; k++) {
                        d_lse += weights[k - i] * (
                            dcr[cell_offset(i, k, n)] + dcl[cell_offset(k + 1, j, n)]
                        );
                    }
                    dir[cell_offset(i, j, n)] = v[cell_offset(i, j, n)] + d_lse;
                    dil[cell_offset(i, j, n)] = v[cell_offset(j, i, n)] + d_lse;

                    // Tangent for C_R[i,j]
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        float val = cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)];
                        terms.push_back(val);
                    }
                    softmax_T(terms, T, weights);

                    d_lse = 0.0f;
                    for (int k = i; k < j; k++) {
                        d_lse += weights[k - i] * (
                            dcr[cell_offset(i, k, n)] + dir[cell_offset(k, j, n)]
                        );
                    }
                    dcr[cell_offset(i, j, n)] = d_lse;

                    // Tangent for C_L[i,j]
                    terms.clear();
                    for (int k = i + 1; k <= j; k++) {
                        float val = il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)];
                        terms.push_back(val);
                    }
                    softmax_T(terms, T, weights);

                    d_lse = 0.0f;
                    for (int k = i + 1; k <= j; k++) {
                        d_lse += weights[k - i - 1] * (
                            dil[cell_offset(i, k, n)] + dcl[cell_offset(k, j, n)]
                        );
                    }
                    dcl[cell_offset(i, j, n)] = d_lse;
                }
            }

            // Backward pass
            for (int len = seq_len - 1; len >= 1; len--) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Backward for C_R[i,j]
                    float beta_cr_ij = bcr[cell_offset(i, j, n)];
                    float d_beta_cr_ij = dbcr[cell_offset(i, j, n)];

                    if (beta_cr_ij != 0.0f || d_beta_cr_ij != 0.0f) {
                        terms.clear();
                        d_terms.clear();
                        for (int k = i; k < j; k++) {
                            float val = cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)];
                            terms.push_back(val);
                            d_terms.push_back(
                                dcr[cell_offset(i, k, n)] + dir[cell_offset(k, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_d_term = 0.0f;
                        for (int k = i; k < j; k++) {
                            E_d_term += weights[k - i] * d_terms[k - i];
                        }

                        for (int k = i; k < j; k++) {
                            float w = weights[k - i];
                            float d_w = w * (d_terms[k - i] - E_d_term) / T;

                            float mass = beta_cr_ij * w;
                            float d_mass = d_beta_cr_ij * w + beta_cr_ij * d_w;

                            bcr[cell_offset(i, k, n)] += mass;
                            bir[cell_offset(k, j, n)] += mass;
                            dbcr[cell_offset(i, k, n)] += d_mass;
                            dbir[cell_offset(k, j, n)] += d_mass;
                        }
                    }

                    // Backward for C_L[i,j]
                    float beta_cl_ij = bcl[cell_offset(i, j, n)];
                    float d_beta_cl_ij = dbcl[cell_offset(i, j, n)];

                    if (beta_cl_ij != 0.0f || d_beta_cl_ij != 0.0f) {
                        terms.clear();
                        d_terms.clear();
                        for (int k = i + 1; k <= j; k++) {
                            float val = il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)];
                            terms.push_back(val);
                            d_terms.push_back(
                                dil[cell_offset(i, k, n)] + dcl[cell_offset(k, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_d_term = 0.0f;
                        for (int k = i + 1; k <= j; k++) {
                            E_d_term += weights[k - i - 1] * d_terms[k - i - 1];
                        }

                        for (int k = i + 1; k <= j; k++) {
                            float w = weights[k - i - 1];
                            float d_w = w * (d_terms[k - i - 1] - E_d_term) / T;

                            float mass = beta_cl_ij * w;
                            float d_mass = d_beta_cl_ij * w + beta_cl_ij * d_w;

                            bil[cell_offset(i, k, n)] += mass;
                            bcl[cell_offset(k, j, n)] += mass;
                            dbil[cell_offset(i, k, n)] += d_mass;
                            dbcl[cell_offset(k, j, n)] += d_mass;
                        }
                    }

                    // Backward for incomplete spans
                    float beta_ir = bir[cell_offset(i, j, n)];
                    float beta_il = bil[cell_offset(i, j, n)];
                    float d_beta_ir = dbir[cell_offset(i, j, n)];
                    float d_beta_il = dbil[cell_offset(i, j, n)];

                    // HVP output
                    hvp_out[cell_offset(i, j, n)] = d_beta_ir;
                    hvp_out[cell_offset(j, i, n)] = d_beta_il;

                    float beta_combined = beta_ir + beta_il;
                    float d_beta_combined = d_beta_ir + d_beta_il;

                    if (beta_combined != 0.0f || d_beta_combined != 0.0f) {
                        terms.clear();
                        d_terms.clear();
                        for (int k = i; k < j; k++) {
                            float val = cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)];
                            terms.push_back(val);
                            d_terms.push_back(
                                dcr[cell_offset(i, k, n)] + dcl[cell_offset(k + 1, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_d_term = 0.0f;
                        for (int k = i; k < j; k++) {
                            E_d_term += weights[k - i] * d_terms[k - i];
                        }

                        for (int k = i; k < j; k++) {
                            float w = weights[k - i];
                            float d_w = w * (d_terms[k - i] - E_d_term) / T;

                            float mass = beta_combined * w;
                            float d_mass = d_beta_combined * w + beta_combined * d_w;

                            bcr[cell_offset(i, k, n)] += mass;
                            bcl[cell_offset(k + 1, j, n)] += mass;
                            dbcr[cell_offset(i, k, n)] += d_mass;
                            dbcl[cell_offset(k + 1, j, n)] += d_mass;
                        }
                    }
                }
            }
        }
    });
}

// =============================================================================
// Parameter Gradient (dP/dT)
// =============================================================================

void param_grad(
    const float* arc_scores,
    const float* C_R,
    const float* C_L,
    const float* I_R,
    const float* I_L,
    float* dP_dT,
    const int* lengths,
    int B, int n, float T
) {
    const size_t stride = (size_t)n * n;

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t batch = begin; batch < end; ++batch) {
            const int b = static_cast<int>(batch);
            const float* cr = C_R + b * stride;
            const float* cl = C_L + b * stride;
            const float* ir = I_R + b * stride;
            const float* il = I_L + b * stride;
            float* dp_dt = dP_dT + b * stride;

            int seq_len = lengths ? lengths[b] : n;

            // Workspace
            std::vector<float> U_CR(stride, 0.0f), U_CL(stride, 0.0f);
            std::vector<float> U_IR(stride, 0.0f), U_IL(stride, 0.0f);
            std::vector<float> b_CR(stride, 0.0f), b_CL(stride, 0.0f);
            std::vector<float> b_IR(stride, 0.0f), b_IL(stride, 0.0f);
            std::vector<float> W_CR(stride, 0.0f), W_CL(stride, 0.0f);
            std::vector<float> W_IR(stride, 0.0f), W_IL(stride, 0.0f);

            for (size_t idx = 0; idx < stride; idx++) {
                dp_dt[idx] = 0.0f;
            }

            // ========== Forward: compute U = dZ/dT ==========
            std::vector<float> terms, weights;

            for (int len = 1; len < seq_len; len++) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Incomplete spans: I_R[i,j] = arc[i,j] + LSE_T{C_R[i,k] + C_L[k+1,j]}
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        terms.push_back(
                            cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)]
                        );
                    }
                    softmax_T(terms, T, weights);

                    float lse = logsumexp_T(terms, T);
                    float E_term = 0.0f;
                    float E_U = 0.0f;
                    for (int k = i; k < j; k++) {
                        E_term += weights[k - i] * terms[k - i];
                        E_U += weights[k - i] * (
                            U_CR[cell_offset(i, k, n)] + U_CL[cell_offset(k + 1, j, n)]
                        );
                    }
                    float U_lse = (lse - E_term) / T + E_U;
                    U_IR[cell_offset(i, j, n)] = U_lse;
                    U_IL[cell_offset(i, j, n)] = U_lse;

                    // C_R[i,j] = LSE_T{C_R[i,k] + I_R[k,j]}
                    terms.clear();
                    for (int k = i; k < j; k++) {
                        terms.push_back(
                            cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)]
                        );
                    }
                    softmax_T(terms, T, weights);

                    float Zij_cr = cr[cell_offset(i, j, n)];
                    E_term = 0.0f;
                    E_U = 0.0f;
                    for (int k = i; k < j; k++) {
                        E_term += weights[k - i] * terms[k - i];
                        E_U += weights[k - i] * (
                            U_CR[cell_offset(i, k, n)] + U_IR[cell_offset(k, j, n)]
                        );
                    }
                    U_CR[cell_offset(i, j, n)] = (Zij_cr - E_term) / T + E_U;

                    // C_L[i,j] = LSE_T{I_L[i,k] + C_L[k,j]}
                    terms.clear();
                    for (int k = i + 1; k <= j; k++) {
                        terms.push_back(
                            il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)]
                        );
                    }
                    softmax_T(terms, T, weights);

                    float Zij_cl = cl[cell_offset(i, j, n)];
                    E_term = 0.0f;
                    E_U = 0.0f;
                    for (int k = i + 1; k <= j; k++) {
                        E_term += weights[k - i - 1] * terms[k - i - 1];
                        E_U += weights[k - i - 1] * (
                            U_IL[cell_offset(i, k, n)] + U_CL[cell_offset(k, j, n)]
                        );
                    }
                    U_CL[cell_offset(i, j, n)] = (Zij_cl - E_term) / T + E_U;
                }
            }

            // ========== Backward: compute W = dbeta/dT and extract dP/dT ==========
            b_CR[cell_offset(0, seq_len - 1, n)] = 1.0f;

            for (int len = seq_len - 1; len >= 1; len--) {
                for (int i = 0; i + len < seq_len; i++) {
                    int j = i + len;

                    // Backward for C_R[i,j] = LSE_T{C_R[i,k] + I_R[k,j]}
                    float beta_cr = b_CR[cell_offset(i, j, n)];
                    float w_cr = W_CR[cell_offset(i, j, n)];

                    if (beta_cr != 0.0f || w_cr != 0.0f) {
                        terms.clear();
                        for (int k = i; k < j; k++) {
                            terms.push_back(
                                cr[cell_offset(i, k, n)] + ir[cell_offset(k, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_term = 0.0f;
                        float E_U_child = 0.0f;
                        for (int k = i; k < j; k++) {
                            E_term += weights[k - i] * terms[k - i];
                            E_U_child += weights[k - i] * (
                                U_CR[cell_offset(i, k, n)] + U_IR[cell_offset(k, j, n)]
                            );
                        }

                        for (int k = i; k < j; k++) {
                            float w = weights[k - i];
                            float diff = terms[k - i] - E_term;
                            float U_child =
                                U_CR[cell_offset(i, k, n)] + U_IR[cell_offset(k, j, n)];
                            float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                            float d_mass = w_cr * w + beta_cr * dw_dT;

                            b_CR[cell_offset(i, k, n)] += beta_cr * w;
                            b_IR[cell_offset(k, j, n)] += beta_cr * w;
                            W_CR[cell_offset(i, k, n)] += d_mass;
                            W_IR[cell_offset(k, j, n)] += d_mass;
                        }
                    }

                    // Backward for C_L[i,j] = LSE_T{I_L[i,k] + C_L[k,j]}
                    float beta_cl = b_CL[cell_offset(i, j, n)];
                    float w_cl = W_CL[cell_offset(i, j, n)];

                    if (beta_cl != 0.0f || w_cl != 0.0f) {
                        terms.clear();
                        for (int k = i + 1; k <= j; k++) {
                            terms.push_back(
                                il[cell_offset(i, k, n)] + cl[cell_offset(k, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_term = 0.0f;
                        float E_U_child = 0.0f;
                        for (int k = i + 1; k <= j; k++) {
                            E_term += weights[k - i - 1] * terms[k - i - 1];
                            E_U_child += weights[k - i - 1] * (
                                U_IL[cell_offset(i, k, n)] + U_CL[cell_offset(k, j, n)]
                            );
                        }

                        for (int k = i + 1; k <= j; k++) {
                            float w = weights[k - i - 1];
                            float diff = terms[k - i - 1] - E_term;
                            float U_child =
                                U_IL[cell_offset(i, k, n)] + U_CL[cell_offset(k, j, n)];
                            float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                            float d_mass = w_cl * w + beta_cl * dw_dT;

                            b_IL[cell_offset(i, k, n)] += beta_cl * w;
                            b_CL[cell_offset(k, j, n)] += beta_cl * w;
                            W_IL[cell_offset(i, k, n)] += d_mass;
                            W_CL[cell_offset(k, j, n)] += d_mass;
                        }
                    }

                    // Extract arc marginal derivatives
                    dp_dt[cell_offset(i, j, n)] = W_IR[cell_offset(i, j, n)];
                    dp_dt[cell_offset(j, i, n)] = W_IL[cell_offset(i, j, n)];

                    // Backward for incomplete spans
                    float beta_ir = b_IR[cell_offset(i, j, n)];
                    float beta_il = b_IL[cell_offset(i, j, n)];
                    float w_ir = W_IR[cell_offset(i, j, n)];
                    float w_il = W_IL[cell_offset(i, j, n)];

                    float beta_combined = beta_ir + beta_il;
                    float w_combined = w_ir + w_il;

                    if (beta_combined != 0.0f || w_combined != 0.0f) {
                        terms.clear();
                        for (int k = i; k < j; k++) {
                            terms.push_back(
                                cr[cell_offset(i, k, n)] + cl[cell_offset(k + 1, j, n)]
                            );
                        }
                        softmax_T(terms, T, weights);

                        float E_term = 0.0f;
                        float E_U_child = 0.0f;
                        for (int k = i; k < j; k++) {
                            E_term += weights[k - i] * terms[k - i];
                            E_U_child += weights[k - i] * (
                                U_CR[cell_offset(i, k, n)] +
                                U_CL[cell_offset(k + 1, j, n)]
                            );
                        }

                        for (int k = i; k < j; k++) {
                            float w = weights[k - i];
                            float diff = terms[k - i] - E_term;
                            float U_child =
                                U_CR[cell_offset(i, k, n)] + U_CL[cell_offset(k + 1, j, n)];
                            float dw_dT = w * (-diff / (T * T) + (U_child - E_U_child) / T);

                            float d_mass = w_combined * w + beta_combined * dw_dT;

                            b_CR[cell_offset(i, k, n)] += beta_combined * w;
                            b_CL[cell_offset(k + 1, j, n)] += beta_combined * w;
                            W_CR[cell_offset(i, k, n)] += d_mass;
                            W_CL[cell_offset(k + 1, j, n)] += d_mass;
                        }
                    }
                }
            }
        }
    });
}

} // namespace cpu
} // namespace eisner
} // namespace d2p
