# SPDX-License-Identifier: Apache-2.0
"""
FakeTensor registrations for torch.compile support.

This module registers "fake" (meta) implementations for all d2p operators,
enabling torch.compile, torch.export, and FX tracing to work correctly.

The fake implementations describe output tensor shapes without running
the actual CUDA kernels.
"""

import torch
from torch.library import register_fake

# Ensure extension is loaded
from . import _ops  # noqa: F401


# =============================================================================
# Smith-Waterman (Regular - Linear Gap)
# =============================================================================


@register_fake("d2p::soft_sw_float")
def soft_sw_float_fake(scores, gap, temperature, lengths):
    """soft_sw_float returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_sw")
def soft_sw_fake(scores, gap, temperature, lengths):
    """soft_sw (tensor params) returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_sw_with_grads")
def soft_sw_with_grads_fake(scores, gap, temperature, lengths):
    """Returns (score, alignment, grad_gap, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_sw_hvp")
def soft_sw_hvp_fake(scores, tangent, gap, temperature, lengths):
    """Returns Hessian-vector product with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_sw_param_jacobian")
def soft_sw_param_jacobian_fake(scores, param_type, gap, temperature, lengths):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_sw_backward_full")
def soft_sw_backward_full_fake(scores, grad_alignment, gap, temperature, lengths):
    """Returns (grad_scores, grad_gap, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


# =============================================================================
# Smith-Waterman (Affine Gap)
# =============================================================================


@register_fake("d2p::soft_sw_affine_float")
def soft_sw_affine_float_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_sw_affine")
def soft_sw_affine_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_sw_affine_with_grads")
def soft_sw_affine_with_grads_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns (score, alignment, grad_open, grad_ext, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_sw_affine_hvp")
def soft_sw_affine_hvp_fake(scores, tangent, gap_open, gap_ext, temperature, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_sw_affine_param_jacobian")
def soft_sw_affine_param_jacobian_fake(
    scores, param_type, gap_open, gap_ext, temperature, lengths
):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_sw_affine_backward_full")
def soft_sw_affine_backward_full_fake(
    scores, grad_alignment, gap_open, gap_ext, temperature, lengths
):
    """Returns (grad_scores, grad_open, grad_ext, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


# =============================================================================
# NEW API: Smith-Waterman (Regular - Linear Gap)
# =============================================================================


@register_fake("d2p::sw_forward")
def sw_forward_fake(scores, gap, temp, lengths):
    """sw_forward returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::sw_forward_t")
def sw_forward_t_fake(scores, gap, temp, lengths):
    """sw_forward_t (tensor params) returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::sw_value_grad_params")
def sw_value_grad_params_fake(scores, gap, temp, lengths):
    """Returns (grad_gap, grad_temp) per batch."""
    B = scores.size(0)
    return (scores.new_empty([B]), scores.new_empty([B]))


@register_fake("d2p::sw_marginals_backward")
def sw_marginals_backward_fake(scores, grad_marginals, gap, temp, lengths):
    """Returns (grad_scores, grad_gap, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@register_fake("d2p::sw_marginals_hvp")
def sw_marginals_hvp_fake(scores, v, gap, temp, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::sw_marginals_grad_gap")
def sw_marginals_grad_gap_fake(scores, gap, temp, lengths):
    """Returns d(marginals)/d(gap) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::sw_marginals_grad_temp")
def sw_marginals_grad_temp_fake(scores, gap, temp, lengths):
    """Returns d(marginals)/d(temperature) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


# =============================================================================
# NEW API: Smith-Waterman (Affine Gap)
# =============================================================================


@register_fake("d2p::sw_affine_forward")
def sw_affine_forward_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::sw_affine_forward_t")
def sw_affine_forward_t_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns [value, marginals]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::sw_affine_value_grad_params")
def sw_affine_value_grad_params_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns (grad_gap_open, grad_gap_ext, grad_temp) per batch."""
    B = scores.size(0)
    return (scores.new_empty([B]), scores.new_empty([B]), scores.new_empty([B]))


@register_fake("d2p::sw_affine_marginals_backward")
def sw_affine_marginals_backward_fake(
    scores, grad_marginals, gap_open, gap_ext, temp, lengths
):
    """Returns (grad_scores, grad_gap_open, grad_gap_ext, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@register_fake("d2p::sw_affine_marginals_hvp")
def sw_affine_marginals_hvp_fake(scores, v, gap_open, gap_ext, temp, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::sw_affine_marginals_grad_gap_open")
def sw_affine_marginals_grad_gap_open_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(gap_open) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::sw_affine_marginals_grad_gap_ext")
def sw_affine_marginals_grad_gap_ext_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(gap_ext) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::sw_affine_marginals_grad_temp")
def sw_affine_marginals_grad_temp_fake(scores, gap_open, gap_ext, temp, lengths):
    """Returns d(marginals)/d(temperature) [B, L1, L2]."""
    return scores.new_empty(scores.shape)


# =============================================================================
# Needleman-Wunsch (Linear Gap)
# =============================================================================


@register_fake("d2p::soft_nw_float")
def soft_nw_float_fake(scores, gap, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_nw")
def soft_nw_fake(scores, gap, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_nw_with_grads")
def soft_nw_with_grads_fake(scores, gap, temperature, lengths):
    """Returns (score, alignment, grad_gap, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_nw_hvp")
def soft_nw_hvp_fake(scores, tangent, gap, temperature, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_nw_param_jacobian")
def soft_nw_param_jacobian_fake(scores, param_type, gap, temperature, lengths):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_nw_backward_full")
def soft_nw_backward_full_fake(scores, grad_alignment, gap, temperature, lengths):
    """Returns (grad_scores, grad_gap, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


# =============================================================================
# Needleman-Wunsch (Affine Gap)
# =============================================================================


@register_fake("d2p::soft_nw_affine_float")
def soft_nw_affine_float_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_nw_affine")
def soft_nw_affine_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_nw_affine_with_grads")
def soft_nw_affine_with_grads_fake(scores, gap_open, gap_ext, temperature, lengths):
    """Returns (score, alignment, grad_open, grad_ext, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_nw_affine_hvp")
def soft_nw_affine_hvp_fake(scores, tangent, gap_open, gap_ext, temperature, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_nw_affine_param_jacobian")
def soft_nw_affine_param_jacobian_fake(
    scores, param_type, gap_open, gap_ext, temperature, lengths
):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_nw_affine_backward_full")
def soft_nw_affine_backward_full_fake(
    scores, grad_alignment, gap_open, gap_ext, temperature, lengths
):
    """Returns (grad_scores, grad_open, grad_ext, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


# =============================================================================
# Dynamic Time Warping (DTW)
# =============================================================================


@register_fake("d2p::soft_dtw_float")
def soft_dtw_float_fake(costs, temperature, lengths, bandwidth):
    """Returns [cost, alignment]."""
    B, L1, L2 = costs.shape
    return [costs.new_empty([B]), costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_dtw")
def soft_dtw_fake(costs, temperature, lengths, bandwidth):
    """Returns [cost, alignment]."""
    B, L1, L2 = costs.shape
    return [costs.new_empty([B]), costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_dtw_with_grads")
def soft_dtw_with_grads_fake(costs, temperature, lengths, bandwidth):
    """Returns (cost, alignment, grad_temp)."""
    B, L1, L2 = costs.shape
    return (
        costs.new_empty([B]),
        costs.new_empty([B, L1, L2]),
        costs.new_empty([B]),
    )


@register_fake("d2p::soft_dtw_hvp")
def soft_dtw_hvp_fake(costs, tangent, temperature, lengths, bandwidth):
    """Returns HVP with same shape as costs."""
    return costs.new_empty(costs.shape)


@register_fake("d2p::soft_dtw_param_jacobian")
def soft_dtw_param_jacobian_fake(costs, temperature, lengths, bandwidth):
    """Returns parameter Jacobian [B, L1, L2]."""
    return costs.new_empty(costs.shape)


@register_fake("d2p::soft_dtw_backward_full")
def soft_dtw_backward_full_fake(costs, grad_alignment, temperature, lengths, bandwidth):
    """Returns (grad_costs, grad_temp)."""
    B = costs.size(0)
    return (
        costs.new_empty(costs.shape),
        costs.new_empty([B]),
    )


# =============================================================================
# CKY Parsing
# =============================================================================


@register_fake("d2p::soft_cky_float")
def soft_cky_float_fake(merge_scores, leaf_scores, temperature):
    """Returns [score, marginals]."""
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return [
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
    ]


@register_fake("d2p::soft_cky")
def soft_cky_fake(merge_scores, leaf_scores, temperature):
    """Returns [score, marginals]."""
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return [
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
    ]


@register_fake("d2p::soft_cky_with_grads")
def soft_cky_with_grads_fake(merge_scores, leaf_scores, temperature):
    """Returns (score, merge_marginals, leaf_marginals, grad_temp)."""
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
        leaf_scores.new_empty([B, N]),
        merge_scores.new_empty([B]),
    )


@register_fake("d2p::soft_cky_hvp")
def soft_cky_hvp_fake(merge_scores, leaf_scores, v_merge, v_leaf, temperature):
    """Returns HVP with same shape as merge_scores."""
    return merge_scores.new_empty(merge_scores.shape)


@register_fake("d2p::soft_cky_param_jacobian")
def soft_cky_param_jacobian_fake(merge_scores, leaf_scores, temperature):
    """Returns parameter Jacobian [B, N, N, N]."""
    return merge_scores.new_empty(merge_scores.shape)


@register_fake("d2p::soft_cky_backward_full")
def soft_cky_backward_full_fake(
    merge_scores, leaf_scores, grad_posteriors, temperature
):
    """Returns (grad_merge, grad_leaf, grad_temp)."""
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B, N, N, N]),
        leaf_scores.new_empty([B, N]),
        merge_scores.new_empty([B]),
    )


# =============================================================================
# Monotonic Alignment Search (MAS)
# =============================================================================


@register_fake("d2p::soft_mas_float")
def soft_mas_float_fake(scores, temperature, lengths):
    """Returns partition function [B]."""
    B = scores.size(0)
    return scores.new_empty([B])


@register_fake("d2p::soft_mas")
def soft_mas_fake(scores, temperature, lengths):
    """Returns partition function [B]."""
    B = scores.size(0)
    return scores.new_empty([B])


@register_fake("d2p::soft_mas_with_grads")
def soft_mas_with_grads_fake(scores, temperature, lengths):
    """Returns (score, alignment)."""
    B, T, S = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, T, S]),
    )


@register_fake("d2p::soft_mas_hvp")
def soft_mas_hvp_fake(scores, V, temperature, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_mas_param_jacobian")
def soft_mas_param_jacobian_fake(scores, temperature, lengths):
    """Returns parameter Jacobian [B, T, S]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_mas_backward_full")
def soft_mas_backward_full_fake(scores, temperature, lengths):
    """Returns (alignment, grad_scores, grad_temp)."""
    B, T, S = scores.shape
    return (
        scores.new_empty([B, T, S]),
        scores.new_empty([B, T, S]),
        scores.new_empty([B]),
    )


# =============================================================================
# Eisner Dependency Parsing
# =============================================================================


@register_fake("d2p::soft_eisner_float")
def soft_eisner_float_fake(arc_scores, temperature, lengths):
    """Returns partition function [B]."""
    B = arc_scores.size(0)
    return arc_scores.new_empty([B])


@register_fake("d2p::soft_eisner")
def soft_eisner_fake(arc_scores, temperature, lengths):
    """Returns [score, marginals]."""
    B, N, _ = arc_scores.shape
    return [
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
    ]


@register_fake("d2p::soft_eisner_with_grads")
def soft_eisner_with_grads_fake(arc_scores, temperature, lengths):
    """Returns (score, marginals)."""
    B, N, _ = arc_scores.shape
    return (
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
    )


@register_fake("d2p::soft_eisner_hvp")
def soft_eisner_hvp_fake(arc_scores, V, temperature, lengths):
    """Returns HVP with same shape as arc_scores."""
    return arc_scores.new_empty(arc_scores.shape)


@register_fake("d2p::soft_eisner_param_jacobian")
def soft_eisner_param_jacobian_fake(arc_scores, temperature, lengths):
    """Returns parameter Jacobian [B, N, N]."""
    return arc_scores.new_empty(arc_scores.shape)


@register_fake("d2p::soft_eisner_backward_full")
def soft_eisner_backward_full_fake(arc_scores, temperature, lengths):
    """Returns (score, marginals, grad_temp)."""
    B, N, _ = arc_scores.shape
    return (
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
        arc_scores.new_empty([B]),
    )


# =============================================================================
# Levenshtein Edit Distance
# =============================================================================


@register_fake("d2p::soft_levenshtein_float")
def soft_levenshtein_float_fake(scores, ins_cost, del_cost, temperature, lengths):
    """Returns [distance, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_levenshtein")
def soft_levenshtein_fake(scores, ins_cost, del_cost, temperature, lengths):
    """Returns [distance, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_levenshtein_with_grads")
def soft_levenshtein_with_grads_fake(
    scores, ins_cost, del_cost, temperature, lengths
):
    """Returns (distance, alignment, grad_ins, grad_del, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_levenshtein_hvp")
def soft_levenshtein_hvp_fake(
    scores, tangent, ins_cost, del_cost, temperature, lengths
):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_levenshtein_param_jacobian")
def soft_levenshtein_param_jacobian_fake(
    scores, param_type, ins_cost, del_cost, temperature, lengths
):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_levenshtein_backward_full")
def soft_levenshtein_backward_full_fake(
    scores, grad_posteriors, ins_cost, del_cost, temperature, lengths
):
    """Returns (grad_scores, grad_ins, grad_del, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


# =============================================================================
# Longest Common Subsequence (LCS)
# =============================================================================


@register_fake("d2p::soft_lcs_float")
def soft_lcs_float_fake(scores, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_lcs")
def soft_lcs_fake(scores, temperature, lengths):
    """Returns [score, alignment]."""
    B, L1, L2 = scores.shape
    return [scores.new_empty([B]), scores.new_empty([B, L1, L2])]


@register_fake("d2p::soft_lcs_with_grads")
def soft_lcs_with_grads_fake(scores, temperature, lengths):
    """Returns (score, alignment, grad_temp)."""
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
    )


@register_fake("d2p::soft_lcs_hvp")
def soft_lcs_hvp_fake(scores, tangent, temperature, lengths):
    """Returns HVP with same shape as scores."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_lcs_param_jacobian")
def soft_lcs_param_jacobian_fake(scores, temperature, lengths):
    """Returns parameter Jacobian [B, L1, L2]."""
    return scores.new_empty(scores.shape)


@register_fake("d2p::soft_lcs_backward_full")
def soft_lcs_backward_full_fake(scores, grad_output, temperature, lengths):
    """Returns (grad_scores, grad_temp)."""
    B = scores.size(0)
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([B]),
    )


# =============================================================================
# Optimal String Alignment (OSA)
# =============================================================================


@register_fake("d2p::soft_osa_float")
def soft_osa_float_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns [distance, alignment]."""
    B, L1, L2 = sub_costs.shape
    return [sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_osa")
def soft_osa_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns [distance, alignment]."""
    B, L1, L2 = sub_costs.shape
    return [sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_osa_with_grads")
def soft_osa_with_grads_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns (distance, alignment, grad_ins, grad_del, grad_trans, grad_temp)."""
    B, L1, L2 = sub_costs.shape
    return (
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B, L1, L2]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


@register_fake("d2p::soft_osa_hvp")
def soft_osa_hvp_fake(
    sub_costs, trans_mask, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns HVP with same shape as sub_costs."""
    return sub_costs.new_empty(sub_costs.shape)


@register_fake("d2p::soft_osa_param_jacobian")
def soft_osa_param_jacobian_fake(
    sub_costs, trans_mask, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns dP/d{param} with same shape as sub_costs."""
    return sub_costs.new_empty(sub_costs.shape)


@register_fake("d2p::soft_osa_backward_full")
def soft_osa_backward_full_fake(
    sub_costs, trans_mask, grad_output, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns (grad_sub, grad_ins, grad_del, grad_trans, grad_temp)."""
    B = sub_costs.size(0)
    return (
        sub_costs.new_empty(sub_costs.shape),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


# =============================================================================
# Damerau-Levenshtein
# =============================================================================


@register_fake("d2p::soft_damerau_float")
def soft_damerau_float_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns [distance, alignment]."""
    B, L1, L2 = sub_costs.shape
    return [sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_damerau")
def soft_damerau_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns [distance, alignment]."""
    B, L1, L2 = sub_costs.shape
    return [sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2])]


@register_fake("d2p::soft_damerau_with_grads")
def soft_damerau_with_grads_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns (distance, alignment, grad_ins, grad_del, grad_trans, grad_temp)."""
    B, L1, L2 = sub_costs.shape
    return (
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B, L1, L2]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


@register_fake("d2p::soft_damerau_hvp")
def soft_damerau_hvp_fake(
    sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns HVP with same shape as sub_costs."""
    return sub_costs.new_empty(sub_costs.shape)


@register_fake("d2p::soft_damerau_param_jacobian")
def soft_damerau_param_jacobian_fake(
    sub_costs, trans_src, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns dP/d{param} with same shape as sub_costs."""
    return sub_costs.new_empty(sub_costs.shape)


@register_fake("d2p::soft_damerau_backward_full")
def soft_damerau_backward_full_fake(
    sub_costs, trans_src, grad_output, ins_cost, del_cost, trans_cost, temperature, lengths
):
    """Returns (grad_sub, grad_ins, grad_del, grad_trans, grad_temp)."""
    B = sub_costs.size(0)
    return (
        sub_costs.new_empty(sub_costs.shape),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


# =============================================================================
# Autocast (AMP) Support
# =============================================================================
#
# Register autocast behavior for all d2p operators. DP algorithms need FP32 for
# numerical stability due to long sequential dependency chains and logsumexp
# operations that accumulate error in reduced precision (FP16/BF16).
#
# When torch.autocast is enabled, inputs are automatically cast to FP32.

try:
    from torch.library import register_autocast

    # -------------------------------------------------------------------------
    # Smith-Waterman (Regular - Linear Gap)
    # -------------------------------------------------------------------------

    # Legacy API
    register_autocast("d2p::soft_sw", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_float", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_backward_full", "cuda", torch.float32)

    # New namespaced API
    register_autocast("d2p::sw_forward", "cuda", torch.float32)
    register_autocast("d2p::sw_forward_t", "cuda", torch.float32)
    register_autocast("d2p::sw_value_grad_params", "cuda", torch.float32)
    register_autocast("d2p::sw_marginals_backward", "cuda", torch.float32)
    register_autocast("d2p::sw_marginals_hvp", "cuda", torch.float32)
    register_autocast("d2p::sw_marginals_grad_gap", "cuda", torch.float32)
    register_autocast("d2p::sw_marginals_grad_temp", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Smith-Waterman (Affine Gap)
    # -------------------------------------------------------------------------

    # Legacy API
    register_autocast("d2p::soft_sw_affine", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_affine_float", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_affine_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_affine_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_affine_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_sw_affine_backward_full", "cuda", torch.float32)

    # New namespaced API
    register_autocast("d2p::sw_affine_forward", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_forward_t", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_value_grad_params", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_marginals_backward", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_marginals_hvp", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_marginals_grad_gap_open", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_marginals_grad_gap_ext", "cuda", torch.float32)
    register_autocast("d2p::sw_affine_marginals_grad_temp", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Needleman-Wunsch (Linear Gap)
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_nw", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_float", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Needleman-Wunsch (Affine Gap)
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_nw_affine", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_affine_float", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_affine_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_affine_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_affine_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_nw_affine_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Dynamic Time Warping (DTW)
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_dtw", "cuda", torch.float32)
    register_autocast("d2p::soft_dtw_float", "cuda", torch.float32)
    register_autocast("d2p::soft_dtw_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_dtw_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_dtw_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_dtw_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # CKY Parsing
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_cky", "cuda", torch.float32)
    register_autocast("d2p::soft_cky_float", "cuda", torch.float32)
    register_autocast("d2p::soft_cky_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_cky_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_cky_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_cky_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Monotonic Alignment Search (MAS)
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_mas", "cuda", torch.float32)
    register_autocast("d2p::soft_mas_float", "cuda", torch.float32)
    register_autocast("d2p::soft_mas_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_mas_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_mas_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_mas_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Eisner Dependency Parsing
    # -------------------------------------------------------------------------
    register_autocast("d2p::soft_eisner", "cuda", torch.float32)
    register_autocast("d2p::soft_eisner_float", "cuda", torch.float32)
    register_autocast("d2p::soft_eisner_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_eisner_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_eisner_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_eisner_backward_full", "cuda", torch.float32)

    # -------------------------------------------------------------------------
    # Edit Distance Operators
    # -------------------------------------------------------------------------

    # Levenshtein
    register_autocast("d2p::soft_levenshtein", "cuda", torch.float32)
    register_autocast("d2p::soft_levenshtein_float", "cuda", torch.float32)
    register_autocast("d2p::soft_levenshtein_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_levenshtein_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_levenshtein_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_levenshtein_backward_full", "cuda", torch.float32)

    # LCS
    register_autocast("d2p::soft_lcs", "cuda", torch.float32)
    register_autocast("d2p::soft_lcs_float", "cuda", torch.float32)
    register_autocast("d2p::soft_lcs_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_lcs_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_lcs_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_lcs_backward_full", "cuda", torch.float32)

    # OSA
    register_autocast("d2p::soft_osa", "cuda", torch.float32)
    register_autocast("d2p::soft_osa_float", "cuda", torch.float32)
    register_autocast("d2p::soft_osa_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_osa_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_osa_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_osa_backward_full", "cuda", torch.float32)

    # Damerau-Levenshtein
    register_autocast("d2p::soft_damerau", "cuda", torch.float32)
    register_autocast("d2p::soft_damerau_float", "cuda", torch.float32)
    register_autocast("d2p::soft_damerau_with_grads", "cuda", torch.float32)
    register_autocast("d2p::soft_damerau_hvp", "cuda", torch.float32)
    register_autocast("d2p::soft_damerau_param_jacobian", "cuda", torch.float32)
    register_autocast("d2p::soft_damerau_backward_full", "cuda", torch.float32)

except ImportError:
    # register_autocast not available in older PyTorch versions
    pass
