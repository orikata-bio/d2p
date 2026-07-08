# SPDX-License-Identifier: Apache-2.0
"""Low-level operator access for d2p.

This module provides direct access to the underlying C++/CUDA operators.
For most users, the high-level API (d2p.soft_sw, d2p.soft_dtw, etc.) is
recommended.

New API (recommended for low-level access):
    from d2p import ops

    # Smith-Waterman (linear gap)
    value, marginals = ops.sw.forward(scores, gap=-1.0, temp=1.0)
    hvp = ops.sw.marginals_hvp(scores, v, gap=-1.0, temp=1.0)

    # Smith-Waterman (affine gap)
    value, marginals = ops.sw_affine.forward(scores, -2.0, -0.5, 1.0)
    hvp = ops.sw_affine.marginals_hvp(scores, v, -2.0, -0.5, 1.0)

Legacy API (for backward compatibility):
    from d2p import ops

    # These functions are re-exported from d2p._ops
    score, alignment = ops.soft_sw_float(scores, -1.0, 1.0, lengths)
"""

# New structured API
from . import sw
from . import sw_affine

# Re-export legacy ops for backward compatibility
from .._ops import (
    # Full d2p namespace for advanced users
    d2p,
    # Smith-Waterman (linear gap) - legacy
    soft_sw,
    soft_sw_float,
    soft_sw_with_grads,
    soft_sw_hvp,
    soft_sw_param_jacobian,
    soft_sw_backward_full,
    # Smith-Waterman (affine gap) - legacy
    soft_sw_affine,
    soft_sw_affine_float,
    soft_sw_affine_with_grads,
    soft_sw_affine_hvp,
    soft_sw_affine_param_jacobian,
    soft_sw_affine_backward_full,
    # Smith-Waterman (linear gap) - new flat names
    sw_forward,
    sw_forward_t,
    sw_value_grad_params,
    sw_marginals_backward,
    sw_marginals_hvp,
    sw_marginals_grad_gap,
    sw_marginals_grad_temp,
    # Smith-Waterman (affine gap) - new flat names
    sw_affine_forward,
    sw_affine_forward_t,
    sw_affine_value_grad_params,
    sw_affine_marginals_backward,
    sw_affine_marginals_hvp,
    sw_affine_marginals_grad_gap_open,
    sw_affine_marginals_grad_gap_ext,
    sw_affine_marginals_grad_temp,
    # Dynamic Time Warping
    soft_dtw,
    soft_dtw_float,
    soft_dtw_with_grads,
    soft_dtw_hvp,
    soft_dtw_param_jacobian,
    soft_dtw_backward_full,
    # CKY Parsing
    soft_cky,
    soft_cky_float,
    soft_cky_with_grads,
    soft_cky_hvp,
    soft_cky_param_jacobian,
    soft_cky_backward_full,
    # Needleman-Wunsch (linear gap)
    soft_nw,
    soft_nw_float,
    soft_nw_with_grads,
    soft_nw_hvp,
    soft_nw_param_jacobian,
    soft_nw_backward_full,
    # Needleman-Wunsch (affine gap)
    soft_nw_affine,
    soft_nw_affine_float,
    soft_nw_affine_with_grads,
    soft_nw_affine_hvp,
    soft_nw_affine_param_jacobian,
    soft_nw_affine_backward_full,
    # Monotonic Alignment Search
    soft_mas,
    soft_mas_float,
    soft_mas_with_grads,
    soft_mas_hvp,
    soft_mas_param_jacobian,
    soft_mas_backward_full,
    # Eisner (Projective Dependency Parsing)
    soft_eisner,
    soft_eisner_float,
    soft_eisner_with_grads,
    soft_eisner_hvp,
    soft_eisner_param_jacobian,
    soft_eisner_backward_full,
    # Levenshtein
    soft_levenshtein,
    soft_levenshtein_float,
    soft_levenshtein_with_grads,
    soft_levenshtein_hvp,
    soft_levenshtein_param_jacobian,
    soft_levenshtein_backward_full,
    # Longest Common Subsequence
    soft_lcs,
    soft_lcs_float,
    soft_lcs_with_grads,
    soft_lcs_hvp,
    soft_lcs_param_jacobian,
    soft_lcs_backward_full,
    # OSA (Optimal String Alignment)
    soft_osa,
    soft_osa_float,
    soft_osa_with_grads,
    soft_osa_hvp,
    soft_osa_param_jacobian,
    soft_osa_backward_full,
    # True Damerau-Levenshtein
    soft_damerau,
    soft_damerau_float,
    soft_damerau_with_grads,
    soft_damerau_hvp,
    soft_damerau_param_jacobian,
    soft_damerau_backward_full,
)

__all__ = [
    # New structured modules
    'sw',
    'sw_affine',
    # Full d2p namespace
    'd2p',
    # Smith-Waterman (linear gap) - legacy
    'soft_sw', 'soft_sw_float', 'soft_sw_with_grads', 'soft_sw_hvp',
    'soft_sw_param_jacobian', 'soft_sw_backward_full',
    # Smith-Waterman (affine gap) - legacy
    'soft_sw_affine', 'soft_sw_affine_float', 'soft_sw_affine_with_grads',
    'soft_sw_affine_hvp', 'soft_sw_affine_param_jacobian', 'soft_sw_affine_backward_full',
    # Smith-Waterman (linear gap) - new flat names
    'sw_forward', 'sw_forward_t', 'sw_value_grad_params', 'sw_marginals_backward',
    'sw_marginals_hvp', 'sw_marginals_grad_gap', 'sw_marginals_grad_temp',
    # Smith-Waterman (affine gap) - new flat names
    'sw_affine_forward', 'sw_affine_forward_t', 'sw_affine_value_grad_params',
    'sw_affine_marginals_backward', 'sw_affine_marginals_hvp',
    'sw_affine_marginals_grad_gap_open', 'sw_affine_marginals_grad_gap_ext',
    'sw_affine_marginals_grad_temp',
    # DTW
    'soft_dtw', 'soft_dtw_float', 'soft_dtw_with_grads', 'soft_dtw_hvp',
    'soft_dtw_param_jacobian', 'soft_dtw_backward_full',
    # CKY
    'soft_cky', 'soft_cky_float', 'soft_cky_with_grads', 'soft_cky_hvp',
    'soft_cky_param_jacobian', 'soft_cky_backward_full',
    # NW (linear)
    'soft_nw', 'soft_nw_float', 'soft_nw_with_grads', 'soft_nw_hvp',
    'soft_nw_param_jacobian', 'soft_nw_backward_full',
    # NW (affine)
    'soft_nw_affine', 'soft_nw_affine_float', 'soft_nw_affine_with_grads',
    'soft_nw_affine_hvp', 'soft_nw_affine_param_jacobian', 'soft_nw_affine_backward_full',
    # MAS
    'soft_mas', 'soft_mas_float', 'soft_mas_with_grads', 'soft_mas_hvp',
    'soft_mas_param_jacobian', 'soft_mas_backward_full',
    # Eisner
    'soft_eisner', 'soft_eisner_float', 'soft_eisner_with_grads', 'soft_eisner_hvp',
    'soft_eisner_param_jacobian', 'soft_eisner_backward_full',
    # Levenshtein
    'soft_levenshtein', 'soft_levenshtein_float', 'soft_levenshtein_with_grads',
    'soft_levenshtein_hvp', 'soft_levenshtein_param_jacobian', 'soft_levenshtein_backward_full',
    # LCS
    'soft_lcs', 'soft_lcs_float', 'soft_lcs_with_grads', 'soft_lcs_hvp',
    'soft_lcs_param_jacobian', 'soft_lcs_backward_full',
    # OSA
    'soft_osa', 'soft_osa_float', 'soft_osa_with_grads', 'soft_osa_hvp',
    'soft_osa_param_jacobian', 'soft_osa_backward_full',
    # Damerau
    'soft_damerau', 'soft_damerau_float', 'soft_damerau_with_grads', 'soft_damerau_hvp',
    'soft_damerau_param_jacobian', 'soft_damerau_backward_full',
]
