# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Dynamic Programming (D2P) - PyTorch operators for soft DP algorithms.

High-Level API (Recommended):
    # Smith-Waterman local alignment
    result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)
    result = d2p.soft_sw_affine(scores, gap_open=-2.0, gap_ext=-0.5)

    # Needleman-Wunsch global alignment
    result = d2p.soft_nw(scores, gap=-1.0, temperature=1.0)

    # Dynamic Time Warping
    result = d2p.soft_dtw(costs, temperature=1.0, bandwidth=0)

    # CKY parsing
    result = d2p.soft_cky(scores, temperature=1.0)

    # Monotonic Alignment Search (TTS/ASR)
    result = d2p.soft_mas(scores, temperature=1.0)

    # Eisner dependency parsing
    result = d2p.soft_eisner(arc_scores, temperature=1.0)

    # Edit distances
    result = d2p.soft_levenshtein(costs, temperature=1.0)
    result = d2p.soft_lcs(match_scores, temperature=1.0)

Module API (nn.Module wrappers):
    sw = d2p.SoftSW(gap=-1.0, temperature=1.0, learnable=True)
    result = sw(scores)

Low-Level API (via d2p.ops):
    from d2p import ops
    score, alignment = ops.soft_sw_float(scores, -1.0, 1.0, lengths)
"""

# Load extension first
from . import _ops

# Register fake tensor implementations for torch.compile support
from . import _compile  # noqa: F401

# High-level API - Smith-Waterman
from .sw import (
    soft_sw,
    soft_sw_affine,
    SWResult,
    SoftSW,
    SoftSWAffine,
)

# High-level API - Dynamic Time Warping
from .dtw import (
    soft_dtw,
    DTWResult,
    SoftDTW,
)

# High-level API - CKY Parsing
from .cky import (
    soft_cky,
    CKYResult,
    SoftCKY,
)

# High-level API - Needleman-Wunsch
from .nw import (
    soft_nw,
    soft_nw_affine,
    NWResult,
    SoftNW,
    SoftNWAffine,
)

# High-level API - Monotonic Alignment Search
from .mas import (
    soft_mas,
    MASResult,
    SoftMAS,
)

# High-level API - Eisner Dependency Parsing
from .eisner import (
    soft_eisner,
    EisnerResult,
    SoftEisner,
)

# High-level API - Edit Distances
from .edit_distance import (
    soft_levenshtein,
    soft_lcs,
    soft_osa,
    soft_damerau,
    LevenshteinResult,
    LCSResult,
    OSAResult,
    DamerauResult,
    SoftLevenshtein,
    SoftLCS,
    SoftOSA,
    SoftDamerau,
)

# Low-level ops access (new structured module)
from . import ops

# =============================================================================
# Backward Compatibility: Direct operator access (deprecated, use high-level API)
# =============================================================================

# CKY
soft_cky_float = _ops.soft_cky_float
soft_cky_with_grads = _ops.soft_cky_with_grads
soft_cky_hvp = _ops.soft_cky_hvp
soft_cky_param_jacobian = _ops.soft_cky_param_jacobian
soft_cky_backward_full = _ops.soft_cky_backward_full

# Smith-Waterman (linear gap)
soft_sw_float = _ops.soft_sw_float
soft_sw_with_grads = _ops.soft_sw_with_grads
soft_sw_hvp = _ops.soft_sw_hvp
soft_sw_param_jacobian = _ops.soft_sw_param_jacobian
soft_sw_backward_full = _ops.soft_sw_backward_full

# Smith-Waterman (affine gap)
soft_sw_affine_float = _ops.soft_sw_affine_float
soft_sw_affine_with_grads = _ops.soft_sw_affine_with_grads
soft_sw_affine_hvp = _ops.soft_sw_affine_hvp
soft_sw_affine_param_jacobian = _ops.soft_sw_affine_param_jacobian
soft_sw_affine_backward_full = _ops.soft_sw_affine_backward_full

# DTW
soft_dtw_float = _ops.soft_dtw_float
soft_dtw_with_grads = _ops.soft_dtw_with_grads
soft_dtw_hvp = _ops.soft_dtw_hvp
soft_dtw_param_jacobian = _ops.soft_dtw_param_jacobian
soft_dtw_backward_full = _ops.soft_dtw_backward_full

# Needleman-Wunsch (linear gap)
soft_nw_float = _ops.soft_nw_float
soft_nw_with_grads = _ops.soft_nw_with_grads
soft_nw_hvp = _ops.soft_nw_hvp
soft_nw_param_jacobian = _ops.soft_nw_param_jacobian
soft_nw_backward_full = _ops.soft_nw_backward_full

# Needleman-Wunsch (affine gap)
soft_nw_affine_float = _ops.soft_nw_affine_float
soft_nw_affine_with_grads = _ops.soft_nw_affine_with_grads
soft_nw_affine_hvp = _ops.soft_nw_affine_hvp
soft_nw_affine_param_jacobian = _ops.soft_nw_affine_param_jacobian
soft_nw_affine_backward_full = _ops.soft_nw_affine_backward_full

# MAS
soft_mas_float = _ops.soft_mas_float
soft_mas_with_grads = _ops.soft_mas_with_grads
soft_mas_hvp = _ops.soft_mas_hvp
soft_mas_param_jacobian = _ops.soft_mas_param_jacobian
soft_mas_backward_full = _ops.soft_mas_backward_full

# Eisner
soft_eisner_float = _ops.soft_eisner_float
soft_eisner_with_grads = _ops.soft_eisner_with_grads
soft_eisner_hvp = _ops.soft_eisner_hvp
soft_eisner_param_jacobian = _ops.soft_eisner_param_jacobian
soft_eisner_backward_full = _ops.soft_eisner_backward_full

# Levenshtein
soft_levenshtein_float = _ops.soft_levenshtein_float
soft_levenshtein_with_grads = _ops.soft_levenshtein_with_grads
soft_levenshtein_hvp = _ops.soft_levenshtein_hvp
soft_levenshtein_param_jacobian = _ops.soft_levenshtein_param_jacobian
soft_levenshtein_backward_full = _ops.soft_levenshtein_backward_full

# LCS
soft_lcs_float = _ops.soft_lcs_float
soft_lcs_with_grads = _ops.soft_lcs_with_grads
soft_lcs_hvp = _ops.soft_lcs_hvp
soft_lcs_param_jacobian = _ops.soft_lcs_param_jacobian
soft_lcs_backward_full = _ops.soft_lcs_backward_full

# OSA
soft_osa_float = _ops.soft_osa_float
soft_osa_with_grads = _ops.soft_osa_with_grads
soft_osa_hvp = _ops.soft_osa_hvp
soft_osa_param_jacobian = _ops.soft_osa_param_jacobian
soft_osa_backward_full = _ops.soft_osa_backward_full

# Damerau
soft_damerau_float = _ops.soft_damerau_float
soft_damerau_with_grads = _ops.soft_damerau_with_grads
soft_damerau_hvp = _ops.soft_damerau_hvp
soft_damerau_param_jacobian = _ops.soft_damerau_param_jacobian
soft_damerau_backward_full = _ops.soft_damerau_backward_full

__all__ = [
    # High-level API
    'soft_sw', 'soft_sw_affine', 'SWResult', 'SoftSW', 'SoftSWAffine',
    'soft_dtw', 'DTWResult', 'SoftDTW',
    'soft_cky', 'CKYResult', 'SoftCKY',
    'soft_nw', 'soft_nw_affine', 'NWResult', 'SoftNW', 'SoftNWAffine',
    'soft_mas', 'MASResult', 'SoftMAS',
    'soft_eisner', 'EisnerResult', 'SoftEisner',
    'soft_levenshtein', 'soft_lcs', 'soft_osa', 'soft_damerau',
    'LevenshteinResult', 'LCSResult', 'OSAResult', 'DamerauResult',
    'SoftLevenshtein', 'SoftLCS', 'SoftOSA', 'SoftDamerau',
    # Low-level ops access
    'ops',
    # Backward compatibility
    'soft_cky_float', 'soft_cky_with_grads', 'soft_cky_hvp', 'soft_cky_param_jacobian', 'soft_cky_backward_full',
    'soft_sw_float', 'soft_sw_with_grads', 'soft_sw_hvp', 'soft_sw_param_jacobian', 'soft_sw_backward_full',
    'soft_sw_affine_float', 'soft_sw_affine_with_grads', 'soft_sw_affine_hvp', 'soft_sw_affine_param_jacobian', 'soft_sw_affine_backward_full',
    'soft_dtw_float', 'soft_dtw_with_grads', 'soft_dtw_hvp', 'soft_dtw_param_jacobian', 'soft_dtw_backward_full',
    'soft_nw_float', 'soft_nw_with_grads', 'soft_nw_hvp', 'soft_nw_param_jacobian', 'soft_nw_backward_full',
    'soft_nw_affine_float', 'soft_nw_affine_with_grads', 'soft_nw_affine_hvp', 'soft_nw_affine_param_jacobian', 'soft_nw_affine_backward_full',
    'soft_mas_float', 'soft_mas_with_grads', 'soft_mas_hvp', 'soft_mas_param_jacobian', 'soft_mas_backward_full',
    'soft_eisner_float', 'soft_eisner_with_grads', 'soft_eisner_hvp', 'soft_eisner_param_jacobian', 'soft_eisner_backward_full',
    'soft_levenshtein_float', 'soft_levenshtein_with_grads', 'soft_levenshtein_hvp', 'soft_levenshtein_param_jacobian', 'soft_levenshtein_backward_full',
    'soft_lcs_float', 'soft_lcs_with_grads', 'soft_lcs_hvp', 'soft_lcs_param_jacobian', 'soft_lcs_backward_full',
    'soft_osa_float', 'soft_osa_with_grads', 'soft_osa_hvp', 'soft_osa_param_jacobian', 'soft_osa_backward_full',
    'soft_damerau_float', 'soft_damerau_with_grads', 'soft_damerau_hvp', 'soft_damerau_param_jacobian', 'soft_damerau_backward_full',
]
