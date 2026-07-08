# SPDX-License-Identifier: Apache-2.0
"""Low-level Smith-Waterman operations (affine gap penalty).

This module provides direct access to the underlying C++/CUDA operators
for Smith-Waterman with affine gap penalty. For most users, the high-level
API (d2p.soft_sw_affine) is recommended.

Affine gap uses the standard 3-state model where the gap penalty is:
gap_open + k * gap_ext for a gap of length k.

Usage:
    from d2p import ops

    # Forward pass with scalar parameters
    value, marginals = ops.sw_affine.forward(
        scores, gap_open=-2.0, gap_ext=-0.5, temp=1.0
    )

    # Forward pass with tensor parameters (for learnable params)
    value, marginals = ops.sw_affine.forward_t(
        scores, gap_open_t, gap_ext_t, temp_t
    )

    # Hessian-vector product
    hvp = ops.sw_affine.marginals_hvp(scores, v, gap_open=-2.0, gap_ext=-0.5, temp=1.0)
"""

from torch import Tensor
from typing import Optional, Tuple
from .. import _ops


def forward(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with scalar parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty (typically negative)
        gap_ext: Gap extension penalty (typically negative)
        temp: Temperature for soft-max
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_affine_forward(scores, gap_open, gap_ext, temp, lengths)


def forward_t(
    scores: Tensor,
    gap_open: Tensor,
    gap_ext: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with tensor parameters (for learnable params).

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty tensor [1]
        gap_ext: Gap extension penalty tensor [1]
        temp: Temperature tensor [1]
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_affine_forward_t(scores, gap_open, gap_ext, temp, lengths)


def value_grad_params(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Gradients of value w.r.t. parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_gap_open: Gradient of value w.r.t. gap_open [B]
        grad_gap_ext: Gradient of value w.r.t. gap_ext [B]
        grad_temp: Gradient of value w.r.t. temperature [B]
    """
    return _ops.sw_affine_value_grad_params(scores, gap_open, gap_ext, temp, lengths)


def marginals_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Full backward through marginals.

    Computes gradients of loss (through marginals) w.r.t. all inputs.

    Args:
        scores: Similarity matrix [B, L1, L2]
        grad_marginals: Gradient w.r.t. marginals [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_scores: Gradient w.r.t. scores [B, L1, L2]
        grad_gap_open: Gradient w.r.t. gap_open [B]
        grad_gap_ext: Gradient w.r.t. gap_ext [B]
        grad_temp: Gradient w.r.t. temperature [B]
    """
    return _ops.sw_affine_marginals_backward(
        scores, grad_marginals, gap_open, gap_ext, temp, lengths
    )


def marginals_hvp(
    scores: Tensor,
    v: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Hessian-vector product: H @ v where H = d^2value/dscores^2.

    This efficiently computes the action of the Hessian on a vector
    without forming the full O(L^4) Hessian matrix.

    Args:
        scores: Similarity matrix [B, L1, L2]
        v: Vector to multiply with Hessian [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        hvp: Hessian-vector product [B, L1, L2]
    """
    return _ops.sw_affine_marginals_hvp(scores, v, gap_open, gap_ext, temp, lengths)


def marginals_grad_gap_open(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap_open (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap_open.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap_open [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_gap_open(
        scores, gap_open, gap_ext, temp, lengths
    )


def marginals_grad_gap_ext(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap_ext (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap_ext.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap_ext [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_gap_ext(
        scores, gap_open, gap_ext, temp, lengths
    )


def marginals_grad_temp(
    scores: Tensor,
    gap_open: float,
    gap_ext: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. temperature (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dtemp.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap_open: Gap opening penalty
        gap_ext: Gap extension penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dtemp [B, L1, L2]
    """
    return _ops.sw_affine_marginals_grad_temp(
        scores, gap_open, gap_ext, temp, lengths
    )
