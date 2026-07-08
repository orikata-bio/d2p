# SPDX-License-Identifier: Apache-2.0
"""Low-level Smith-Waterman operations (linear gap penalty).

This module provides direct access to the underlying C++/CUDA operators
for Smith-Waterman with linear gap penalty. For most users, the high-level
API (d2p.soft_sw) is recommended.

Usage:
    from d2p import ops

    # Forward pass with scalar parameters
    value, marginals = ops.sw.forward(scores, gap=-1.0, temp=1.0)

    # Forward pass with tensor parameters (for learnable params)
    value, marginals = ops.sw.forward_t(scores, gap_tensor, temp_tensor)

    # Hessian-vector product
    hvp = ops.sw.marginals_hvp(scores, v, gap=-1.0, temp=1.0)
"""

from torch import Tensor
from typing import Optional, Tuple
from .. import _ops


def forward(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with scalar parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty (typically negative)
        temp: Temperature for soft-max
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_forward(scores, gap, temp, lengths)


def forward_t(
    scores: Tensor,
    gap: Tensor,
    temp: Tensor,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass with tensor parameters (for learnable params).

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty tensor [1]
        temp: Temperature tensor [1]
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        value: Log partition function [B]
        marginals: Alignment marginals [B, L1, L2]
    """
    return _ops.sw_forward_t(scores, gap, temp, lengths)


def value_grad_params(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Gradients of value w.r.t. parameters.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_gap: Gradient of value w.r.t. gap [B]
        grad_temp: Gradient of value w.r.t. temperature [B]
    """
    return _ops.sw_value_grad_params(scores, gap, temp, lengths)


def marginals_backward(
    scores: Tensor,
    grad_marginals: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Full backward through marginals.

    Computes gradients of loss (through marginals) w.r.t. all inputs.

    Args:
        scores: Similarity matrix [B, L1, L2]
        grad_marginals: Gradient w.r.t. marginals [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        grad_scores: Gradient w.r.t. scores [B, L1, L2]
        grad_gap: Gradient w.r.t. gap [B]
        grad_temp: Gradient w.r.t. temperature [B]
    """
    return _ops.sw_marginals_backward(scores, grad_marginals, gap, temp, lengths)


def marginals_hvp(
    scores: Tensor,
    v: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Hessian-vector product: H @ v where H = d^2value/dscores^2.

    This efficiently computes the action of the Hessian on a vector
    without forming the full O(L^4) Hessian matrix.

    Args:
        scores: Similarity matrix [B, L1, L2]
        v: Vector to multiply with Hessian [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        hvp: Hessian-vector product [B, L1, L2]
    """
    return _ops.sw_marginals_hvp(scores, v, gap, temp, lengths)


def marginals_grad_gap(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. gap (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dgap.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dgap [B, L1, L2]
    """
    return _ops.sw_marginals_grad_gap(scores, gap, temp, lengths)


def marginals_grad_temp(
    scores: Tensor,
    gap: float,
    temp: float,
    lengths: Optional[Tensor] = None,
) -> Tensor:
    """Gradient of marginals w.r.t. temperature (full Jacobian).

    Returns the full [B, L1, L2] tensor of dmarginals/dtemp.

    Args:
        scores: Similarity matrix [B, L1, L2]
        gap: Gap penalty
        temp: Temperature
        lengths: Optional [B, 2] tensor of actual sequence lengths

    Returns:
        jacobian: dmarginals/dtemp [B, L1, L2]
    """
    return _ops.sw_marginals_grad_temp(scores, gap, temp, lengths)
