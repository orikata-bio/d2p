# SPDX-License-Identifier: Apache-2.0
"""
Differentiable CKY parsing module.

CKY (Cocke-Kasami-Younger) is a chart parsing algorithm for context-free grammars.
This implementation computes soft inside scores for binary constituency parsing.

API Overview:

High-Level API:
    result = soft_cky(merge_scores, leaf_scores, temperature=1.0)
    # result.score [B], result.marginals [B, N, N]

Module API:
    cky = SoftCKY(temperature=1.0, learnable=True)
    result = cky(merge_scores, leaf_scores)

Low-Level API (via d2p.ops):
    score, marginals = ops.soft_cky_float(merge_scores, leaf_scores, temperature)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


class CKYResult(NamedTuple):
    """Result of soft CKY parsing.

    Attributes:
        score: Log partition function (soft parse score) [B].
        marginals: Span marginal probabilities [B, N, N].
            marginals[b, i, j] = probability that span (i, j) is in the parse.
    """
    score: torch.Tensor
    marginals: torch.Tensor


def _validate_temperature(
    temperature: Union[float, torch.Tensor],
    merge_scores: Optional[torch.Tensor] = None,
) -> None:
    if isinstance(temperature, (int, float)) and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    elif isinstance(temperature, torch.Tensor):
        if use_pt2_ops(temperature):
            return
        if temperature.numel() == 1:
            if not temperature.requires_grad and temperature.item() <= 0:
                raise ValueError(f"temperature must be positive, got {temperature.item()}")
            return
        if merge_scores is None:
            raise ValueError("merge_scores are required to validate local temperature shape")
        B, N = merge_scores.shape[:2]
        if temperature.dim() != 3 or temperature.shape != (B, N, N):
            raise ValueError(
                f"temperature must be scalar or [B, N, N]; got {tuple(temperature.shape)}"
            )
        if not temperature.requires_grad and torch.any(temperature <= 0):
            raise ValueError("temperature entries must be positive")


def soft_cky(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
) -> CKYResult:
    """Compute soft CKY parsing.

    Computes the soft inside scores for all spans in a constituency parse.

    Args:
        merge_scores: Span merge scores [B, N, N, N]. merge_scores[b, i, k, j] is the score
            for creating span (i, j) by merging (i, k) and (k+1, j).
        leaf_scores: Leaf (terminal) scores [B, N]. Score for each position as a leaf.
        temperature: Temperature for softmax (default: 1.0).

    Returns:
        CKYResult with:
            - score: Log partition function [B]
            - marginals: Span marginal probabilities [B, N, N]

    Example:
        >>> merge_scores = torch.randn(2, 8, 8, 8, device="cuda", requires_grad=True)
        >>> leaf_scores = torch.randn(2, 8, device="cuda", requires_grad=True)
        >>> result = soft_cky(merge_scores, leaf_scores, temperature=1.0)
        >>> loss = result.score.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature, merge_scores=merge_scores)

    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if temp_is_tensor:
        if temperature.numel() == 1 and temperature.dim() == 0:
            temperature = temperature.reshape(1)
        result = _ops.soft_cky(merge_scores, leaf_scores, temperature)
    else:
        result = _ops.soft_cky_float(merge_scores, leaf_scores, temperature)

    return CKYResult(result[0], result[1])


class SoftCKY(nn.Module):
    """Differentiable CKY parsing module.

    Args:
        temperature: Temperature for softmax (default: 1.0).
        learnable: Whether temperature is learnable (default: False).

    Example:
        >>> cky = SoftCKY(temperature=1.0, learnable=True)
        >>> merge_scores = torch.randn(2, 8, 8, device="cuda")
        >>> leaf_scores = torch.randn(2, 8, device="cuda")
        >>> result = cky(merge_scores, leaf_scores)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        merge_scores: torch.Tensor,
        leaf_scores: torch.Tensor,
    ) -> CKYResult:
        result = _ops.soft_cky(merge_scores, leaf_scores, self.temperature)
        return CKYResult(result[0], result[1])


# Low-Level API
soft_cky_forward = _ops.soft_cky_float
soft_cky_with_grads = _ops.soft_cky_with_grads
soft_cky_hvp = _ops.soft_cky_hvp
soft_cky_backward_full = _ops.soft_cky_backward_full
