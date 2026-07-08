# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Eisner parsing module for projective dependency parsing.

Eisner's algorithm finds the best projective dependency tree given
arc scores. A projective tree has no crossing arcs.

API Overview:

High-Level API:
    result = soft_eisner(arc_scores, temperature=1.0)
    # result.score [B], result.marginals [B, N, N]

Module API:
    eisner = SoftEisner(temperature=1.0, learnable=True)
    result = eisner(arc_scores)

Low-Level API (via d2p.ops):
    score, marginals = ops.soft_eisner_float(arc_scores, temperature, lengths)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


class EisnerResult(NamedTuple):
    """Result of soft Eisner parsing.

    Attributes:
        score: Log partition function [B].
        marginals: Arc marginal probabilities [B, N, N].
            marginals[b, i, j] = probability of arc from i to j.
    """
    score: torch.Tensor
    marginals: torch.Tensor


def _validate_temperature(temperature: Union[float, torch.Tensor]) -> None:
    if isinstance(temperature, (int, float)) and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    elif isinstance(temperature, torch.Tensor):
        if use_pt2_ops(temperature):
            return
        if not temperature.requires_grad and temperature.numel() == 1:
            if temperature.item() <= 0:
                raise ValueError(f"temperature must be positive, got {temperature.item()}")


def _make_lengths(arc_scores: torch.Tensor) -> torch.Tensor:
    B, N, _ = arc_scores.shape
    return torch.full((B,), N, dtype=torch.int32, device=arc_scores.device)


def soft_eisner(
    arc_scores: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> EisnerResult:
    """Compute soft Eisner projective dependency parsing.

    Finds the soft distribution over projective dependency trees given
    arc scores. Position 0 is typically the ROOT token.

    Args:
        arc_scores: Arc scores [B, N, N]. arc_scores[b, i, j] is the score
            for an arc from head i to dependent j.
        temperature: Temperature for softmax (default: 1.0).
        lengths: Optional [B] tensor of actual sentence lengths.

    Returns:
        EisnerResult with:
            - score: Log partition function [B]
            - marginals: Arc marginal probabilities [B, N, N]

    Example:
        >>> # Dependency parsing for 10-word sentence + ROOT
        >>> arc_scores = torch.randn(2, 11, 11, device="cuda", requires_grad=True)
        >>> result = soft_eisner(arc_scores, temperature=1.0)
        >>> loss = result.score.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths(arc_scores)

    # Convert temperature to tensor if needed
    if isinstance(temperature, (int, float)):
        temp_t = torch.tensor([temperature], device=arc_scores.device)
    else:
        temp_t = temperature

    # soft_eisner with tensor temp returns [score, marginals]
    result = _ops.soft_eisner(arc_scores, temp_t, lengths)

    return EisnerResult(result[0], result[1])


class SoftEisner(nn.Module):
    """Differentiable Eisner parsing module.

    Args:
        temperature: Temperature for softmax (default: 1.0).
        learnable: Whether temperature is learnable (default: False).

    Example:
        >>> eisner = SoftEisner(temperature=1.0, learnable=True)
        >>> arc_scores = torch.randn(2, 11, 11, device="cuda")
        >>> result = eisner(arc_scores)
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
        arc_scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> EisnerResult:
        if lengths is None:
            lengths = _make_lengths(arc_scores)

        result = _ops.soft_eisner(arc_scores, self.temperature, lengths)
        return EisnerResult(result[0], result[1])


# Low-Level API
soft_eisner_forward = _ops.soft_eisner_float
soft_eisner_with_grads = _ops.soft_eisner_with_grads
soft_eisner_hvp = _ops.soft_eisner_hvp
soft_eisner_param_jacobian = _ops.soft_eisner_param_jacobian
soft_eisner_backward_full = _ops.soft_eisner_backward_full
