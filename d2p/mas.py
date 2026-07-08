# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Monotonic Alignment Search (MAS) module.

MAS is used in TTS/ASR systems to find monotonic alignments between
text and audio. Enforces that the alignment can only move forward
(no backtracking).

API Overview:

High-Level API:
    result = soft_mas(scores, temperature=1.0)
    # result.score [B], result.alignment [B, T, S]

Module API:
    mas = SoftMAS(temperature=1.0, learnable=True)
    result = mas(scores)

Low-Level API (via d2p.ops):
    score, alignment = ops.soft_mas_float(scores, temperature, lengths)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from . import _pt2_ops
from ._pt2_utils import use_pt2_ops


class MASResult(NamedTuple):
    """Result of soft Monotonic Alignment Search.

    Attributes:
        score: Log partition function [B].
        alignment: Soft monotonic alignment [B, T, S], with T >= S. T is the
            longer sequence (e.g. audio frames), S the shorter (e.g. text
            tokens); each of the T rows maps to one of the S columns.
    """
    score: torch.Tensor
    alignment: torch.Tensor


def _validate_temperature(temperature: Union[float, torch.Tensor]) -> None:
    if isinstance(temperature, (int, float)) and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    elif isinstance(temperature, torch.Tensor):
        if use_pt2_ops(temperature):
            return
        if not temperature.requires_grad and temperature.numel() == 1:
            if temperature.item() <= 0:
                raise ValueError(f"temperature must be positive, got {temperature.item()}")


def _make_lengths(scores: torch.Tensor) -> torch.Tensor:
    B, T, S = scores.shape
    return torch.tensor([[T, S]] * B, dtype=torch.int32, device=scores.device)


def _validate_mas_shape(scores: torch.Tensor, lengths: Optional[torch.Tensor]) -> None:
    """MAS needs T >= S (at least one row per column); otherwise no monotonic
    alignment exists and the DP has no feasible path."""
    T, S = scores.shape[-2], scores.shape[-1]
    if lengths is None:
        if T < S:
            raise ValueError(
                f"soft_mas requires scores.shape[-2] >= scores.shape[-1] (T >= S), "
                f"i.e. at least one row per column for a valid monotonic alignment; "
                f"got scores shape [..., {T}, {S}] with T={T} < S={S}."
            )
    elif lengths.dim() == 2 and lengths.shape[-1] == 2:
        # Feasibility check for IN-RANGE lengths only; out-of-range values are left
        # to the operator's own bounds checks (which also guard memory safety).
        tl, sl = lengths[:, 0], lengths[:, 1]
        in_range = (tl >= 1) & (tl <= scores.shape[-2]) & (sl >= 1) & (sl <= scores.shape[-1])
        if bool((in_range & (tl < sl)).any()):
            raise ValueError(
                "soft_mas requires lengths[:, 0] >= lengths[:, 1] (T_len >= S_len) for "
                "every batch element; at least one in-range element has T_len < S_len."
            )


def soft_mas(
    scores: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> MASResult:
    """Compute soft Monotonic Alignment Search.

    Finds a monotonic alignment between two sequences (e.g., text and audio
    in TTS). The alignment can only move forward, never backward.

    Args:
        scores: Similarity matrix [B, T, S] with T >= S. T is the longer
            sequence (e.g. audio frames), S the shorter (e.g. text tokens).
            Raises ValueError if T < S (no valid monotonic alignment exists).
        temperature: Temperature for softmax (default: 1.0).
        lengths: Optional [B, 2] tensor of [T_len, S_len] per element,
            each satisfying T_len >= S_len.

    Returns:
        MASResult with:
            - score: Log partition function [B]
            - alignment: Soft monotonic alignment [B, T, S]

    Example:
        >>> # TTS alignment: 50 mel frames -> 10 text tokens (T=50 >= S=10)
        >>> scores = torch.randn(2, 50, 10, device="cuda", requires_grad=True)
        >>> result = soft_mas(scores, temperature=1.0)
        >>> loss = result.score.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)
    _validate_mas_shape(scores, lengths)

    use_custom = use_pt2_ops(scores, temperature) or isinstance(temperature, torch.Tensor)
    if use_custom:
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor([temperature], device=scores.device)
        result = _pt2_ops.soft_mas(scores, temperature, lengths)
        return MASResult(result[0], result[1])

    temp_val = temperature.item() if isinstance(temperature, torch.Tensor) else temperature
    partition, alignment = _ops.soft_mas_float(scores, temp_val, lengths)

    return MASResult(partition, alignment)


class SoftMAS(nn.Module):
    """Differentiable Monotonic Alignment Search module.

    Args:
        temperature: Temperature for softmax (default: 1.0).
        learnable: Whether temperature is learnable (default: False).

    Example:
        >>> mas = SoftMAS(temperature=1.0, learnable=True)
        >>> scores = torch.randn(2, 50, 10, device="cuda")  # T=50 >= S=10
        >>> result = mas(scores)
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
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> MASResult:
        _validate_mas_shape(scores, lengths)
        use_custom = use_pt2_ops(scores, self.temperature) or isinstance(
            self.temperature, torch.Tensor
        )
        if use_custom:
            result = _pt2_ops.soft_mas(scores, self.temperature, lengths)
            return MASResult(result[0], result[1])

        temp_val = self.temperature.item()
        partition, alignment = _ops.soft_mas_float(scores, temp_val, lengths)
        return MASResult(partition, alignment)


# Low-Level API
soft_mas_forward = _ops.soft_mas_float
soft_mas_with_grads = _ops.soft_mas_with_grads
soft_mas_hvp = _ops.soft_mas_hvp
soft_mas_backward_full = _ops.soft_mas_backward_full
