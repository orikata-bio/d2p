# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Needleman-Wunsch module for soft global sequence alignment.

Needleman-Wunsch is global alignment: both sequences must be fully aligned
(unlike Smith-Waterman which finds local alignments).

API Overview:

High-Level API:
    result = soft_nw(scores, gap=-1.0, temperature=1.0)
    result = soft_nw_affine(scores, gap_open=-2.0, gap_ext=-0.5, temperature=1.0)
    # result.score [B], result.alignment [B, L1, L2]

Module API:
    nw = SoftNW(gap=-1.0, temperature=1.0, learnable=True)
    result = nw(scores)

Low-Level API (via d2p.ops):
    score, alignment = ops.soft_nw_float(scores, gap, temperature, lengths)
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


class NWResult(NamedTuple):
    """Result of soft Needleman-Wunsch alignment.

    Attributes:
        score: Log partition function [B].
        alignment: Soft alignment matrix [B, L1, L2].
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


def _validate_gap(gap: Union[float, torch.Tensor], name: str = "gap") -> None:
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"{name}={gap} is positive. Gap penalties are typically negative.",
            UserWarning,
            stacklevel=3,
        )


def _make_lengths(scores: torch.Tensor) -> torch.Tensor:
    B, L1, L2 = scores.shape
    return torch.tensor([[L1, L2]] * B, dtype=torch.int32, device=scores.device)


def soft_nw(
    scores: torch.Tensor,
    gap: Union[float, torch.Tensor] = -1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> NWResult:
    """Compute soft Needleman-Wunsch global alignment.

    Unlike Smith-Waterman (local alignment), NW aligns the entire sequences
    from end to end. Useful when you know both sequences should be fully aligned.

    Args:
        scores: Similarity/substitution matrix [B, L1, L2]
        gap: Gap penalty (typically negative, default: -1.0).
        temperature: Temperature for softmax (default: 1.0).
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        NWResult with score [B] and alignment [B, L1, L2].

    Example:
        >>> scores = torch.randn(2, 100, 120, device="cuda", requires_grad=True)
        >>> result = soft_nw(scores, gap=-1.0, temperature=1.0)
        >>> loss = result.score.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)
    _validate_gap(gap)

    if lengths is None:
        lengths = _make_lengths(scores)

    gap_is_tensor = isinstance(gap, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if gap_is_tensor or temp_is_tensor:
        if not gap_is_tensor:
            gap = torch.tensor([gap], device=scores.device)
        if not temp_is_tensor:
            temperature = torch.tensor([temperature], device=scores.device)
        result = _ops.soft_nw(scores, gap, temperature, lengths)
    else:
        result = _ops.soft_nw_float(scores, gap, temperature, lengths)

    return NWResult(result[0], result[1])


def soft_nw_affine(
    scores: torch.Tensor,
    gap_open: Union[float, torch.Tensor] = -2.0,
    gap_ext: Union[float, torch.Tensor] = -0.5,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> NWResult:
    """Compute soft Needleman-Wunsch with affine gap penalties.

    Args:
        scores: Similarity/substitution matrix [B, L1, L2]
        gap_open: Gap opening penalty (default: -2.0).
        gap_ext: Gap extension penalty (default: -0.5).
        temperature: Temperature for softmax (default: 1.0).
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        NWResult with score [B] and alignment [B, L1, L2].
    """
    _validate_temperature(temperature)
    _validate_gap(gap_open, "gap_open")
    _validate_gap(gap_ext, "gap_ext")

    if lengths is None:
        lengths = _make_lengths(scores)

    open_is_tensor = isinstance(gap_open, torch.Tensor)
    ext_is_tensor = isinstance(gap_ext, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if open_is_tensor or ext_is_tensor or temp_is_tensor:
        if not open_is_tensor:
            gap_open = torch.tensor([gap_open], device=scores.device)
        if not ext_is_tensor:
            gap_ext = torch.tensor([gap_ext], device=scores.device)
        if not temp_is_tensor:
            temperature = torch.tensor([temperature], device=scores.device)
        result = _ops.soft_nw_affine(scores, gap_open, gap_ext, temperature, lengths)
    else:
        result = _ops.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, lengths)

    return NWResult(result[0], result[1])


class SoftNW(nn.Module):
    """Differentiable Needleman-Wunsch global alignment module.

    Args:
        gap: Gap penalty (default: -1.0).
        temperature: Temperature parameter (default: 1.0).
        learnable: Whether parameters are learnable (default: False).
    """

    def __init__(
        self,
        gap: float = -1.0,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.gap = nn.Parameter(torch.tensor([gap]))
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("gap", torch.tensor([gap]))
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> NWResult:
        if lengths is None:
            lengths = _make_lengths(scores)

        result = _ops.soft_nw(scores, self.gap, self.temperature, lengths)
        return NWResult(result[0], result[1])


class SoftNWAffine(nn.Module):
    """Differentiable Needleman-Wunsch with affine gap penalties.

    Args:
        gap_open: Gap opening penalty (default: -2.0).
        gap_ext: Gap extension penalty (default: -0.5).
        temperature: Temperature parameter (default: 1.0).
        learnable: Whether parameters are learnable (default: False).
    """

    def __init__(
        self,
        gap_open: float = -2.0,
        gap_ext: float = -0.5,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.gap_open = nn.Parameter(torch.tensor([gap_open]))
            self.gap_ext = nn.Parameter(torch.tensor([gap_ext]))
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("gap_open", torch.tensor([gap_open]))
            self.register_buffer("gap_ext", torch.tensor([gap_ext]))
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> NWResult:
        if lengths is None:
            lengths = _make_lengths(scores)

        result = _ops.soft_nw_affine(
            scores, self.gap_open, self.gap_ext, self.temperature, lengths
        )
        return NWResult(result[0], result[1])


# Low-Level API
soft_nw_forward = _ops.soft_nw_float
soft_nw_with_grads = _ops.soft_nw_with_grads
soft_nw_hvp = _ops.soft_nw_hvp
soft_nw_backward_full = _ops.soft_nw_backward_full

soft_nw_affine_forward = _ops.soft_nw_affine_float
soft_nw_affine_with_grads = _ops.soft_nw_affine_with_grads
soft_nw_affine_hvp = _ops.soft_nw_affine_hvp
soft_nw_affine_backward_full = _ops.soft_nw_affine_backward_full
