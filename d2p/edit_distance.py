# SPDX-License-Identifier: Apache-2.0
"""
Differentiable edit distance modules.

This module provides differentiable versions of edit distance algorithms:
- Levenshtein: Standard edit distance (insert, delete, substitute)
- LCS: Longest Common Subsequence (maximize matches, no gap penalty)
- OSA: Optimal String Alignment (restricted transpositions)
- Damerau-Levenshtein: True DL with unrestricted transpositions

All algorithms use cost-based (softmin) formulation.

API Overview:

High-Level API:
    result = soft_levenshtein(costs, temperature=1.0)
    result = soft_lcs(match_scores, temperature=1.0)
    result = soft_osa(costs, trans_costs, temperature=1.0)
    result = soft_damerau(costs, trans_costs, temperature=1.0)

Module API:
    lev = SoftLevenshtein(temperature=1.0, learnable=True)
    result = lev(costs)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


# =============================================================================
# Result Types
# =============================================================================


class LevenshteinResult(NamedTuple):
    """Result of soft Levenshtein distance.

    Attributes:
        distance: Soft edit distance [B].
        alignment: Soft alignment matrix [B, L1, L2].
    """
    distance: torch.Tensor
    alignment: torch.Tensor


class LCSResult(NamedTuple):
    """Result of soft Longest Common Subsequence.

    Attributes:
        score: Soft LCS score (higher = longer common subsequence) [B].
        alignment: Soft alignment matrix [B, L1, L2].
    """
    score: torch.Tensor
    alignment: torch.Tensor


class OSAResult(NamedTuple):
    """Result of soft Optimal String Alignment distance.

    Attributes:
        distance: Soft OSA distance [B].
        alignment: Soft alignment matrix [B, L1, L2].
    """
    distance: torch.Tensor
    alignment: torch.Tensor


class DamerauResult(NamedTuple):
    """Result of soft Damerau-Levenshtein distance.

    Attributes:
        distance: Soft DL distance [B].
        alignment: Soft alignment matrix [B, L1, L2].
    """
    distance: torch.Tensor
    alignment: torch.Tensor


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_temperature(temperature: Union[float, torch.Tensor]) -> None:
    if isinstance(temperature, (int, float)) and temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    elif isinstance(temperature, torch.Tensor):
        if use_pt2_ops(temperature):
            return
        if not temperature.requires_grad and temperature.numel() == 1:
            if temperature.item() <= 0:
                raise ValueError(f"temperature must be positive, got {temperature.item()}")


def _make_lengths_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Make lengths for [B, L1, L2] shaped tensors."""
    B, L1, L2 = tensor.shape
    return torch.tensor([[L1, L2]] * B, dtype=torch.int32, device=tensor.device)


# =============================================================================
# Levenshtein (Standard Edit Distance)
# =============================================================================


def soft_levenshtein(
    sub_costs: torch.Tensor,
    ins_cost: Union[float, torch.Tensor] = 1.0,
    del_cost: Union[float, torch.Tensor] = 1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> LevenshteinResult:
    """Compute soft Levenshtein edit distance.

    Finds the minimum cost to transform sequence 1 into sequence 2 using
    insertions, deletions, and substitutions.

    Args:
        sub_costs: Substitution cost matrix [B, L1, L2]. sub_costs[b, i, j] is
            the cost of substituting position i in seq1 with position j in seq2.
        ins_cost: Scalar insertion cost or learnable tensor (default: 1.0).
        del_cost: Scalar deletion cost or learnable tensor (default: 1.0).
        temperature: Temperature for softmin (default: 1.0). Can be learnable.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        LevenshteinResult with distance [B] and alignment [B, L1, L2].

    Example:
        >>> sub_costs = torch.rand(2, 10, 12, device="cuda", requires_grad=True)
        >>> result = soft_levenshtein(sub_costs, ins_cost=1.0, del_cost=1.0)
        >>> loss = result.distance.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths_2d(sub_costs)

    ins_is_tensor = isinstance(ins_cost, torch.Tensor)
    del_is_tensor = isinstance(del_cost, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if ins_is_tensor or del_is_tensor or temp_is_tensor:
        if not ins_is_tensor:
            ins_cost = sub_costs.new_tensor([ins_cost])
        if not del_is_tensor:
            del_cost = sub_costs.new_tensor([del_cost])
        if not temp_is_tensor:
            temperature = sub_costs.new_tensor([temperature])
        result = _ops.soft_levenshtein(
            sub_costs, ins_cost, del_cost, temperature, lengths
        )
    else:
        result = _ops.soft_levenshtein_float(
            sub_costs, ins_cost, del_cost, temperature, lengths
        )
    return LevenshteinResult(result[0], result[1])


class SoftLevenshtein(nn.Module):
    """Differentiable Levenshtein edit distance module.

    Args:
        ins_cost: Cost of insertion (default: 1.0).
        del_cost: Cost of deletion (default: 1.0).
        temperature: Temperature for softmin (default: 1.0).
        learnable: Whether parameters are learnable (default: False).
    """

    def __init__(
        self,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.ins_cost = nn.Parameter(torch.tensor([ins_cost]))
            self.del_cost = nn.Parameter(torch.tensor([del_cost]))
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("ins_cost", torch.tensor([ins_cost]))
            self.register_buffer("del_cost", torch.tensor([del_cost]))
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        sub_costs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> LevenshteinResult:
        return soft_levenshtein(
            sub_costs,
            self.ins_cost,
            self.del_cost,
            self.temperature,
            lengths,
        )


# =============================================================================
# LCS (Longest Common Subsequence)
# =============================================================================


def soft_lcs(
    match_scores: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> LCSResult:
    """Compute soft Longest Common Subsequence.

    LCS finds the longest subsequence common to both sequences. Unlike
    alignment, gaps have no penalty - we only reward matches.

    Args:
        match_scores: Match score matrix [B, L1, L2]. Higher scores indicate
            better matches between positions.
        temperature: Temperature for softmax (default: 1.0).
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        LCSResult with score [B] and alignment [B, L1, L2].

    Example:
        >>> match_scores = torch.rand(2, 10, 12, device="cuda", requires_grad=True)
        >>> result = soft_lcs(match_scores, temperature=1.0)
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths_2d(match_scores)

    if isinstance(temperature, torch.Tensor):
        result = _ops.soft_lcs(match_scores, temperature, lengths)
    else:
        result = _ops.soft_lcs_float(match_scores, temperature, lengths)
    return LCSResult(result[0], result[1])


class SoftLCS(nn.Module):
    """Differentiable Longest Common Subsequence module.

    Args:
        temperature: Temperature for softmax (default: 1.0).
        learnable: Whether temperature is learnable (default: False).
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
        match_scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> LCSResult:
        return soft_lcs(match_scores, self.temperature, lengths)


# =============================================================================
# OSA (Optimal String Alignment / Restricted Damerau-Levenshtein)
# =============================================================================


def soft_osa(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: Union[float, torch.Tensor] = 1.0,
    del_cost: Union[float, torch.Tensor] = 1.0,
    trans_cost: Union[float, torch.Tensor] = 1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> OSAResult:
    """Compute soft Optimal String Alignment distance.

    OSA extends Levenshtein with adjacent transpositions but doesn't allow
    editing a substring more than once (restricted transpositions).

    Args:
        sub_costs: Substitution cost matrix [B, L1, L2].
        trans_mask: Transposition mask [B, L1-1, L2-1] (1 where trans allowed).
        ins_cost: Scalar insertion cost or learnable tensor (default: 1.0).
        del_cost: Scalar deletion cost or learnable tensor (default: 1.0).
        trans_cost: Scalar transposition cost or learnable tensor (default: 1.0).
        temperature: Temperature for softmin (default: 1.0). Can be learnable.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        OSAResult with distance [B] and alignment [B, L1, L2].
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths_2d(sub_costs)

    ins_is_tensor = isinstance(ins_cost, torch.Tensor)
    del_is_tensor = isinstance(del_cost, torch.Tensor)
    trans_is_tensor = isinstance(trans_cost, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if ins_is_tensor or del_is_tensor or trans_is_tensor or temp_is_tensor:
        if not ins_is_tensor:
            ins_cost = sub_costs.new_tensor([ins_cost])
        if not del_is_tensor:
            del_cost = sub_costs.new_tensor([del_cost])
        if not trans_is_tensor:
            trans_cost = sub_costs.new_tensor([trans_cost])
        if not temp_is_tensor:
            temperature = sub_costs.new_tensor([temperature])
        result = _ops.soft_osa(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
    else:
        result = _ops.soft_osa_float(
            sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
        )
    return OSAResult(result[0], result[1])


class SoftOSA(nn.Module):
    """Differentiable Optimal String Alignment distance module."""

    def __init__(
        self,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        trans_cost: float = 1.0,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.ins_cost = nn.Parameter(torch.tensor([ins_cost]))
            self.del_cost = nn.Parameter(torch.tensor([del_cost]))
            self.trans_cost = nn.Parameter(torch.tensor([trans_cost]))
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("ins_cost", torch.tensor([ins_cost]))
            self.register_buffer("del_cost", torch.tensor([del_cost]))
            self.register_buffer("trans_cost", torch.tensor([trans_cost]))
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        sub_costs: torch.Tensor,
        trans_mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> OSAResult:
        return soft_osa(
            sub_costs,
            trans_mask,
            self.ins_cost,
            self.del_cost,
            self.trans_cost,
            self.temperature,
            lengths,
        )


# =============================================================================
# Damerau-Levenshtein (True DL with unrestricted transpositions)
# =============================================================================


def soft_damerau(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: Union[float, torch.Tensor] = 1.0,
    del_cost: Union[float, torch.Tensor] = 1.0,
    trans_cost: Union[float, torch.Tensor] = 1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
) -> DamerauResult:
    """Compute soft Damerau-Levenshtein distance.

    True Damerau-Levenshtein allows unrestricted transpositions (characters
    can be edited after transposition). More complex than OSA but handles
    more editing scenarios correctly.

    Args:
        sub_costs: Substitution cost matrix [B, L1, L2].
        trans_src: Transposition source tensor [B, L1, L2].
        ins_cost: Scalar insertion cost or learnable tensor (default: 1.0).
        del_cost: Scalar deletion cost or learnable tensor (default: 1.0).
        trans_cost: Scalar transposition cost or learnable tensor (default: 1.0).
        temperature: Temperature for softmin (default: 1.0). Can be learnable.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        DamerauResult with distance [B] and alignment [B, L1, L2].
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths_2d(sub_costs)

    ins_is_tensor = isinstance(ins_cost, torch.Tensor)
    del_is_tensor = isinstance(del_cost, torch.Tensor)
    trans_is_tensor = isinstance(trans_cost, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if ins_is_tensor or del_is_tensor or trans_is_tensor or temp_is_tensor:
        if not ins_is_tensor:
            ins_cost = sub_costs.new_tensor([ins_cost])
        if not del_is_tensor:
            del_cost = sub_costs.new_tensor([del_cost])
        if not trans_is_tensor:
            trans_cost = sub_costs.new_tensor([trans_cost])
        if not temp_is_tensor:
            temperature = sub_costs.new_tensor([temperature])
        result = _ops.soft_damerau(
            sub_costs,
            trans_src,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
    else:
        result = _ops.soft_damerau_float(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
        )
    return DamerauResult(result[0], result[1])


class SoftDamerau(nn.Module):
    """Differentiable Damerau-Levenshtein distance module."""

    def __init__(
        self,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        trans_cost: float = 1.0,
        temperature: float = 1.0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        if learnable:
            self.ins_cost = nn.Parameter(torch.tensor([ins_cost]))
            self.del_cost = nn.Parameter(torch.tensor([del_cost]))
            self.trans_cost = nn.Parameter(torch.tensor([trans_cost]))
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("ins_cost", torch.tensor([ins_cost]))
            self.register_buffer("del_cost", torch.tensor([del_cost]))
            self.register_buffer("trans_cost", torch.tensor([trans_cost]))
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        sub_costs: torch.Tensor,
        trans_src: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> DamerauResult:
        return soft_damerau(
            sub_costs,
            trans_src,
            self.ins_cost,
            self.del_cost,
            self.trans_cost,
            self.temperature,
            lengths,
        )


# =============================================================================
# Low-Level API
# =============================================================================

# Levenshtein
soft_levenshtein_forward = _ops.soft_levenshtein_float
soft_levenshtein_with_grads = _ops.soft_levenshtein_with_grads
soft_levenshtein_hvp = _ops.soft_levenshtein_hvp
soft_levenshtein_backward_full = _ops.soft_levenshtein_backward_full

# LCS
soft_lcs_forward = _ops.soft_lcs_float
soft_lcs_with_grads = _ops.soft_lcs_with_grads
soft_lcs_hvp = _ops.soft_lcs_hvp
soft_lcs_backward_full = _ops.soft_lcs_backward_full

# OSA
soft_osa_forward = _ops.soft_osa_float
soft_osa_with_grads = _ops.soft_osa_with_grads
soft_osa_hvp = _ops.soft_osa_hvp
soft_osa_param_jacobian = _ops.soft_osa_param_jacobian
soft_osa_backward_full = _ops.soft_osa_backward_full

# Damerau-Levenshtein
soft_damerau_forward = _ops.soft_damerau_float
soft_damerau_with_grads = _ops.soft_damerau_with_grads
soft_damerau_hvp = _ops.soft_damerau_hvp
soft_damerau_param_jacobian = _ops.soft_damerau_param_jacobian
soft_damerau_backward_full = _ops.soft_damerau_backward_full
