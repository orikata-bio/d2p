# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Smith-Waterman module for soft local sequence alignment.

This module provides PyTorch-compatible implementations of:
- Regular soft Smith-Waterman with linear gap penalty
- Affine soft Smith-Waterman with gap open/extend penalties

Both implementations support:
- Full autodiff through scores, gap parameters, and temperature
- Hessian-vector products for second-order optimization
- Batched computation with variable-length sequences

API Overview:

High-Level API (Normal Users):
    result = soft_sw(scores, gap=-1.0, temperature=1.0)
    result = soft_sw_affine(scores, gap_open=-2.0, gap_ext=-0.5, temperature=1.0)
    # result.score [B], result.alignment [B, L1, L2]

Module API (nn.Module wrappers):
    sw = SoftSW(gap=-1.0, temperature=1.0, learnable=True)
    result = sw(scores)

Low-Level API (via d2p.ops):
    from d2p import ops
    score, alignment = ops.soft_sw_float(scores, gap, temperature, lengths)
"""

import warnings
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


# =============================================================================
# Result Types
# =============================================================================


class SWResult(NamedTuple):
    """Result of soft Smith-Waterman alignment.

    Primary attributes (new API):
        value: Log partition function of shape [B].
        marginals: Soft alignment matrix of shape [B, L1, L2].

    Backward-compatibility aliases:
        score: Alias for value.
        alignment: Alias for marginals.
        posteriors: Alias for marginals.
    """
    value: torch.Tensor
    marginals: torch.Tensor

    @property
    def score(self) -> torch.Tensor:
        """Alias for value (backward compatibility)."""
        return self.value

    @property
    def alignment(self) -> torch.Tensor:
        """Alias for marginals (backward compatibility)."""
        return self.marginals

    @property
    def posteriors(self) -> torch.Tensor:
        """Alias for marginals."""
        return self.marginals


# =============================================================================
# Validation Helpers
# =============================================================================


def _validate_temperature(
    temperature: Union[float, torch.Tensor],
    name: str = "temperature",
) -> None:
    """Validate temperature parameter.

    Raises ValueError if temperature is non-positive for:
    - Python scalars (int, float)
    - Non-learnable scalar tensors (requires_grad=False, numel=1)
    """
    if isinstance(temperature, (int, float)):
        if temperature <= 0:
            raise ValueError(f"{name} must be positive, got {temperature}")
    elif isinstance(temperature, torch.Tensor):
        if use_pt2_ops(temperature):
            return
        if not temperature.requires_grad and temperature.numel() == 1:
            val = temperature.item()
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")


def _validate_gap(
    gap: Union[float, torch.Tensor],
    name: str = "gap",
) -> None:
    """Warn about positive gap (likely mistake) - only for fixed Python scalars."""
    if isinstance(gap, (int, float)) and gap > 0:
        warnings.warn(
            f"{name}={gap} is positive. Gap penalties are typically negative. "
            "If this is intentional, pass as tensor to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )


def _normalize_param_tensor(
    param: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Normalize a parameter tensor to match scores device/dtype.

    Handles:
    - 0-D tensors (scalars) -> reshape to [1]
    - Wrong device -> move to scores.device
    - Wrong dtype -> cast to scores.dtype

    Args:
        param: Parameter tensor (may be 0-D, wrong device, or wrong dtype)
        scores: Reference tensor for device and dtype

    Returns:
        Normalized parameter tensor with shape [1], matching device and dtype
    """
    # Reshape 0-D to 1-D
    if param.dim() == 0:
        param = param.view(1)

    # Move to correct device and dtype
    if param.device != scores.device or param.dtype != scores.dtype:
        param = param.to(device=scores.device, dtype=scores.dtype)

    return param


def _make_lengths(scores: torch.Tensor) -> torch.Tensor:
    """Create default lengths tensor for full sequences."""
    B, L1, L2 = scores.shape
    return torch.tensor(
        [[L1, L2]] * B, dtype=torch.int32, device=scores.device
    )


def _validate_prefix_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Validate that mask is prefix-only and return lengths.

    A prefix-only mask has all True values at the beginning and all False
    values at the end. No holes (False values before True values) are allowed.

    Args:
        mask: Boolean mask [B, L] where True = valid position
        name: Name of mask for error messages

    Returns:
        lengths: Tensor [B] of sequence lengths

    Raises:
        ValueError: If mask is not prefix-only (has holes)
    """
    # Sum of True values gives the expected length
    lengths = mask.sum(dim=1)

    # For prefix-only masks, cumsum should equal a ramp from 1 to length
    # If there are holes, cumsum will be less than expected at some positions
    B, L = mask.shape

    # Create expected cumsum for prefix-only mask: [1, 2, 3, ..., length, length, ...]
    # First, get the cumsum of the actual mask
    cumsum = mask.to(torch.int32).cumsum(dim=1)

    # For each position i, if mask is prefix-only:
    # - For i < length: cumsum[i] should equal i + 1
    # - For i >= length: cumsum[i] should equal length
    #
    # Equivalently: at position length-1, cumsum should equal length
    # AND at position L-1, cumsum should still equal length
    # AND the mask at position length should be False (if length < L)

    # Simple validation: if any False appears before a True, it's not prefix-only
    # Check: mask values should be non-increasing (True=1, False=0)
    # If mask[b, i] is False and mask[b, i+1] is True, we have a hole
    mask_int = mask.to(torch.int32)
    diff = mask_int[:, 1:] - mask_int[:, :-1]  # [B, L-1]
    has_hole = (diff > 0).any(dim=1)  # [B]

    if has_hole.any():
        bad_indices = has_hole.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            f"{name} is not prefix-only (has holes) at batch indices: {bad_indices}. "
            f"Masks must have all True values at the beginning, followed by all False values."
        )

    return lengths.to(torch.int32)


def _masks_to_lengths(
    mask1: Optional[torch.Tensor],
    mask2: Optional[torch.Tensor],
    B: int,
    L1: int,
    L2: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert prefix boolean masks to lengths tensor.

    Args:
        mask1: Boolean mask [B, L1] where True = valid position
        mask2: Boolean mask [B, L2] where True = valid position
        B, L1, L2: Batch and sequence dimensions
        device: Target device

    Returns:
        lengths: Tensor [B, 2] of sequence lengths

    Raises:
        ValueError: If masks are not prefix-only (have holes)
    """
    lengths = torch.empty((B, 2), dtype=torch.int32, device=device)

    if mask1 is not None:
        lengths[:, 0] = _validate_prefix_mask(mask1, "mask1")
    else:
        lengths[:, 0] = L1

    if mask2 is not None:
        lengths[:, 1] = _validate_prefix_mask(mask2, "mask2")
    else:
        lengths[:, 1] = L2

    return lengths


# =============================================================================
# High-Level API
# =============================================================================


def soft_sw(
    scores: torch.Tensor,
    gap: Union[float, torch.Tensor] = -1.0,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
    *,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
) -> SWResult:
    """Compute soft Smith-Waterman local alignment.

    Smith-Waterman finds the best local alignment (subsequence match)
    between two sequences. As temperature -> 0, this recovers hard SW.

    Args:
        scores: Similarity/substitution matrix [B, L1, L2]
        gap: Gap penalty (typically negative, default: -1.0).
            Can be a float or scalar tensor. If tensor with requires_grad=True,
            gradients will flow through the gap parameter.
        temperature: Temperature for soft-max (default: 1.0).
            Lower values approach hard alignment.
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        mask1: Optional boolean mask [B, L1] for sequence 1 (True = valid).
            Cannot be used together with lengths.
        mask2: Optional boolean mask [B, L2] for sequence 2 (True = valid).
            Cannot be used together with lengths.

    Returns:
        SWResult with:
            - value: Soft alignment score (log partition function) [B]
            - marginals: Soft alignment matrix (marginal probabilities) [B, L1, L2]
            (Also accessible via .score and .alignment for backward compatibility)

    Example:
        >>> scores = torch.randn(2, 100, 120, device="cuda", requires_grad=True)
        >>> result = soft_sw(scores, gap=-1.0, temperature=1.0)
        >>> loss = result.score.sum()
        >>> loss.backward()

        >>> # Learnable parameters
        >>> gap = torch.tensor([-1.0], device="cuda", requires_grad=True)
        >>> result = soft_sw(scores, gap=gap, temperature=1.0)

        >>> # Using masks instead of lengths
        >>> mask1 = torch.tensor([[True, True, True, False, False]] * 2)
        >>> mask2 = torch.tensor([[True, True, True, True, False, False, False]] * 2)
        >>> result = soft_sw(scores, mask1=mask1, mask2=mask2)
    """
    _validate_temperature(temperature)
    _validate_gap(gap)

    B, L1, L2 = scores.shape

    # Validate mutual exclusivity
    if lengths is not None and (mask1 is not None or mask2 is not None):
        raise ValueError("Cannot specify both 'lengths' and 'mask1'/'mask2'")

    # Convert masks to lengths if provided
    if mask1 is not None or mask2 is not None:
        lengths = _masks_to_lengths(mask1, mask2, B, L1, L2, scores.device)
    elif lengths is None:
        lengths = _make_lengths(scores)

    gap_is_tensor = isinstance(gap, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if gap_is_tensor or temp_is_tensor:
        # Normalize tensor parameters: ensure [1] shape, correct device/dtype
        if gap_is_tensor:
            gap = _normalize_param_tensor(gap, scores)
        else:
            gap = torch.tensor([gap], device=scores.device, dtype=scores.dtype)

        if temp_is_tensor:
            temperature = _normalize_param_tensor(temperature, scores)
        else:
            temperature = torch.tensor(
                [temperature], device=scores.device, dtype=scores.dtype
            )

        result = _ops.soft_sw(scores, gap, temperature, lengths)
    else:
        result = _ops.soft_sw_float(scores, gap, temperature, lengths)

    return SWResult(result[0], result[1])


def soft_sw_affine(
    scores: torch.Tensor,
    gap_open: Union[float, torch.Tensor] = -2.0,
    gap_ext: Union[float, torch.Tensor] = -0.5,
    temperature: Union[float, torch.Tensor] = 1.0,
    lengths: Optional[torch.Tensor] = None,
    *,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
) -> SWResult:
    """Compute soft Smith-Waterman with affine gap penalties.

    Uses the standard 3-state affine gap model where the gap penalty
    is: gap_open + k * gap_ext for a gap of length k.

    Args:
        scores: Similarity/substitution matrix [B, L1, L2]
        gap_open: Gap opening penalty (typically negative, default: -2.0).
        gap_ext: Gap extension penalty (typically negative, default: -0.5).
        temperature: Temperature for soft-max (default: 1.0).
        lengths: Optional [B, 2] tensor of actual sequence lengths.
        mask1: Optional boolean mask [B, L1] for sequence 1 (True = valid).
            Cannot be used together with lengths.
        mask2: Optional boolean mask [B, L2] for sequence 2 (True = valid).
            Cannot be used together with lengths.

    Returns:
        SWResult with:
            - value: Soft alignment score (log partition function) [B]
            - marginals: Soft alignment matrix (marginal probabilities) [B, L1, L2]
            (Also accessible via .score and .alignment for backward compatibility)

    Example:
        >>> scores = torch.randn(2, 100, 120, device="cuda", requires_grad=True)
        >>> result = soft_sw_affine(scores, gap_open=-2.0, gap_ext=-0.5)
        >>> loss = result.value.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)
    _validate_gap(gap_open, "gap_open")
    _validate_gap(gap_ext, "gap_ext")

    B, L1, L2 = scores.shape

    # Validate mutual exclusivity
    if lengths is not None and (mask1 is not None or mask2 is not None):
        raise ValueError("Cannot specify both 'lengths' and 'mask1'/'mask2'")

    # Convert masks to lengths if provided
    if mask1 is not None or mask2 is not None:
        lengths = _masks_to_lengths(mask1, mask2, B, L1, L2, scores.device)
    elif lengths is None:
        lengths = _make_lengths(scores)

    open_is_tensor = isinstance(gap_open, torch.Tensor)
    ext_is_tensor = isinstance(gap_ext, torch.Tensor)
    temp_is_tensor = isinstance(temperature, torch.Tensor)

    if open_is_tensor or ext_is_tensor or temp_is_tensor:
        # Normalize tensor parameters: ensure [1] shape, correct device/dtype
        if open_is_tensor:
            gap_open = _normalize_param_tensor(gap_open, scores)
        else:
            gap_open = torch.tensor(
                [gap_open], device=scores.device, dtype=scores.dtype
            )

        if ext_is_tensor:
            gap_ext = _normalize_param_tensor(gap_ext, scores)
        else:
            gap_ext = torch.tensor(
                [gap_ext], device=scores.device, dtype=scores.dtype
            )

        if temp_is_tensor:
            temperature = _normalize_param_tensor(temperature, scores)
        else:
            temperature = torch.tensor(
                [temperature], device=scores.device, dtype=scores.dtype
            )

        result = _ops.soft_sw_affine(scores, gap_open, gap_ext, temperature, lengths)
    else:
        result = _ops.soft_sw_affine_float(
            scores, gap_open, gap_ext, temperature, lengths
        )

    return SWResult(result[0], result[1])


# =============================================================================
# Module API (nn.Module wrappers)
# =============================================================================


class SoftSW(nn.Module):
    """Differentiable Smith-Waterman alignment module.

    Args:
        gap: Gap penalty (default: -1.0).
        temperature: Temperature parameter (default: 1.0).
        learnable: DEPRECATED. Use learn_gap/learn_temperature instead.
        learn_gap: Whether gap is learnable (default: False).
        learn_temperature: Whether temperature is learnable (default: False).

    Example:
        >>> sw = SoftSW(gap=-1.0, temperature=1.0, learn_gap=True)
        >>> scores = torch.randn(2, 100, 120, device="cuda")
        >>> result = sw(scores)
        >>> loss = result.value.sum()
        >>> loss.backward()
        >>> print(sw.gap.grad)
    """

    def __init__(
        self,
        gap: float = -1.0,
        temperature: float = 1.0,
        learnable: Optional[bool] = None,
        learn_gap: bool = False,
        learn_temperature: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Handle deprecated 'learnable' parameter
        if learnable is not None:
            warnings.warn(
                "The 'learnable' parameter is deprecated. "
                "Use 'learn_gap' and/or 'learn_temperature' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if learnable:
                learn_gap = True
                learn_temperature = True

        if learn_gap:
            self.gap = nn.Parameter(torch.tensor([gap]))
        else:
            self.register_buffer("gap", torch.tensor([gap]))

        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        *,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> SWResult:
        """Compute soft alignment.

        Args:
            scores: Similarity matrix [B, L1, L2]
            lengths: Optional [B, 2] tensor of actual sequence lengths.
            mask1: Optional boolean mask [B, L1] for sequence 1 (True = valid).
                Cannot be used together with lengths.
            mask2: Optional boolean mask [B, L2] for sequence 2 (True = valid).
                Cannot be used together with lengths.

        Returns:
            SWResult with value [B] and marginals [B, L1, L2].
        """
        return soft_sw(
            scores,
            gap=self.gap,
            temperature=self.temperature,
            lengths=lengths,
            mask1=mask1,
            mask2=mask2,
        )


class SoftSWAffine(nn.Module):
    """Differentiable Smith-Waterman with affine gap penalties.

    Args:
        gap_open: Gap opening penalty (default: -2.0).
        gap_ext: Gap extension penalty (default: -0.5).
        temperature: Temperature parameter (default: 1.0).
        learnable: DEPRECATED. Use learn_gap_open/learn_gap_ext/learn_temperature instead.
        learn_gap_open: Whether gap_open is learnable (default: False).
        learn_gap_ext: Whether gap_ext is learnable (default: False).
        learn_temperature: Whether temperature is learnable (default: False).

    Example:
        >>> sw = SoftSWAffine(gap_open=-2.0, gap_ext=-0.5, learn_gap_open=True)
        >>> scores = torch.randn(2, 100, 120, device="cuda")
        >>> result = sw(scores)
    """

    def __init__(
        self,
        gap_open: float = -2.0,
        gap_ext: float = -0.5,
        temperature: float = 1.0,
        learnable: Optional[bool] = None,
        learn_gap_open: bool = False,
        learn_gap_ext: bool = False,
        learn_temperature: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        # Handle deprecated 'learnable' parameter
        if learnable is not None:
            warnings.warn(
                "The 'learnable' parameter is deprecated. "
                "Use 'learn_gap_open', 'learn_gap_ext', and/or 'learn_temperature' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if learnable:
                learn_gap_open = True
                learn_gap_ext = True
                learn_temperature = True

        if learn_gap_open:
            self.gap_open = nn.Parameter(torch.tensor([gap_open]))
        else:
            self.register_buffer("gap_open", torch.tensor([gap_open]))

        if learn_gap_ext:
            self.gap_ext = nn.Parameter(torch.tensor([gap_ext]))
        else:
            self.register_buffer("gap_ext", torch.tensor([gap_ext]))

        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        scores: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        *,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None,
    ) -> SWResult:
        """Compute soft alignment.

        Args:
            scores: Similarity matrix [B, L1, L2]
            lengths: Optional [B, 2] tensor of actual sequence lengths.
            mask1: Optional boolean mask [B, L1] for sequence 1 (True = valid).
                Cannot be used together with lengths.
            mask2: Optional boolean mask [B, L2] for sequence 2 (True = valid).
                Cannot be used together with lengths.

        Returns:
            SWResult with value [B] and marginals [B, L1, L2].
        """
        return soft_sw_affine(
            scores,
            gap_open=self.gap_open,
            gap_ext=self.gap_ext,
            temperature=self.temperature,
            lengths=lengths,
            mask1=mask1,
            mask2=mask2,
        )


# =============================================================================
# Low-Level API (re-exported for convenience)
# =============================================================================

# Direct access to ops for advanced users
soft_sw_forward = _ops.soft_sw_float
soft_sw_with_grads = _ops.soft_sw_with_grads
soft_sw_hvp = _ops.soft_sw_hvp
soft_sw_backward_full = _ops.soft_sw_backward_full

soft_sw_affine_forward = _ops.soft_sw_affine_float
soft_sw_affine_with_grads = _ops.soft_sw_affine_with_grads
soft_sw_affine_hvp = _ops.soft_sw_affine_hvp
soft_sw_affine_backward_full = _ops.soft_sw_affine_backward_full
