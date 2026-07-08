# SPDX-License-Identifier: Apache-2.0
"""
Differentiable Dynamic Time Warping module.

DTW finds the optimal alignment between two sequences by warping the time axis.
Uses softmin (cost-based) formulation with optional Sakoe-Chiba band constraint.

API Overview:

High-Level API:
    result = soft_dtw(costs, temperature=1.0)
    # result.cost [B], result.alignment [B, L1, L2]

Module API:
    dtw = SoftDTW(temperature=1.0, learnable=True)
    result = dtw(costs)

Low-Level API (via d2p.ops):
    cost, alignment = ops.soft_dtw_float(costs, temperature, bandwidth, lengths)
"""

import torch
import torch.nn as nn
from typing import Optional, Union, NamedTuple

from . import _ops
from ._pt2_utils import use_pt2_ops


class DTWResult(NamedTuple):
    """Result of soft Dynamic Time Warping.

    Attributes:
        cost: Soft DTW cost (soft minimum alignment cost) [B].
        alignment: Soft alignment matrix [B, L1, L2].
    """
    cost: torch.Tensor
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


def _make_lengths(costs: torch.Tensor) -> torch.Tensor:
    B, L1, L2 = costs.shape
    return torch.tensor([[L1, L2]] * B, dtype=torch.int32, device=costs.device)


def soft_dtw(
    costs: torch.Tensor,
    temperature: Union[float, torch.Tensor] = 1.0,
    bandwidth: int = 0,
    lengths: Optional[torch.Tensor] = None,
) -> DTWResult:
    """Compute soft Dynamic Time Warping alignment.

    DTW finds the optimal warping path through a cost matrix, allowing
    sequences of different lengths to be aligned. Uses softmin for
    differentiability.

    Args:
        costs: Cost matrix [B, L1, L2] (lower = better match)
        temperature: Temperature for softmin (default: 1.0).
            Lower values approach hard DTW.
        bandwidth: Sakoe-Chiba band width (0 = no constraint, default: 0).
            Restricts alignment to within bandwidth of diagonal.
        lengths: Optional [B, 2] tensor of actual sequence lengths.

    Returns:
        DTWResult with:
            - cost: Soft DTW cost [B]
            - alignment: Soft alignment matrix [B, L1, L2]

    Example:
        >>> costs = torch.rand(2, 100, 120, device="cuda", requires_grad=True)
        >>> result = soft_dtw(costs, temperature=1.0)
        >>> loss = result.cost.sum()
        >>> loss.backward()
    """
    _validate_temperature(temperature)

    if lengths is None:
        lengths = _make_lengths(costs)

    temp_is_tensor = isinstance(temperature, torch.Tensor)

    # bandwidth of 0 means no constraint; the kernels treat bandwidth < 0 as
    # "no band", so use the int sentinel -1. (Both ops require an int; the
    # tensor-temperature op `soft_dtw` is non-optional and rejects None.)
    bw = bandwidth if bandwidth > 0 else -1

    if temp_is_tensor:
        result = _ops.soft_dtw(costs, temperature, lengths, bw)
    else:
        result = _ops.soft_dtw_float(costs, temperature, lengths, bw)

    return DTWResult(result[0], result[1])


class SoftDTW(nn.Module):
    """Differentiable Dynamic Time Warping module.

    Args:
        temperature: Temperature for softmin (default: 1.0).
        bandwidth: Sakoe-Chiba band width (0 = no constraint, default: 0).
        learnable: Whether temperature is learnable (default: False).

    Example:
        >>> dtw = SoftDTW(temperature=1.0, bandwidth=10, learnable=True)
        >>> costs = torch.rand(2, 100, 120, device="cuda")
        >>> result = dtw(costs)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        bandwidth: int = 0,
        learnable: bool = False,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.bandwidth = bandwidth

        if learnable:
            self.temperature = nn.Parameter(torch.tensor([temperature]))
        else:
            self.register_buffer("temperature", torch.tensor([temperature]))

    def forward(
        self,
        costs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> DTWResult:
        if lengths is None:
            lengths = _make_lengths(costs)

        bw = self.bandwidth if self.bandwidth > 0 else -1
        result = _ops.soft_dtw(costs, self.temperature, lengths, bw)
        return DTWResult(result[0], result[1])


# Low-Level API
soft_dtw_forward = _ops.soft_dtw_float
soft_dtw_with_grads = _ops.soft_dtw_with_grads
soft_dtw_hvp = _ops.soft_dtw_hvp
soft_dtw_backward_full = _ops.soft_dtw_backward_full
