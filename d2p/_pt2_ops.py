# SPDX-License-Identifier: Apache-2.0
"""
PT2-compatible custom ops that wrap the C++ extension.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.library import custom_op


def _is_fake_tensor(value: object) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    try:
        from torch._subclasses.fake_tensor import is_fake
    except Exception:
        return False
    return is_fake(value)


def _extract_scalar(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if _is_fake_tensor(value):
            return default
        return float(value.item())
    return float(value)


def _add_grad(existing: Optional[torch.Tensor], new: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if existing is None:
        return new
    if new is None:
        return existing
    return existing + new


def _expand_batch_grad(
    grad_score: Optional[torch.Tensor], alignment: torch.Tensor
) -> Optional[torch.Tensor]:
    if grad_score is None:
        return None
    view_shape = [alignment.size(0)] + [1] * (alignment.dim() - 1)
    return grad_score.view(view_shape) * alignment


def _sum_batch_grad(
    grad_score: Optional[torch.Tensor], grad_param_fwd: torch.Tensor
) -> Optional[torch.Tensor]:
    if grad_score is None:
        return None
    return (grad_score * grad_param_fwd).sum().reshape([1])


def _sum_alignment_grad(
    grad_alignment: Optional[torch.Tensor], jacobian: torch.Tensor
) -> Optional[torch.Tensor]:
    if grad_alignment is None:
        return None
    return (grad_alignment * jacobian).sum().reshape([1])


# =============================================================================
# Smith-Waterman (Regular - Linear Gap)
# =============================================================================


@custom_op("d2p_py::soft_sw_with_grads", mutates_args=())
def soft_sw_with_grads(
    scores: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, alignment, grad_gap, grad_temp = torch.ops.d2p.soft_sw_with_grads(
        scores, gap, temperature, lengths
    )
    return score, alignment, grad_gap, grad_temp


@soft_sw_with_grads.register_fake
def soft_sw_with_grads_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_sw_hvp", mutates_args=())
def soft_sw_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_sw_hvp(scores, tangent, gap, temperature, lengths)


@soft_sw_hvp.register_fake
def soft_sw_hvp_fake(scores, tangent, gap, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_sw_param_jacobian", mutates_args=())
def soft_sw_param_jacobian(
    scores: torch.Tensor,
    param_type: int,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_sw_param_jacobian(
        scores, param_type, gap, temperature, lengths
    )


@soft_sw_param_jacobian.register_fake
def soft_sw_param_jacobian_fake(scores, param_type, gap, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_sw_backward_full", mutates_args=())
def soft_sw_backward_full(
    scores: torch.Tensor,
    grad_alignment: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_sw_backward_full(
        scores, grad_alignment, gap, temperature, lengths
    )


@soft_sw_backward_full.register_fake
def soft_sw_backward_full_fake(scores, grad_alignment, gap, temperature, lengths):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_sw", mutates_args=())
def soft_sw(
    scores: torch.Tensor,
    gap: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_sw(scores, gap, temperature, lengths)
    return score, alignment


@soft_sw.register_fake
def soft_sw_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_sw_setup_context(ctx, inputs, output):
    scores, gap, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap = _extract_scalar(gap)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_sw_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_gap = ctx.needs_input_grad[1]
    needs_temp = ctx.needs_input_grad[2]
    grad_gap = None
    grad_temp = None

    if grad_score is not None and (needs_gap or needs_temp):
        _, _, grad_gap_fwd, grad_temp_fwd = soft_sw_with_grads(
            scores, ctx.gap, ctx.temperature, ctx.lengths
        )
        if needs_gap:
            grad_gap = _sum_batch_grad(grad_score, grad_gap_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_scores_align, grad_gap_align, grad_temp_align = soft_sw_backward_full(
            scores, grad_alignment, ctx.gap, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_gap:
            grad_gap = _add_grad(grad_gap, grad_gap_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_gap, grad_temp, None


soft_sw.register_autograd(_soft_sw_backward, setup_context=_soft_sw_setup_context)


@custom_op("d2p_py::soft_sw_float", mutates_args=())
def soft_sw_float(
    scores: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_sw_float(scores, gap, temperature, lengths)
    return score, alignment


@soft_sw_float.register_fake
def soft_sw_float_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_sw_float_setup_context(ctx, inputs, output):
    scores, gap, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap = gap
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_sw_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _, _ = soft_sw_backward_full(
            scores, grad_alignment, ctx.gap, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None, None


soft_sw_float.register_autograd(
    _soft_sw_float_backward, setup_context=_soft_sw_float_setup_context
)


# =============================================================================
# Smith-Waterman (Affine Gap)
# =============================================================================


@custom_op("d2p_py::soft_sw_affine_with_grads", mutates_args=())
def soft_sw_affine_with_grads(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, alignment, grad_open, grad_ext, grad_temp = (
        torch.ops.d2p.soft_sw_affine_with_grads(
            scores, gap_open, gap_ext, temperature, lengths
        )
    )
    return score, alignment, grad_open, grad_ext, grad_temp


@soft_sw_affine_with_grads.register_fake
def soft_sw_affine_with_grads_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_sw_affine_hvp", mutates_args=())
def soft_sw_affine_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_sw_affine_hvp(
        scores, tangent, gap_open, gap_ext, temperature, lengths
    )


@soft_sw_affine_hvp.register_fake
def soft_sw_affine_hvp_fake(scores, tangent, gap_open, gap_ext, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_sw_affine_param_jacobian", mutates_args=())
def soft_sw_affine_param_jacobian(
    scores: torch.Tensor,
    param_type: int,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_sw_affine_param_jacobian(
        scores, param_type, gap_open, gap_ext, temperature, lengths
    )


@soft_sw_affine_param_jacobian.register_fake
def soft_sw_affine_param_jacobian_fake(
    scores, param_type, gap_open, gap_ext, temperature, lengths
):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_sw_affine_backward_full", mutates_args=())
def soft_sw_affine_backward_full(
    scores: torch.Tensor,
    grad_alignment: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_sw_affine_backward_full(
        scores, grad_alignment, gap_open, gap_ext, temperature, lengths
    )


@soft_sw_affine_backward_full.register_fake
def soft_sw_affine_backward_full_fake(
    scores, grad_alignment, gap_open, gap_ext, temperature, lengths
):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_sw_affine", mutates_args=())
def soft_sw_affine(
    scores: torch.Tensor,
    gap_open: torch.Tensor,
    gap_ext: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_sw_affine(
        scores, gap_open, gap_ext, temperature, lengths
    )
    return score, alignment


@soft_sw_affine.register_fake
def soft_sw_affine_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_sw_affine_setup_context(ctx, inputs, output):
    scores, gap_open, gap_ext, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap_open = _extract_scalar(gap_open)
    ctx.gap_ext = _extract_scalar(gap_ext)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_sw_affine_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_open = ctx.needs_input_grad[1]
    needs_ext = ctx.needs_input_grad[2]
    needs_temp = ctx.needs_input_grad[3]
    grad_open = None
    grad_ext = None
    grad_temp = None

    if grad_score is not None and (needs_open or needs_ext or needs_temp):
        _, _, grad_open_fwd, grad_ext_fwd, grad_temp_fwd = soft_sw_affine_with_grads(
            scores, ctx.gap_open, ctx.gap_ext, ctx.temperature, ctx.lengths
        )
        if needs_open:
            grad_open = _sum_batch_grad(grad_score, grad_open_fwd)
        if needs_ext:
            grad_ext = _sum_batch_grad(grad_score, grad_ext_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        (
            grad_scores_align,
            grad_open_align,
            grad_ext_align,
            grad_temp_align,
        ) = soft_sw_affine_backward_full(
            scores,
            grad_alignment,
            ctx.gap_open,
            ctx.gap_ext,
            ctx.temperature,
            ctx.lengths,
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_open:
            grad_open = _add_grad(grad_open, grad_open_align)
        if needs_ext:
            grad_ext = _add_grad(grad_ext, grad_ext_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_open, grad_ext, grad_temp, None


soft_sw_affine.register_autograd(
    _soft_sw_affine_backward, setup_context=_soft_sw_affine_setup_context
)


@custom_op("d2p_py::soft_sw_affine_float", mutates_args=())
def soft_sw_affine_float(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_sw_affine_float(
        scores, gap_open, gap_ext, temperature, lengths
    )
    return score, alignment


@soft_sw_affine_float.register_fake
def soft_sw_affine_float_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_sw_affine_float_setup_context(ctx, inputs, output):
    scores, gap_open, gap_ext, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap_open = gap_open
    ctx.gap_ext = gap_ext
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_sw_affine_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _, _, _ = soft_sw_affine_backward_full(
            scores,
            grad_alignment,
            ctx.gap_open,
            ctx.gap_ext,
            ctx.temperature,
            ctx.lengths,
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None, None, None


soft_sw_affine_float.register_autograd(
    _soft_sw_affine_float_backward, setup_context=_soft_sw_affine_float_setup_context
)


# =============================================================================
# Needleman-Wunsch (Linear Gap)
# =============================================================================


@custom_op("d2p_py::soft_nw_with_grads", mutates_args=())
def soft_nw_with_grads(
    scores: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, alignment, grad_gap, grad_temp = torch.ops.d2p.soft_nw_with_grads(
        scores, gap, temperature, lengths
    )
    return score, alignment, grad_gap, grad_temp


@soft_nw_with_grads.register_fake
def soft_nw_with_grads_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_nw_hvp", mutates_args=())
def soft_nw_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_nw_hvp(scores, tangent, gap, temperature, lengths)


@soft_nw_hvp.register_fake
def soft_nw_hvp_fake(scores, tangent, gap, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_nw_param_jacobian", mutates_args=())
def soft_nw_param_jacobian(
    scores: torch.Tensor,
    param_type: int,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_nw_param_jacobian(scores, param_type, gap, temperature, lengths)


@soft_nw_param_jacobian.register_fake
def soft_nw_param_jacobian_fake(scores, param_type, gap, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_nw_backward_full", mutates_args=())
def soft_nw_backward_full(
    scores: torch.Tensor,
    grad_alignment: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_nw_backward_full(scores, grad_alignment, gap, temperature, lengths)


@soft_nw_backward_full.register_fake
def soft_nw_backward_full_fake(scores, grad_alignment, gap, temperature, lengths):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_nw", mutates_args=())
def soft_nw(
    scores: torch.Tensor,
    gap: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_nw(scores, gap, temperature, lengths)
    return score, alignment


@soft_nw.register_fake
def soft_nw_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_nw_setup_context(ctx, inputs, output):
    scores, gap, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap = _extract_scalar(gap)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_nw_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_gap = ctx.needs_input_grad[1]
    needs_temp = ctx.needs_input_grad[2]
    grad_gap = None
    grad_temp = None

    if grad_score is not None and (needs_gap or needs_temp):
        _, _, grad_gap_fwd, grad_temp_fwd = soft_nw_with_grads(
            scores, ctx.gap, ctx.temperature, ctx.lengths
        )
        if needs_gap:
            grad_gap = _sum_batch_grad(grad_score, grad_gap_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_scores_align, grad_gap_align, grad_temp_align = soft_nw_backward_full(
            scores, grad_alignment, ctx.gap, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_gap:
            grad_gap = _add_grad(grad_gap, grad_gap_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_gap, grad_temp, None


soft_nw.register_autograd(_soft_nw_backward, setup_context=_soft_nw_setup_context)


@custom_op("d2p_py::soft_nw_float", mutates_args=())
def soft_nw_float(
    scores: torch.Tensor,
    gap: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_nw_float(scores, gap, temperature, lengths)
    return score, alignment


@soft_nw_float.register_fake
def soft_nw_float_fake(scores, gap, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_nw_float_setup_context(ctx, inputs, output):
    scores, gap, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap = gap
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_nw_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _, _ = soft_nw_backward_full(
            scores, grad_alignment, ctx.gap, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None, None


soft_nw_float.register_autograd(
    _soft_nw_float_backward, setup_context=_soft_nw_float_setup_context
)


# =============================================================================
# Needleman-Wunsch (Affine Gap)
# =============================================================================


@custom_op("d2p_py::soft_nw_affine_with_grads", mutates_args=())
def soft_nw_affine_with_grads(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, alignment, grad_open, grad_ext, grad_temp = (
        torch.ops.d2p.soft_nw_affine_with_grads(
            scores, gap_open, gap_ext, temperature, lengths
        )
    )
    return score, alignment, grad_open, grad_ext, grad_temp


@soft_nw_affine_with_grads.register_fake
def soft_nw_affine_with_grads_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_nw_affine_hvp", mutates_args=())
def soft_nw_affine_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_nw_affine_hvp(
        scores, tangent, gap_open, gap_ext, temperature, lengths
    )


@soft_nw_affine_hvp.register_fake
def soft_nw_affine_hvp_fake(scores, tangent, gap_open, gap_ext, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_nw_affine_param_jacobian", mutates_args=())
def soft_nw_affine_param_jacobian(
    scores: torch.Tensor,
    param_type: int,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_nw_affine_param_jacobian(
        scores, param_type, gap_open, gap_ext, temperature, lengths
    )


@soft_nw_affine_param_jacobian.register_fake
def soft_nw_affine_param_jacobian_fake(
    scores, param_type, gap_open, gap_ext, temperature, lengths
):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_nw_affine_backward_full", mutates_args=())
def soft_nw_affine_backward_full(
    scores: torch.Tensor,
    grad_alignment: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_nw_affine_backward_full(
        scores, grad_alignment, gap_open, gap_ext, temperature, lengths
    )


@soft_nw_affine_backward_full.register_fake
def soft_nw_affine_backward_full_fake(
    scores, grad_alignment, gap_open, gap_ext, temperature, lengths
):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_nw_affine", mutates_args=())
def soft_nw_affine(
    scores: torch.Tensor,
    gap_open: torch.Tensor,
    gap_ext: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_nw_affine(
        scores, gap_open, gap_ext, temperature, lengths
    )
    return score, alignment


@soft_nw_affine.register_fake
def soft_nw_affine_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_nw_affine_setup_context(ctx, inputs, output):
    scores, gap_open, gap_ext, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap_open = _extract_scalar(gap_open)
    ctx.gap_ext = _extract_scalar(gap_ext)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_nw_affine_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_open = ctx.needs_input_grad[1]
    needs_ext = ctx.needs_input_grad[2]
    needs_temp = ctx.needs_input_grad[3]
    grad_open = None
    grad_ext = None
    grad_temp = None

    if grad_score is not None and (needs_open or needs_ext or needs_temp):
        _, _, grad_open_fwd, grad_ext_fwd, grad_temp_fwd = soft_nw_affine_with_grads(
            scores, ctx.gap_open, ctx.gap_ext, ctx.temperature, ctx.lengths
        )
        if needs_open:
            grad_open = _sum_batch_grad(grad_score, grad_open_fwd)
        if needs_ext:
            grad_ext = _sum_batch_grad(grad_score, grad_ext_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        (
            grad_scores_align,
            grad_open_align,
            grad_ext_align,
            grad_temp_align,
        ) = soft_nw_affine_backward_full(
            scores,
            grad_alignment,
            ctx.gap_open,
            ctx.gap_ext,
            ctx.temperature,
            ctx.lengths,
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_open:
            grad_open = _add_grad(grad_open, grad_open_align)
        if needs_ext:
            grad_ext = _add_grad(grad_ext, grad_ext_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_open, grad_ext, grad_temp, None


soft_nw_affine.register_autograd(
    _soft_nw_affine_backward, setup_context=_soft_nw_affine_setup_context
)


@custom_op("d2p_py::soft_nw_affine_float", mutates_args=())
def soft_nw_affine_float(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_nw_affine_float(
        scores, gap_open, gap_ext, temperature, lengths
    )
    return score, alignment


@soft_nw_affine_float.register_fake
def soft_nw_affine_float_fake(scores, gap_open, gap_ext, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_nw_affine_float_setup_context(ctx, inputs, output):
    scores, gap_open, gap_ext, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.gap_open = gap_open
    ctx.gap_ext = gap_ext
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_nw_affine_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _, _, _ = soft_nw_affine_backward_full(
            scores,
            grad_alignment,
            ctx.gap_open,
            ctx.gap_ext,
            ctx.temperature,
            ctx.lengths,
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None, None, None


soft_nw_affine_float.register_autograd(
    _soft_nw_affine_float_backward, setup_context=_soft_nw_affine_float_setup_context
)


# =============================================================================
# Dynamic Time Warping (DTW)
# =============================================================================


@custom_op("d2p_py::soft_dtw_with_grads", mutates_args=())
def soft_dtw_with_grads(
    costs: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
    bandwidth: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cost, alignment, grad_temp = torch.ops.d2p.soft_dtw_with_grads(
        costs, temperature, lengths, bandwidth
    )
    return cost, alignment, grad_temp


@soft_dtw_with_grads.register_fake
def soft_dtw_with_grads_fake(costs, temperature, lengths, bandwidth):
    B, L1, L2 = costs.shape
    return (
        costs.new_empty([B]),
        costs.new_empty([B, L1, L2]),
        costs.new_empty([B]),
    )


@custom_op("d2p_py::soft_dtw_hvp", mutates_args=())
def soft_dtw_hvp(
    costs: torch.Tensor,
    tangent: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
    bandwidth: Optional[int],
) -> torch.Tensor:
    return torch.ops.d2p.soft_dtw_hvp(costs, tangent, temperature, lengths, bandwidth)


@soft_dtw_hvp.register_fake
def soft_dtw_hvp_fake(costs, tangent, temperature, lengths, bandwidth):
    return costs.new_empty(costs.shape)


@custom_op("d2p_py::soft_dtw_param_jacobian", mutates_args=())
def soft_dtw_param_jacobian(
    costs: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
    bandwidth: Optional[int],
) -> torch.Tensor:
    return torch.ops.d2p.soft_dtw_param_jacobian(costs, temperature, lengths, bandwidth)


@soft_dtw_param_jacobian.register_fake
def soft_dtw_param_jacobian_fake(costs, temperature, lengths, bandwidth):
    return costs.new_empty(costs.shape)


@custom_op("d2p_py::soft_dtw_backward_full", mutates_args=())
def soft_dtw_backward_full(
    costs: torch.Tensor,
    grad_alignment: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
    bandwidth: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_dtw_backward_full(
        costs, grad_alignment, temperature, lengths, bandwidth
    )


@soft_dtw_backward_full.register_fake
def soft_dtw_backward_full_fake(costs, grad_alignment, temperature, lengths, bandwidth):
    return (
        costs.new_empty(costs.shape),
        costs.new_empty([1]),
    )


@custom_op("d2p_py::soft_dtw", mutates_args=())
def soft_dtw(
    costs: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
    bandwidth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cost, alignment = torch.ops.d2p.soft_dtw(costs, temperature, lengths, bandwidth)
    return cost, alignment


@soft_dtw.register_fake
def soft_dtw_fake(costs, temperature, lengths, bandwidth):
    B, L1, L2 = costs.shape
    return (costs.new_empty([B]), costs.new_empty([B, L1, L2]))


def _soft_dtw_setup_context(ctx, inputs, output):
    costs, temperature, lengths, bandwidth = inputs
    cost, alignment = output
    ctx.save_for_backward(costs, alignment)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths
    ctx.bandwidth = bandwidth


def _soft_dtw_backward(ctx, grad_score, grad_alignment):
    costs, alignment = ctx.saved_tensors
    grad_costs = _expand_batch_grad(grad_score, alignment)

    needs_temp = ctx.needs_input_grad[1]
    grad_temp = None

    if grad_score is not None and needs_temp:
        _, _, grad_temp_fwd = soft_dtw_with_grads(
            costs, ctx.temperature, ctx.lengths, ctx.bandwidth
        )
        grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_costs_align, grad_temp_align = soft_dtw_backward_full(
            costs, grad_alignment, ctx.temperature, ctx.lengths, ctx.bandwidth
        )
        grad_costs = _add_grad(grad_costs, grad_costs_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_costs, grad_temp, None, None


soft_dtw.register_autograd(_soft_dtw_backward, setup_context=_soft_dtw_setup_context)


@custom_op("d2p_py::soft_dtw_float", mutates_args=())
def soft_dtw_float(
    costs: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
    bandwidth: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    cost, alignment = torch.ops.d2p.soft_dtw_float(costs, temperature, lengths, bandwidth)
    return cost, alignment


@soft_dtw_float.register_fake
def soft_dtw_float_fake(costs, temperature, lengths, bandwidth):
    B, L1, L2 = costs.shape
    return (costs.new_empty([B]), costs.new_empty([B, L1, L2]))


def _soft_dtw_float_setup_context(ctx, inputs, output):
    costs, temperature, lengths, bandwidth = inputs
    cost, alignment = output
    ctx.save_for_backward(costs, alignment)
    ctx.temperature = temperature
    ctx.lengths = lengths
    ctx.bandwidth = bandwidth


def _soft_dtw_float_backward(ctx, grad_score, grad_alignment):
    costs, alignment = ctx.saved_tensors
    grad_costs = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_costs_align, _ = soft_dtw_backward_full(
            costs, grad_alignment, ctx.temperature, ctx.lengths, ctx.bandwidth
        )
        grad_costs = _add_grad(grad_costs, grad_costs_align)

    return grad_costs, None, None, None


soft_dtw_float.register_autograd(
    _soft_dtw_float_backward, setup_context=_soft_dtw_float_setup_context
)


# =============================================================================
# CKY Parsing
# =============================================================================


@custom_op("d2p_py::soft_cky_with_grads", mutates_args=())
def soft_cky_with_grads(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    score, merge_marginals, leaf_marginals, grad_temp = torch.ops.d2p.soft_cky_with_grads(
        merge_scores, leaf_scores, temperature
    )
    return score, merge_marginals, leaf_marginals, grad_temp


@soft_cky_with_grads.register_fake
def soft_cky_with_grads_fake(merge_scores, leaf_scores, temperature):
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
        leaf_scores.new_empty([B, N]),
        merge_scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_cky_hvp", mutates_args=())
def soft_cky_hvp(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    v_merge: torch.Tensor,
    v_leaf: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return torch.ops.d2p.soft_cky_hvp(merge_scores, leaf_scores, v_merge, v_leaf, temperature)


@soft_cky_hvp.register_fake
def soft_cky_hvp_fake(merge_scores, leaf_scores, v_merge, v_leaf, temperature):
    return merge_scores.new_empty(merge_scores.shape)


@custom_op("d2p_py::soft_cky_param_jacobian", mutates_args=())
def soft_cky_param_jacobian(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return torch.ops.d2p.soft_cky_param_jacobian(merge_scores, leaf_scores, temperature)


@soft_cky_param_jacobian.register_fake
def soft_cky_param_jacobian_fake(merge_scores, leaf_scores, temperature):
    return merge_scores.new_empty(merge_scores.shape)


@custom_op("d2p_py::soft_cky_backward_full", mutates_args=())
def soft_cky_backward_full(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    grad_posteriors: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_cky_backward_full(
        merge_scores, leaf_scores, grad_posteriors.contiguous(), temperature
    )


@soft_cky_backward_full.register_fake
def soft_cky_backward_full_fake(
    merge_scores, leaf_scores, grad_posteriors, temperature
):
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B, N, N, N]),
        leaf_scores.new_empty([B, N]),
        merge_scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_cky", mutates_args=())
def soft_cky(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, marginals = torch.ops.d2p.soft_cky(merge_scores, leaf_scores, temperature)
    return score, marginals


@soft_cky.register_fake
def soft_cky_fake(merge_scores, leaf_scores, temperature):
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
    )


def _soft_cky_setup_context(ctx, inputs, output):
    merge_scores, leaf_scores, temperature = inputs
    score, marginals = output
    ctx.save_for_backward(merge_scores, leaf_scores, marginals, temperature)
    ctx.temperature = _extract_scalar(temperature)
    ctx.temperature_is_local = isinstance(temperature, torch.Tensor) and temperature.numel() > 1


def _soft_cky_backward(ctx, grad_score, grad_marginals):
    merge_scores, leaf_scores, marginals, temperature = ctx.saved_tensors
    grad_merge = _expand_batch_grad(grad_score, marginals)
    grad_leaf = None

    needs_temp = ctx.needs_input_grad[2]
    grad_temp = None

    if grad_score is not None:
        if ctx.temperature_is_local:
            merge_scores_local = merge_scores.detach().requires_grad_(True)
            leaf_scores_local = leaf_scores.detach().requires_grad_(True)
            temperature_local = temperature.detach().requires_grad_(needs_temp)
            score_local, _ = torch.ops.d2p.soft_cky(
                merge_scores_local, leaf_scores_local, temperature_local
            )
            grad_inputs = [merge_scores_local, leaf_scores_local]
            if needs_temp:
                grad_inputs.append(temperature_local)
            grads = torch.autograd.grad(
                score_local,
                grad_inputs,
                grad_outputs=grad_score,
            )
            grad_merge = grads[0]
            grad_leaf = grads[1]
            if needs_temp:
                grad_temp = grads[2]
        else:
            _, merge_marginals, leaf_marginals, grad_temp_fwd = soft_cky_with_grads(
                merge_scores, leaf_scores, ctx.temperature
            )
            grad_merge = _expand_batch_grad(grad_score, merge_marginals)
            grad_leaf = _expand_batch_grad(grad_score, leaf_marginals)
            if needs_temp:
                grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_marginals is not None:
        if ctx.temperature_is_local:
            raise RuntimeError(
                "soft_cky backward through marginals is only supported for scalar temperature"
            )
        grad_merge_align, grad_leaf_align, grad_temp_align = soft_cky_backward_full(
            merge_scores, leaf_scores, grad_marginals, ctx.temperature
        )
        grad_merge = _add_grad(grad_merge, grad_merge_align)
        grad_leaf = _add_grad(grad_leaf, grad_leaf_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_merge, grad_leaf, grad_temp


soft_cky.register_autograd(_soft_cky_backward, setup_context=_soft_cky_setup_context)


@custom_op("d2p_py::soft_cky_float", mutates_args=())
def soft_cky_float(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, marginals = torch.ops.d2p.soft_cky_float(merge_scores, leaf_scores, temperature)
    return score, marginals


@soft_cky_float.register_fake
def soft_cky_float_fake(merge_scores, leaf_scores, temperature):
    B = merge_scores.size(0)
    N = merge_scores.size(1)
    return (
        merge_scores.new_empty([B]),
        merge_scores.new_empty([B, N, N, N]),
    )


def _soft_cky_float_setup_context(ctx, inputs, output):
    merge_scores, leaf_scores, temperature = inputs
    score, marginals = output
    ctx.save_for_backward(merge_scores, leaf_scores, marginals)
    ctx.temperature = temperature


def _soft_cky_float_backward(ctx, grad_score, grad_marginals):
    merge_scores, leaf_scores, marginals = ctx.saved_tensors
    grad_merge = _expand_batch_grad(grad_score, marginals)
    grad_leaf = None

    if grad_score is not None:
        _, merge_marginals, leaf_marginals, _ = soft_cky_with_grads(
            merge_scores, leaf_scores, ctx.temperature
        )
        grad_merge = _expand_batch_grad(grad_score, merge_marginals)
        grad_leaf = _expand_batch_grad(grad_score, leaf_marginals)

    if grad_marginals is not None:
        grad_merge_align, grad_leaf_align, _ = soft_cky_backward_full(
            merge_scores, leaf_scores, grad_marginals, ctx.temperature
        )
        grad_merge = _add_grad(grad_merge, grad_merge_align)
        grad_leaf = _add_grad(grad_leaf, grad_leaf_align)

    return grad_merge, grad_leaf, None


soft_cky_float.register_autograd(
    _soft_cky_float_backward, setup_context=_soft_cky_float_setup_context
)


# =============================================================================
# Monotonic Alignment Search (MAS)
# =============================================================================


@custom_op("d2p_py::soft_mas_with_grads", mutates_args=())
def soft_mas_with_grads(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_mas_with_grads(scores, temperature, lengths)
    return score, alignment


@soft_mas_with_grads.register_fake
def soft_mas_with_grads_fake(scores, temperature, lengths):
    B, T, S = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, T, S]),
    )


@custom_op("d2p_py::soft_mas_hvp", mutates_args=())
def soft_mas_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_mas_hvp(scores, tangent, temperature, lengths)


@soft_mas_hvp.register_fake
def soft_mas_hvp_fake(scores, tangent, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_mas_param_jacobian", mutates_args=())
def soft_mas_param_jacobian(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_mas_param_jacobian(scores, temperature, lengths)


@soft_mas_param_jacobian.register_fake
def soft_mas_param_jacobian_fake(scores, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_mas_backward_full", mutates_args=())
def soft_mas_backward_full(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_mas_backward_full(scores, temperature, lengths)


@soft_mas_backward_full.register_fake
def soft_mas_backward_full_fake(scores, temperature, lengths):
    B, T, S = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, T, S]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_mas", mutates_args=())
def soft_mas(
    scores: torch.Tensor,
    temperature: torch.Tensor,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    temp_val = _extract_scalar(temperature)
    score, alignment = torch.ops.d2p.soft_mas_with_grads(scores, temp_val, lengths)
    return score, alignment


@soft_mas.register_fake
def soft_mas_fake(scores, temperature, lengths):
    B, T, S = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, T, S]),
    )


def _soft_mas_setup_context(ctx, inputs, output):
    scores, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_mas_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_temp = ctx.needs_input_grad[1]
    grad_temp = None

    if grad_score is not None and needs_temp:
        _, _, grad_temp_fwd = soft_mas_backward_full(
            scores, ctx.temperature, ctx.lengths
        )
        grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_scores_align = soft_mas_hvp(
            scores, grad_alignment, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_temp:
            dP_dT = soft_mas_param_jacobian(scores, ctx.temperature, ctx.lengths)
            grad_temp_align = _sum_alignment_grad(grad_alignment, dP_dT)
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_temp, None


soft_mas.register_autograd(_soft_mas_backward, setup_context=_soft_mas_setup_context)


@custom_op("d2p_py::soft_mas_float", mutates_args=())
def soft_mas_float(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_mas_float(scores, temperature, lengths)


@soft_mas_float.register_fake
def soft_mas_float_fake(scores, temperature, lengths):
    B = scores.size(0)
    return scores.new_empty([B])


def _soft_mas_float_setup_context(ctx, inputs, output):
    scores, temperature, lengths = inputs
    ctx.save_for_backward(scores)
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_mas_float_backward(ctx, grad_score):
    (scores,) = ctx.saved_tensors
    if grad_score is None:
        return None, None, None
    _, posteriors, _ = soft_mas_backward_full(scores, ctx.temperature, ctx.lengths)
    grad_scores = _expand_batch_grad(grad_score, posteriors)
    return grad_scores, None, None


soft_mas_float.register_autograd(
    _soft_mas_float_backward, setup_context=_soft_mas_float_setup_context
)


# =============================================================================
# Eisner Dependency Parsing
# =============================================================================


@custom_op("d2p_py::soft_eisner_with_grads", mutates_args=())
def soft_eisner_with_grads(
    arc_scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, marginals = torch.ops.d2p.soft_eisner_with_grads(
        arc_scores, temperature, lengths
    )
    return score, marginals


@soft_eisner_with_grads.register_fake
def soft_eisner_with_grads_fake(arc_scores, temperature, lengths):
    B, N, _ = arc_scores.shape
    return (
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
    )


@custom_op("d2p_py::soft_eisner_hvp", mutates_args=())
def soft_eisner_hvp(
    arc_scores: torch.Tensor,
    tangent: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_eisner_hvp(arc_scores, tangent.contiguous(), temperature, lengths)


@soft_eisner_hvp.register_fake
def soft_eisner_hvp_fake(arc_scores, tangent, temperature, lengths):
    return arc_scores.new_empty(arc_scores.shape)


@custom_op("d2p_py::soft_eisner_param_jacobian", mutates_args=())
def soft_eisner_param_jacobian(
    arc_scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_eisner_param_jacobian(arc_scores, temperature, lengths)


@soft_eisner_param_jacobian.register_fake
def soft_eisner_param_jacobian_fake(arc_scores, temperature, lengths):
    return arc_scores.new_empty(arc_scores.shape)


@custom_op("d2p_py::soft_eisner_backward_full", mutates_args=())
def soft_eisner_backward_full(
    arc_scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_eisner_backward_full(arc_scores, temperature, lengths)


@soft_eisner_backward_full.register_fake
def soft_eisner_backward_full_fake(arc_scores, temperature, lengths):
    B, N, _ = arc_scores.shape
    return (
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
        arc_scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_eisner", mutates_args=())
def soft_eisner(
    arc_scores: torch.Tensor,
    temperature: torch.Tensor,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, marginals = torch.ops.d2p.soft_eisner(arc_scores, temperature, lengths)
    return score, marginals


@soft_eisner.register_fake
def soft_eisner_fake(arc_scores, temperature, lengths):
    B, N, _ = arc_scores.shape
    return (
        arc_scores.new_empty([B]),
        arc_scores.new_empty([B, N, N]),
    )


def _soft_eisner_setup_context(ctx, inputs, output):
    arc_scores, temperature, lengths = inputs
    score, marginals = output
    ctx.save_for_backward(arc_scores, marginals)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_eisner_backward(ctx, grad_score, grad_marginals):
    arc_scores, marginals = ctx.saved_tensors
    grad_arc = _expand_batch_grad(grad_score, marginals)

    needs_temp = ctx.needs_input_grad[1]
    grad_temp = None

    if grad_score is not None and needs_temp:
        _, _, grad_temp_fwd = soft_eisner_backward_full(
            arc_scores, ctx.temperature, ctx.lengths
        )
        grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_marginals is not None:
        grad_arc_align = soft_eisner_hvp(
            arc_scores, grad_marginals, ctx.temperature, ctx.lengths
        )
        grad_arc = _add_grad(grad_arc, grad_arc_align)

    return grad_arc, grad_temp, None


soft_eisner.register_autograd(_soft_eisner_backward, setup_context=_soft_eisner_setup_context)


@custom_op("d2p_py::soft_eisner_float", mutates_args=())
def soft_eisner_float(
    arc_scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_eisner_float(arc_scores, temperature, lengths)


@soft_eisner_float.register_fake
def soft_eisner_float_fake(arc_scores, temperature, lengths):
    B = arc_scores.size(0)
    return arc_scores.new_empty([B])


def _soft_eisner_float_setup_context(ctx, inputs, output):
    arc_scores, temperature, lengths = inputs
    ctx.save_for_backward(arc_scores)
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_eisner_float_backward(ctx, grad_score):
    (arc_scores,) = ctx.saved_tensors
    if grad_score is None:
        return None, None, None
    _, marginals = soft_eisner_with_grads(arc_scores, ctx.temperature, ctx.lengths)
    grad_arc = _expand_batch_grad(grad_score, marginals)
    return grad_arc, None, None


soft_eisner_float.register_autograd(
    _soft_eisner_float_backward, setup_context=_soft_eisner_float_setup_context
)


# =============================================================================
# Edit Distances (Levenshtein, LCS, OSA, Damerau)
# =============================================================================


@custom_op("d2p_py::soft_levenshtein_with_grads", mutates_args=())
def soft_levenshtein_with_grads(
    scores: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    distance, alignment, grad_ins, grad_del, grad_temp = (
        torch.ops.d2p.soft_levenshtein_with_grads(
            scores, ins_cost, del_cost, temperature, lengths
        )
    )
    return distance, alignment, grad_ins, grad_del, grad_temp


@soft_levenshtein_with_grads.register_fake
def soft_levenshtein_with_grads_fake(scores, ins_cost, del_cost, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
        scores.new_empty([B]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_levenshtein_hvp", mutates_args=())
def soft_levenshtein_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_levenshtein_hvp(
        scores, tangent, ins_cost, del_cost, temperature, lengths
    )


@soft_levenshtein_hvp.register_fake
def soft_levenshtein_hvp_fake(scores, tangent, ins_cost, del_cost, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_levenshtein_param_jacobian", mutates_args=())
def soft_levenshtein_param_jacobian(
    scores: torch.Tensor,
    param_type: int,
    ins_cost: float,
    del_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_levenshtein_param_jacobian(
        scores, param_type, ins_cost, del_cost, temperature, lengths
    )


@soft_levenshtein_param_jacobian.register_fake
def soft_levenshtein_param_jacobian_fake(
    scores, param_type, ins_cost, del_cost, temperature, lengths
):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_levenshtein_backward_full", mutates_args=())
def soft_levenshtein_backward_full(
    scores: torch.Tensor,
    grad_posteriors: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_levenshtein_backward_full(
        scores, grad_posteriors, ins_cost, del_cost, temperature, lengths
    )


@soft_levenshtein_backward_full.register_fake
def soft_levenshtein_backward_full_fake(
    scores, grad_posteriors, ins_cost, del_cost, temperature, lengths
):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
        scores.new_empty([1]),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_levenshtein", mutates_args=())
def soft_levenshtein(
    scores: torch.Tensor,
    ins_cost: torch.Tensor,
    del_cost: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_levenshtein(
        scores, ins_cost, del_cost, temperature, lengths
    )
    return distance, alignment


@soft_levenshtein.register_fake
def soft_levenshtein_fake(scores, ins_cost, del_cost, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_levenshtein_setup_context(ctx, inputs, output):
    scores, ins_cost, del_cost, temperature, lengths = inputs
    distance, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.ins_cost = _extract_scalar(ins_cost)
    ctx.del_cost = _extract_scalar(del_cost)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_levenshtein_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_ins = ctx.needs_input_grad[1]
    needs_del = ctx.needs_input_grad[2]
    needs_temp = ctx.needs_input_grad[3]
    grad_ins = None
    grad_del = None
    grad_temp = None

    if grad_score is not None and (needs_ins or needs_del or needs_temp):
        _, _, grad_ins_fwd, grad_del_fwd, grad_temp_fwd = soft_levenshtein_with_grads(
            scores, ctx.ins_cost, ctx.del_cost, ctx.temperature, ctx.lengths
        )
        if needs_ins:
            grad_ins = _sum_batch_grad(grad_score, grad_ins_fwd)
        if needs_del:
            grad_del = _sum_batch_grad(grad_score, grad_del_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_scores_align, grad_ins_align, grad_del_align, grad_temp_align = (
            soft_levenshtein_backward_full(
                scores,
                grad_alignment,
                ctx.ins_cost,
                ctx.del_cost,
                ctx.temperature,
                ctx.lengths,
            )
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_ins:
            grad_ins = _add_grad(grad_ins, grad_ins_align)
        if needs_del:
            grad_del = _add_grad(grad_del, grad_del_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_ins, grad_del, grad_temp, None


soft_levenshtein.register_autograd(
    _soft_levenshtein_backward, setup_context=_soft_levenshtein_setup_context
)


@custom_op("d2p_py::soft_levenshtein_float", mutates_args=())
def soft_levenshtein_float(
    scores: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_levenshtein_float(
        scores, ins_cost, del_cost, temperature, lengths
    )
    return distance, alignment


@soft_levenshtein_float.register_fake
def soft_levenshtein_float_fake(scores, ins_cost, del_cost, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_levenshtein_float_setup_context(ctx, inputs, output):
    scores, ins_cost, del_cost, temperature, lengths = inputs
    distance, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.ins_cost = ins_cost
    ctx.del_cost = del_cost
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_levenshtein_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _, _, _ = soft_levenshtein_backward_full(
            scores,
            grad_alignment,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.temperature,
            ctx.lengths,
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None, None, None


soft_levenshtein_float.register_autograd(
    _soft_levenshtein_float_backward, setup_context=_soft_levenshtein_float_setup_context
)


@custom_op("d2p_py::soft_lcs_with_grads", mutates_args=())
def soft_lcs_with_grads(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score, alignment, grad_temp = torch.ops.d2p.soft_lcs_with_grads(
        scores, temperature, lengths
    )
    return score, alignment, grad_temp


@soft_lcs_with_grads.register_fake
def soft_lcs_with_grads_fake(scores, temperature, lengths):
    B, L1, L2 = scores.shape
    return (
        scores.new_empty([B]),
        scores.new_empty([B, L1, L2]),
        scores.new_empty([B]),
    )


@custom_op("d2p_py::soft_lcs_hvp", mutates_args=())
def soft_lcs_hvp(
    scores: torch.Tensor,
    tangent: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_lcs_hvp(scores, tangent, temperature, lengths)


@soft_lcs_hvp.register_fake
def soft_lcs_hvp_fake(scores, tangent, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_lcs_param_jacobian", mutates_args=())
def soft_lcs_param_jacobian(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_lcs_param_jacobian(scores, temperature, lengths)


@soft_lcs_param_jacobian.register_fake
def soft_lcs_param_jacobian_fake(scores, temperature, lengths):
    return scores.new_empty(scores.shape)


@custom_op("d2p_py::soft_lcs_backward_full", mutates_args=())
def soft_lcs_backward_full(
    scores: torch.Tensor,
    grad_output: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.d2p.soft_lcs_backward_full(scores, grad_output, temperature, lengths)


@soft_lcs_backward_full.register_fake
def soft_lcs_backward_full_fake(scores, grad_output, temperature, lengths):
    return (
        scores.new_empty(scores.shape),
        scores.new_empty([1]),
    )


@custom_op("d2p_py::soft_lcs", mutates_args=())
def soft_lcs(
    scores: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_lcs(scores, temperature, lengths)
    return score, alignment


@soft_lcs.register_fake
def soft_lcs_fake(scores, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_lcs_setup_context(ctx, inputs, output):
    scores, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_lcs_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    needs_temp = ctx.needs_input_grad[1]
    grad_temp = None

    if grad_score is not None and needs_temp:
        _, _, grad_temp_fwd = soft_lcs_with_grads(
            scores, ctx.temperature, ctx.lengths
        )
        grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        grad_scores_align, grad_temp_align = soft_lcs_backward_full(
            scores, grad_alignment, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_scores, grad_temp, None


soft_lcs.register_autograd(_soft_lcs_backward, setup_context=_soft_lcs_setup_context)


@custom_op("d2p_py::soft_lcs_float", mutates_args=())
def soft_lcs_float(
    scores: torch.Tensor,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    score, alignment = torch.ops.d2p.soft_lcs_float(scores, temperature, lengths)
    return score, alignment


@soft_lcs_float.register_fake
def soft_lcs_float_fake(scores, temperature, lengths):
    B, L1, L2 = scores.shape
    return (scores.new_empty([B]), scores.new_empty([B, L1, L2]))


def _soft_lcs_float_setup_context(ctx, inputs, output):
    scores, temperature, lengths = inputs
    score, alignment = output
    ctx.save_for_backward(scores, alignment)
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_lcs_float_backward(ctx, grad_score, grad_alignment):
    scores, alignment = ctx.saved_tensors
    grad_scores = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_scores_align, _ = soft_lcs_backward_full(
            scores, grad_alignment, ctx.temperature, ctx.lengths
        )
        grad_scores = _add_grad(grad_scores, grad_scores_align)

    return grad_scores, None, None


soft_lcs_float.register_autograd(
    _soft_lcs_float_backward, setup_context=_soft_lcs_float_setup_context
)


@custom_op("d2p_py::soft_osa_with_grads", mutates_args=())
def soft_osa_with_grads(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    distance, alignment, grad_ins, grad_del, grad_trans, grad_temp = (
        torch.ops.d2p.soft_osa_with_grads(
            sub_costs,
            trans_mask,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
    )
    return distance, alignment, grad_ins, grad_del, grad_trans, grad_temp


@soft_osa_with_grads.register_fake
def soft_osa_with_grads_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B, L1, L2]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


@custom_op("d2p_py::soft_osa_hvp", mutates_args=())
def soft_osa_hvp(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    tangent: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_osa_hvp(
        sub_costs,
        trans_mask,
        tangent,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    )


@soft_osa_hvp.register_fake
def soft_osa_hvp_fake(
    sub_costs, trans_mask, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
):
    return sub_costs.new_empty(sub_costs.shape)


@custom_op("d2p_py::soft_osa_param_jacobian", mutates_args=())
def soft_osa_param_jacobian(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    param_type: int,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_osa_param_jacobian(
        sub_costs, trans_mask, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
    )


@soft_osa_param_jacobian.register_fake
def soft_osa_param_jacobian_fake(
    sub_costs, trans_mask, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
):
    return sub_costs.new_empty(sub_costs.shape)


@custom_op("d2p_py::soft_osa_backward_full", mutates_args=())
def soft_osa_backward_full(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    grad_output: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return torch.ops.d2p.soft_osa_backward_full(
        sub_costs,
        trans_mask,
        grad_output,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    )


@soft_osa_backward_full.register_fake
def soft_osa_backward_full_fake(
    sub_costs,
    trans_mask,
    grad_output,
    ins_cost,
    del_cost,
    trans_cost,
    temperature,
    lengths,
):
    return (
        sub_costs.new_empty(sub_costs.shape),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
    )


@custom_op("d2p_py::soft_osa", mutates_args=())
def soft_osa(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: torch.Tensor,
    del_cost: torch.Tensor,
    trans_cost: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_osa(
        sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
    )
    return distance, alignment


@soft_osa.register_fake
def soft_osa_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2]))


def _soft_osa_setup_context(ctx, inputs, output):
    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    ) = inputs
    distance, alignment = output
    ctx.save_for_backward(sub_costs, trans_mask, alignment)
    ctx.ins_cost = _extract_scalar(ins_cost)
    ctx.del_cost = _extract_scalar(del_cost)
    ctx.trans_cost = _extract_scalar(trans_cost)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_osa_backward(ctx, grad_score, grad_alignment):
    sub_costs, trans_mask, alignment = ctx.saved_tensors
    grad_sub = _expand_batch_grad(grad_score, alignment)

    needs_ins = ctx.needs_input_grad[2]
    needs_del = ctx.needs_input_grad[3]
    needs_trans = ctx.needs_input_grad[4]
    needs_temp = ctx.needs_input_grad[5]
    grad_ins = None
    grad_del = None
    grad_trans = None
    grad_temp = None

    if grad_score is not None and (needs_ins or needs_del or needs_trans or needs_temp):
        (
            _,
            _,
            grad_ins_fwd,
            grad_del_fwd,
            grad_trans_fwd,
            grad_temp_fwd,
        ) = soft_osa_with_grads(
            sub_costs,
            trans_mask,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        if needs_ins:
            grad_ins = _sum_batch_grad(grad_score, grad_ins_fwd)
        if needs_del:
            grad_del = _sum_batch_grad(grad_score, grad_del_fwd)
        if needs_trans:
            grad_trans = _sum_batch_grad(grad_score, grad_trans_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        (
            grad_sub_align,
            grad_ins_align,
            grad_del_align,
            grad_trans_align,
            grad_temp_align,
        ) = soft_osa_backward_full(
            sub_costs,
            trans_mask,
            grad_alignment,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        grad_sub = _add_grad(grad_sub, grad_sub_align)
        if needs_ins:
            grad_ins = _add_grad(grad_ins, grad_ins_align)
        if needs_del:
            grad_del = _add_grad(grad_del, grad_del_align)
        if needs_trans:
            grad_trans = _add_grad(grad_trans, grad_trans_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_sub, None, grad_ins, grad_del, grad_trans, grad_temp, None


soft_osa.register_autograd(_soft_osa_backward, setup_context=_soft_osa_setup_context)


@custom_op("d2p_py::soft_osa_float", mutates_args=())
def soft_osa_float(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_osa_float(
        sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
    )
    return distance, alignment


@soft_osa_float.register_fake
def soft_osa_float_fake(
    sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2]))


def _soft_osa_float_setup_context(ctx, inputs, output):
    (
        sub_costs,
        trans_mask,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    ) = inputs
    distance, alignment = output
    ctx.save_for_backward(sub_costs, trans_mask, alignment)
    ctx.ins_cost = ins_cost
    ctx.del_cost = del_cost
    ctx.trans_cost = trans_cost
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_osa_float_backward(ctx, grad_score, grad_alignment):
    sub_costs, trans_mask, alignment = ctx.saved_tensors
    grad_sub = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_sub_align, _, _, _, _ = soft_osa_backward_full(
            sub_costs,
            trans_mask,
            grad_alignment,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        grad_sub = _add_grad(grad_sub, grad_sub_align)

    return grad_sub, None, None, None, None, None, None


soft_osa_float.register_autograd(
    _soft_osa_float_backward, setup_context=_soft_osa_float_setup_context
)


@custom_op("d2p_py::soft_damerau_with_grads", mutates_args=())
def soft_damerau_with_grads(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    distance, alignment, grad_ins, grad_del, grad_trans, grad_temp = (
        torch.ops.d2p.soft_damerau_with_grads(
            sub_costs,
            trans_src,
            ins_cost,
            del_cost,
            trans_cost,
            temperature,
            lengths,
        )
    )
    return distance, alignment, grad_ins, grad_del, grad_trans, grad_temp


@soft_damerau_with_grads.register_fake
def soft_damerau_with_grads_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B, L1, L2]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
        sub_costs.new_empty([B]),
    )


@custom_op("d2p_py::soft_damerau_hvp", mutates_args=())
def soft_damerau_hvp(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    tangent: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_damerau_hvp(
        sub_costs,
        trans_src,
        tangent,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    )


@soft_damerau_hvp.register_fake
def soft_damerau_hvp_fake(
    sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
):
    return sub_costs.new_empty(sub_costs.shape)


@custom_op("d2p_py::soft_damerau_param_jacobian", mutates_args=())
def soft_damerau_param_jacobian(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    param_type: int,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.d2p.soft_damerau_param_jacobian(
        sub_costs, trans_src, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
    )


@soft_damerau_param_jacobian.register_fake
def soft_damerau_param_jacobian_fake(
    sub_costs, trans_src, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
):
    return sub_costs.new_empty(sub_costs.shape)


@custom_op("d2p_py::soft_damerau_backward_full", mutates_args=())
def soft_damerau_backward_full(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    grad_output: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    return torch.ops.d2p.soft_damerau_backward_full(
        sub_costs,
        trans_src,
        grad_output,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    )


@soft_damerau_backward_full.register_fake
def soft_damerau_backward_full_fake(
    sub_costs,
    trans_src,
    grad_output,
    ins_cost,
    del_cost,
    trans_cost,
    temperature,
    lengths,
):
    return (
        sub_costs.new_empty(sub_costs.shape),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
        sub_costs.new_empty([1]),
    )


@custom_op("d2p_py::soft_damerau", mutates_args=())
def soft_damerau(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: torch.Tensor,
    del_cost: torch.Tensor,
    trans_cost: torch.Tensor,
    temperature: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_damerau(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
    )
    return distance, alignment


@soft_damerau.register_fake
def soft_damerau_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2]))


def _soft_damerau_setup_context(ctx, inputs, output):
    (
        sub_costs,
        trans_src,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    ) = inputs
    distance, alignment = output
    ctx.save_for_backward(sub_costs, trans_src, alignment)
    ctx.ins_cost = _extract_scalar(ins_cost)
    ctx.del_cost = _extract_scalar(del_cost)
    ctx.trans_cost = _extract_scalar(trans_cost)
    ctx.temperature = _extract_scalar(temperature)
    ctx.lengths = lengths


def _soft_damerau_backward(ctx, grad_score, grad_alignment):
    sub_costs, trans_src, alignment = ctx.saved_tensors
    grad_sub = _expand_batch_grad(grad_score, alignment)

    needs_ins = ctx.needs_input_grad[2]
    needs_del = ctx.needs_input_grad[3]
    needs_trans = ctx.needs_input_grad[4]
    needs_temp = ctx.needs_input_grad[5]
    grad_ins = None
    grad_del = None
    grad_trans = None
    grad_temp = None

    if grad_score is not None and (needs_ins or needs_del or needs_trans or needs_temp):
        (
            _,
            _,
            grad_ins_fwd,
            grad_del_fwd,
            grad_trans_fwd,
            grad_temp_fwd,
        ) = soft_damerau_with_grads(
            sub_costs,
            trans_src,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        if needs_ins:
            grad_ins = _sum_batch_grad(grad_score, grad_ins_fwd)
        if needs_del:
            grad_del = _sum_batch_grad(grad_score, grad_del_fwd)
        if needs_trans:
            grad_trans = _sum_batch_grad(grad_score, grad_trans_fwd)
        if needs_temp:
            grad_temp = _sum_batch_grad(grad_score, grad_temp_fwd)

    if grad_alignment is not None:
        (
            grad_sub_align,
            grad_ins_align,
            grad_del_align,
            grad_trans_align,
            grad_temp_align,
        ) = soft_damerau_backward_full(
            sub_costs,
            trans_src,
            grad_alignment,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        grad_sub = _add_grad(grad_sub, grad_sub_align)
        if needs_ins:
            grad_ins = _add_grad(grad_ins, grad_ins_align)
        if needs_del:
            grad_del = _add_grad(grad_del, grad_del_align)
        if needs_trans:
            grad_trans = _add_grad(grad_trans, grad_trans_align)
        if needs_temp:
            grad_temp = _add_grad(grad_temp, grad_temp_align)

    return grad_sub, None, grad_ins, grad_del, grad_trans, grad_temp, None


soft_damerau.register_autograd(
    _soft_damerau_backward, setup_context=_soft_damerau_setup_context
)


@custom_op("d2p_py::soft_damerau_float", mutates_args=())
def soft_damerau_float(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float,
    lengths: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    distance, alignment = torch.ops.d2p.soft_damerau_float(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
    )
    return distance, alignment


@soft_damerau_float.register_fake
def soft_damerau_float_fake(
    sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
):
    B, L1, L2 = sub_costs.shape
    return (sub_costs.new_empty([B]), sub_costs.new_empty([B, L1, L2]))


def _soft_damerau_float_setup_context(ctx, inputs, output):
    (
        sub_costs,
        trans_src,
        ins_cost,
        del_cost,
        trans_cost,
        temperature,
        lengths,
    ) = inputs
    distance, alignment = output
    ctx.save_for_backward(sub_costs, trans_src, alignment)
    ctx.ins_cost = ins_cost
    ctx.del_cost = del_cost
    ctx.trans_cost = trans_cost
    ctx.temperature = temperature
    ctx.lengths = lengths


def _soft_damerau_float_backward(ctx, grad_score, grad_alignment):
    sub_costs, trans_src, alignment = ctx.saved_tensors
    grad_sub = _expand_batch_grad(grad_score, alignment)

    if grad_alignment is not None:
        grad_sub_align, _, _, _, _ = soft_damerau_backward_full(
            sub_costs,
            trans_src,
            grad_alignment,
            ctx.ins_cost,
            ctx.del_cost,
            ctx.trans_cost,
            ctx.temperature,
            ctx.lengths,
        )
        grad_sub = _add_grad(grad_sub, grad_sub_align)

    return grad_sub, None, None, None, None, None, None


soft_damerau_float.register_autograd(
    _soft_damerau_float_backward, setup_context=_soft_damerau_float_setup_context
)
