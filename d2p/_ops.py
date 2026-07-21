# SPDX-License-Identifier: Apache-2.0
"""
Low-level operator access for d2p.

This module provides direct access to the underlying C++/CUDA operators
registered with PyTorch's dispatcher. These are the raw functions without
any Python-side validation or wrapping.

For most users, the high-level API (d2p.soft_sw, d2p.soft_dtw, etc.) is
recommended. Use this module only when you need:
- Direct control over forward/backward passes
- Integration with custom autograd functions
- Performance-critical code that can skip Python validation

Usage:
    from d2p import ops

    # Direct operator access
    score, alignment = ops.soft_sw_float(scores, -1.0, 1.0, lengths)

    # Or access all d2p operators
    result = ops.d2p.soft_sw_float(scores, -1.0, 1.0, lengths)
"""

import os
import glob
import torch

from ._pt2_utils import use_pt2_ops

# Extension loading state
_extension_loaded = False
_cuda_dispatch_available = False


def _has_cuda_dispatch():
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key("d2p::soft_sw_float", "CUDA")
    except (AttributeError, RuntimeError):
        return False


def _load_extension():
    """Load the C++/CUDA extension library."""
    global _extension_loaded, _cuda_dispatch_available

    if _extension_loaded:
        return

    lib_dir = os.path.dirname(__file__)
    lib_pattern = os.path.join(lib_dir, '_C*.so')
    libs = sorted(glob.glob(lib_pattern))

    # For editable installs (meson-python), check the build directory
    if not libs:
        project_root = os.path.dirname(lib_dir)
        build_pattern = os.path.join(project_root, 'build', '*', '_C*.so')
        libs = sorted(glob.glob(build_pattern))

    if libs:
        try:
            torch.ops.load_library(libs[0])
        except OSError as exc:
            # The extension is ABI-locked to the exact torch it was compiled
            # against: libc10/libtorch symbol versions move every torch minor.
            # The common cause is `pip install d2p` building from source under
            # PEP 517 isolation, which installs its own (latest) torch into the
            # build environment and compiles against that instead of yours --
            # pip then reports success and this is the first sign anything is
            # wrong. Say so, rather than surfacing a bare undefined symbol.
            if "undefined symbol" in str(exc):
                raise ImportError(
                    f"d2p's compiled extension is not compatible with the "
                    f"installed torch {torch.__version__}.\n\n"
                    f"This usually means it was built against a different "
                    f"torch version. If you installed from source, PEP 517 "
                    f"build isolation pulls in its own torch, so build "
                    f"against yours instead:\n\n"
                    f"    pip install torch=={torch.__version__.split('+')[0]}\n"
                    f"    pip install meson-python meson ninja\n"
                    f"    pip install --no-build-isolation --no-cache-dir py-d2p\n\n"
                    f"(--no-cache-dir matters: pip keys its wheel cache on the "
                    f"sdist, not on the torch it was built against, so a "
                    f"previously built incompatible wheel is otherwise reused.)"
                    f"\n\nOriginal error: {exc}"
                ) from exc
            raise
        _extension_loaded = True
        _cuda_dispatch_available = _has_cuda_dispatch()
    else:
        raise ImportError(
            f"Could not find _C extension library in {lib_dir}. "
            "Run a source build with either "
            "`python -m pip install . --no-build-isolation` or "
            "`python -m pip install . --no-build-isolation "
            "--config-settings=setup-args=-Dcuda=disabled`."
        )


def _ensure_loaded():
    """Ensure extension is loaded before accessing ops."""
    if not _extension_loaded:
        _load_extension()


def _contains_cuda_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, (list, tuple)):
        return any(_contains_cuda_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(item) for item in value.values())
    return False


def _ensure_cuda_supported(args, kwargs):
    _ensure_loaded()
    if _cuda_dispatch_available:
        return
    if _contains_cuda_tensor(args) or _contains_cuda_tensor(kwargs):
        raise RuntimeError(
            "This d2p installation was built without CUDA support. "
            "Install a CUDA-enabled artifact or rebuild from source with "
            "`--config-settings=setup-args=-Dcuda=enabled`."
        )


# Load on module import
_load_extension()


# Expose the full d2p namespace for advanced users
d2p = torch.ops.d2p


_pt2_ops = None


def _get_pt2_ops():
    global _pt2_ops
    if _pt2_ops is None:
        from . import _pt2_ops as pt2_ops
        _pt2_ops = pt2_ops
    return _pt2_ops


def _should_use_pt2(args, kwargs) -> bool:
    if not kwargs:
        return use_pt2_ops(*args)
    return use_pt2_ops(*args, *kwargs.values())


def _wrap(name: str, *, pt2_name=None, postprocess=None):
    raw = getattr(torch.ops.d2p, name)
    pt2_name = pt2_name or name

    def wrapped(*args, **kwargs):
        _ensure_cuda_supported(args, kwargs)
        if _should_use_pt2(args, kwargs):
            pt2_ops = _get_pt2_ops()
            result = getattr(pt2_ops, pt2_name)(*args, **kwargs)
            if postprocess is not None:
                return postprocess(result)
            return result
        return raw(*args, **kwargs)

    return wrapped


# =============================================================================
# Smith-Waterman (linear gap)
# =============================================================================

soft_sw = _wrap("soft_sw")
soft_sw_float = _wrap("soft_sw_float")
soft_sw_with_grads = _wrap("soft_sw_with_grads")
soft_sw_hvp = _wrap("soft_sw_hvp")
soft_sw_param_jacobian = _wrap("soft_sw_param_jacobian")
soft_sw_backward_full = _wrap("soft_sw_backward_full")

# Smith-Waterman (affine gap)
soft_sw_affine = _wrap("soft_sw_affine")
soft_sw_affine_float = _wrap("soft_sw_affine_float")
soft_sw_affine_with_grads = _wrap("soft_sw_affine_with_grads")
soft_sw_affine_hvp = _wrap("soft_sw_affine_hvp")
soft_sw_affine_param_jacobian = _wrap("soft_sw_affine_param_jacobian")
soft_sw_affine_backward_full = _wrap("soft_sw_affine_backward_full")

# =============================================================================
# NEW API: Smith-Waterman (linear gap)
# These use cleaner naming: sw_forward instead of soft_sw_float
# =============================================================================

sw_forward = _wrap("sw_forward")
sw_forward_t = _wrap("sw_forward_t")
sw_value_grad_params = _wrap("sw_value_grad_params")
sw_marginals_backward = _wrap("sw_marginals_backward")
sw_marginals_hvp = _wrap("sw_marginals_hvp")
sw_marginals_grad_gap = _wrap("sw_marginals_grad_gap")
sw_marginals_grad_temp = _wrap("sw_marginals_grad_temp")

# NEW API: Smith-Waterman (affine gap)
sw_affine_forward = _wrap("sw_affine_forward")
sw_affine_forward_t = _wrap("sw_affine_forward_t")
sw_affine_value_grad_params = _wrap("sw_affine_value_grad_params")
sw_affine_marginals_backward = _wrap("sw_affine_marginals_backward")
sw_affine_marginals_hvp = _wrap("sw_affine_marginals_hvp")
sw_affine_marginals_grad_gap_open = _wrap("sw_affine_marginals_grad_gap_open")
sw_affine_marginals_grad_gap_ext = _wrap("sw_affine_marginals_grad_gap_ext")
sw_affine_marginals_grad_temp = _wrap("sw_affine_marginals_grad_temp")


# =============================================================================
# Dynamic Time Warping
# =============================================================================

soft_dtw = _wrap("soft_dtw")
soft_dtw_float = _wrap("soft_dtw_float")
soft_dtw_with_grads = _wrap("soft_dtw_with_grads")
soft_dtw_hvp = _wrap("soft_dtw_hvp")
soft_dtw_param_jacobian = _wrap("soft_dtw_param_jacobian")
soft_dtw_backward_full = _wrap("soft_dtw_backward_full")


# =============================================================================
# CKY Parsing
# =============================================================================

soft_cky = _wrap("soft_cky")
soft_cky_float = _wrap("soft_cky_float")
soft_cky_with_grads = _wrap("soft_cky_with_grads")
soft_cky_hvp = _wrap("soft_cky_hvp")
soft_cky_param_jacobian = _wrap("soft_cky_param_jacobian")
soft_cky_backward_full = _wrap("soft_cky_backward_full")


# =============================================================================
# Needleman-Wunsch (linear gap)
# =============================================================================

soft_nw = _wrap("soft_nw")
soft_nw_float = _wrap("soft_nw_float")
soft_nw_with_grads = _wrap("soft_nw_with_grads")
soft_nw_hvp = _wrap("soft_nw_hvp")
soft_nw_param_jacobian = _wrap("soft_nw_param_jacobian")
soft_nw_backward_full = _wrap("soft_nw_backward_full")

# Needleman-Wunsch (affine gap)
soft_nw_affine = _wrap("soft_nw_affine")
soft_nw_affine_float = _wrap("soft_nw_affine_float")
soft_nw_affine_with_grads = _wrap("soft_nw_affine_with_grads")
soft_nw_affine_hvp = _wrap("soft_nw_affine_hvp")
soft_nw_affine_param_jacobian = _wrap("soft_nw_affine_param_jacobian")
soft_nw_affine_backward_full = _wrap("soft_nw_affine_backward_full")


# =============================================================================
# Monotonic Alignment Search
# =============================================================================

soft_mas = _wrap("soft_mas", postprocess=lambda result: result[0])
soft_mas_float = _wrap("soft_mas_float")
soft_mas_with_grads = _wrap("soft_mas_with_grads")
soft_mas_hvp = _wrap("soft_mas_hvp")
soft_mas_param_jacobian = _wrap("soft_mas_param_jacobian")
soft_mas_backward_full = _wrap("soft_mas_backward_full")


# =============================================================================
# Eisner (Projective Dependency Parsing)
# =============================================================================

soft_eisner = _wrap("soft_eisner")
soft_eisner_float = _wrap("soft_eisner_float")
soft_eisner_with_grads = _wrap("soft_eisner_with_grads")
soft_eisner_hvp = _wrap("soft_eisner_hvp")
soft_eisner_param_jacobian = _wrap("soft_eisner_param_jacobian")
soft_eisner_backward_full = _wrap("soft_eisner_backward_full")


# =============================================================================
# Edit Distance Family
# =============================================================================

# Levenshtein
soft_levenshtein = _wrap("soft_levenshtein")
soft_levenshtein_float = _wrap("soft_levenshtein_float")
soft_levenshtein_with_grads = _wrap("soft_levenshtein_with_grads")
soft_levenshtein_hvp = _wrap("soft_levenshtein_hvp")
soft_levenshtein_param_jacobian = _wrap("soft_levenshtein_param_jacobian")
soft_levenshtein_backward_full = _wrap("soft_levenshtein_backward_full")

# Longest Common Subsequence
soft_lcs = _wrap("soft_lcs")
soft_lcs_float = _wrap("soft_lcs_float")
soft_lcs_with_grads = _wrap("soft_lcs_with_grads")
soft_lcs_hvp = _wrap("soft_lcs_hvp")
soft_lcs_param_jacobian = _wrap("soft_lcs_param_jacobian")
soft_lcs_backward_full = _wrap("soft_lcs_backward_full")

# OSA (Optimal String Alignment / Restricted Damerau-Levenshtein)
soft_osa = _wrap("soft_osa")
soft_osa_float = _wrap("soft_osa_float")
soft_osa_with_grads = _wrap("soft_osa_with_grads")
soft_osa_hvp = _wrap("soft_osa_hvp")
soft_osa_param_jacobian = _wrap("soft_osa_param_jacobian")
soft_osa_backward_full = _wrap("soft_osa_backward_full")

# True Damerau-Levenshtein
soft_damerau = _wrap("soft_damerau")
soft_damerau_float = _wrap("soft_damerau_float")
soft_damerau_with_grads = _wrap("soft_damerau_with_grads")
soft_damerau_hvp = _wrap("soft_damerau_hvp")
soft_damerau_param_jacobian = _wrap("soft_damerau_param_jacobian")
soft_damerau_backward_full = _wrap("soft_damerau_backward_full")
