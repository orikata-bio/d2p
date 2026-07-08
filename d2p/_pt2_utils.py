# SPDX-License-Identifier: Apache-2.0
"""
Helpers for PT2/torch.compile compatibility checks.
"""

from __future__ import annotations

import torch


def _is_fake_tensor(value: object) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    try:
        from torch._subclasses.fake_tensor import is_fake
    except Exception:
        return False
    return is_fake(value)


def use_pt2_ops(*args: object) -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and getattr(compiler, "is_compiling", None) is not None:
        if compiler.is_compiling():
            return True
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and getattr(dynamo, "is_compiling", None) is not None:
        if dynamo.is_compiling():
            return True
    return any(_is_fake_tensor(arg) for arg in args)
