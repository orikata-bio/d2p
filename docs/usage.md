# Usage Guide

Cross-cutting features shared by every d2p operator: result objects, differentiability and learnable parameters, `torch.compile`, variable-length batching, the `nn.Module` API, automatic mixed precision, the temperature parameter, and the low-level operator surface.

For per-operator inputs and outputs see the [algorithm guides](algorithms/); for complexity and throughput see [performance.md](performance.md).

## Result objects

Every high-level call returns a small named result (a `NamedTuple`) whose fields depend on the family. You can access fields by name or unpack/index the tuple.

| Family | Call(s) | Result | Fields |
| --- | --- | --- | --- |
| Smith-Waterman | `soft_sw`, `soft_sw_affine` | `SWResult` | `value`, `marginals` (aliases: `score`, `alignment`, `posteriors`) |
| Needleman-Wunsch | `soft_nw`, `soft_nw_affine` | `NWResult` | `score`, `alignment` |
| Dynamic Time Warping | `soft_dtw` | `DTWResult` | `cost`, `alignment` |
| CKY | `soft_cky` | `CKYResult` | `score`, `marginals` |
| Eisner | `soft_eisner` | `EisnerResult` | `score`, `marginals` |
| Monotonic Alignment Search | `soft_mas` | `MASResult` | `score`, `alignment` |
| Levenshtein | `soft_levenshtein` | `LevenshteinResult` | `distance`, `alignment` |
| LCS | `soft_lcs` | `LCSResult` | `score`, `alignment` |
| OSA | `soft_osa` | `OSAResult` | `distance`, `alignment` |
| Damerau-Levenshtein | `soft_damerau` | `DamerauResult` | `distance`, `alignment` |

```python
import torch, d2p

scores = torch.randn(2, 100, 120)
result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)
result.value          # [B] log-partition value
result.marginals      # [B, L1, L2] soft alignment
result.score          # alias of .value
```

## Full differentiability

Gradients flow through the input tensor **and** the scalar hyperparameters. Pass a tensor with `requires_grad=True` instead of a Python float to learn a hyperparameter:

```python
import torch, d2p

scores = torch.randn(2, 100, 120, requires_grad=True)
gap    = torch.tensor(-1.0, requires_grad=True)
temp   = torch.tensor(1.0, requires_grad=True)

result = d2p.soft_sw(scores, gap=gap, temperature=temp)
result.value.sum().backward()

scores.grad   # [2, 100, 120]
gap.grad      # gradient w.r.t. the gap penalty
temp.grad     # gradient w.r.t. temperature
```

The same holds for `gap_open`/`gap_ext` (affine variants) and `ins_cost`/`del_cost`/`trans_cost` (edit-distance family). Second-order information (Hessian-vector products) and first-order parameter Jacobians are exposed in the [low-level API](#low-level-api-d2pops).

## torch.compile

All operators run under `torch.compile` on PyTorch `2.x`:

```python
import torch, d2p

scores = torch.randn(2, 100, 120)

@torch.compile
def align(scores):
    return d2p.soft_sw(scores, gap=-1.0, temperature=1.0)

result = align(scores)
```

## Variable-length batching

Batch sequences of different lengths without splitting them apart. Pass a `lengths` tensor (the valid extent of each batch element, `[B, 2]` for the pairwise operators, `[B]` for Eisner); padded positions are excluded from the DP:

```python
import torch, d2p

scores = torch.randn(2, 100, 120)
lengths = torch.tensor([[50, 60],
                        [80, 100]], dtype=torch.int32)   # [B, 2]
result = d2p.soft_sw(scores, gap=-1.0, lengths=lengths)
```

Smith-Waterman also accepts boolean **prefix masks** (keyword-only `mask1`, `mask2`), which are equivalent to `lengths` for the common left-aligned case:

```python
import torch, d2p

scores = torch.randn(2, 100, 120)
mask1 = torch.zeros(2, 100, dtype=torch.bool); mask1[:, :50] = True
mask2 = torch.zeros(2, 120, dtype=torch.bool); mask2[:, :60] = True
result = d2p.soft_sw(scores, gap=-1.0, mask1=mask1, mask2=mask2)
```

## Module API

Each operator has an `nn.Module` wrapper for use inside a network. The constructor takes the same hyperparameters; make them learnable `nn.Parameter`s with the module's learn flags:

```python
import torch
import torch.nn as nn
import d2p

class Aligner(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.encoder = nn.Linear(dim, dim)
        self.sw = d2p.SoftSW(gap=-1.0, temperature=1.0,
                             learn_gap=True, learn_temperature=True)

    def forward(self, seq1, seq2):
        # encode both sequences, then form a [B, L1, L2] pairwise score matrix
        a, b = self.encoder(seq1), self.encoder(seq2)
        scores = torch.einsum("bik,bjk->bij", a, b)
        return self.sw(scores)

model  = Aligner()
result = model(torch.randn(2, 40, 64), torch.randn(2, 55, 64))
```

Learn-flag conventions:

- `SoftSW` / `SoftSWAffine` expose granular flags: `learn_gap`, `learn_temperature` (and `learn_gap_open`, `learn_gap_ext` for the affine variant).
- Every other module takes a single `learnable=True` that makes all of its scalar hyperparameters learnable.

Available modules: `SoftSW`, `SoftSWAffine`, `SoftNW`, `SoftNWAffine`, `SoftDTW`, `SoftCKY`, `SoftMAS`, `SoftEisner`, `SoftLevenshtein`, `SoftLCS`, `SoftOSA`, `SoftDamerau`.

## Mixed precision (AMP)

The operators require **float32** inputs and compute in FP32 internally for numerical stability. They are not autocast-eligible: `torch.autocast` does not downcast their inputs, and a `float16`/`bfloat16` input raises `scores must be float32`. Inside an autocast region, keep the d2p inputs in float32 (cast with `.float()` if an upstream op produced half precision):

```python
import torch, d2p

scores = torch.randn(2, 100, 120)   # float32
device = "cuda" if torch.cuda.is_available() else "cpu"
amp_dtype = torch.float16 if device == "cuda" else torch.bfloat16
with torch.autocast(device, dtype=amp_dtype):
    result = d2p.soft_sw(scores.to(device).float(), gap=-1.0, temperature=1.0)
result.value.dtype   # torch.float32
```

## Temperature

Temperature controls how soft the alignment is:

| Temperature | Behavior |
| --- | --- |
| `T → 0` | sharp, nearly hard (argmax) alignment |
| `T = 1` | standard soft alignment |
| `T → ∞` | uniform, maximally smooth |

```python
import torch, d2p

scores = torch.randn(2, 100, 120)
sharp  = d2p.soft_sw(scores, gap=-1.0, temperature=0.1)
smooth = d2p.soft_sw(scores, gap=-1.0, temperature=10.0)
```

Lower temperatures give sharper marginals but stiffer gradients; higher temperatures give a smoother optimization landscape. Temperature is itself differentiable (pass a tensor to learn it).

## Low-level API (`d2p.ops`)

`d2p.ops` exposes the underlying operators plus their derivative surfaces. Each operator `X` provides a consistent family of entry points:

| Entry point | Purpose |
| --- | --- |
| `ops.soft_X` | forward op (autograd-aware) |
| `ops.soft_X_float` | raw forward returning bare tensors |
| `ops.soft_X_with_grads` | forward plus input/parameter gradients |
| `ops.soft_X_backward_full` | full backward pass |
| `ops.soft_X_hvp` | Hessian-vector product |
| `ops.soft_X_param_jacobian` | first-order parameter Jacobian |

The arguments mirror the high-level call (scores/costs, then scalar hyperparameters, then `lengths`), and these entry points return bare tensors. For the Smith-Waterman family:

```python
import torch, d2p

scores = torch.randn(2, 40, 50)
value, marginals = d2p.ops.soft_sw_float(scores, -1.0, 1.0, None)              # value [B], marginals [B, L1, L2]
value, marginals, grad_gap, grad_temp = d2p.ops.soft_sw_with_grads(scores, -1.0, 1.0, None)
hvp = d2p.ops.soft_sw_hvp(scores, torch.randn_like(scores), -1.0, 1.0, None)   # [B, L1, L2]
```

Return arities vary by family (the edit-distance ops return extra per-cost gradients), so check a specific op before relying on it. The legacy `torch.ops.d2p.*` surface remains available as well. Prefer the high-level functions and modules unless you need a specific derivative surface.
