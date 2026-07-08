# FAQ / Troubleshooting

Common questions and error messages when using d²p. For a feature-by-feature tour see the
[Usage Guide](usage.md); for per-operator inputs and outputs see the
[algorithm guides](algorithms/).

All examples run on CPU; swap in `device="cuda"` where a GPU is available.

### What shape does OSA's `trans_mask` take?

`[B, L1, L2]`, the **same shape as `sub_costs`**. It is nonzero wherever an adjacent
transposition is allowed to end at cell `(i, j)`. It is *not* `[B, L1-1, L2-1]`.

```python
import torch, d2p

sub_costs = torch.rand(2, 4, 5)
trans_mask = torch.zeros(2, 4, 5)        # same shape as sub_costs
result = d2p.soft_osa(sub_costs, trans_mask)
result.distance   # [B]
result.alignment  # [B, L1, L2]
```

Passing any other shape raises:

```
RuntimeError: trans_mask must have same shape as sub_costs
```

### What about Damerau's `trans_src`?

`trans_src` is `[B, L1, L2, 2]` and **must be `int32`**. The trailing pair holds the source
`(i, j)` index that each cell may transpose from. Two mistakes are common:

```python
import torch, d2p

sub_costs = torch.rand(2, 4, 5)
trans_src = torch.full((2, 4, 5, 2), -1, dtype=torch.int32)   # -1 = no transposition (0 is a valid source index)
result = d2p.soft_damerau(sub_costs, trans_src)
```

A 3-D tensor (dropping the trailing `2`) raises:

```
RuntimeError: trans_src must be 4D (B, L1, L2, 2)
```

A float tensor of the right shape raises:

```
RuntimeError: trans_src must be int32
```

### Why did `soft_mas` raise a `ValueError` about `T >= S`?

`soft_mas` takes `scores [B, T, S]` and requires **`T >= S`**: the row axis `T` is the number
of **frames** and the column axis `S` is the number of **tokens** (text positions). Each frame
is softly assigned to one token (rows sum to 1) and *every token must be covered*, which is only
possible when there are at least as many frames as tokens.

```python
import torch, d2p

# 50 frames aligned to 10 tokens: T=50 >= S=10
scores = torch.randn(2, 50, 10)
result = d2p.soft_mas(scores)
result.score      # [B] log-partition
result.alignment  # [B, T, S] = [B, 50, 10]
```

If `T < S` there is no valid full-coverage monotonic alignment, so `soft_mas` (and the `SoftMAS`
module) **raises `ValueError`**. Transpose your `scores` so the frame axis comes first
(`[B, T, S]` with `T >= S`). When you pass `lengths`, each row must likewise satisfy
`T_len >= S_len`.

### Which result field do I read?

Every high-level call returns a small `NamedTuple`. The scalar (score / cost / distance) is
`[B]`; the soft matrix (alignment / marginals) is the DP matrix, `[B, L1, L2]` for the pairwise
operators. Fields can be accessed by name or unpacked positionally.

| Family | Call(s) | Result | Scalar field | Matrix field |
| --- | --- | --- | --- | --- |
| Smith-Waterman | `soft_sw`, `soft_sw_affine` | `SWResult` | `value` | `marginals` |
| Needleman-Wunsch | `soft_nw`, `soft_nw_affine` | `NWResult` | `score` | `alignment` |
| Dynamic Time Warping | `soft_dtw` | `DTWResult` | `cost` | `alignment` |
| CKY | `soft_cky` | `CKYResult` | `score` | `marginals` |
| Eisner | `soft_eisner` | `EisnerResult` | `score` | `marginals` |
| Monotonic Alignment Search | `soft_mas` | `MASResult` | `score` | `alignment` |
| Levenshtein | `soft_levenshtein` | `LevenshteinResult` | `distance` | `alignment` |
| LCS | `soft_lcs` | `LCSResult` | `score` | `alignment` |
| OSA | `soft_osa` | `OSAResult` | `distance` | `alignment` |
| Damerau-Levenshtein | `soft_damerau` | `DamerauResult` | `distance` | `alignment` |

Two things to watch:

- **LCS uses `score`, not `distance`**: it is score-based (higher is better), unlike the other
  edit-distance operators.
- **Smith-Waterman exposes aliases.** On `SWResult`, `.score` is an alias of `.value`, and both
  `.alignment` and `.posteriors` alias `.marginals`.

```python
import torch, d2p

result = d2p.soft_sw(torch.rand(2, 4, 5))
result.value          # [B] log-partition value
result.marginals      # [B, L1, L2] soft alignment
result.score          # alias of .value
result.alignment      # alias of .marginals
```

The parser families return posterior charts rather than a pairwise matrix: Eisner `marginals`
are arc posteriors `[B, N, N]`, and CKY `marginals` are the 4-D merge field `[B, N, N, N]` (sum the
split axis, `marginals.sum(dim=2)`, for the `[B, N, N]` span chart).

### How do I make gap / temperature / costs learnable?

Any scalar hyperparameter (`gap`, `gap_open`, `gap_ext`, `temperature`, `ins_cost`, `del_cost`,
`trans_cost`) accepts either a Python `float` **or** a `torch.Tensor`. Pass a tensor with
`requires_grad=True` and gradients flow back to it:

```python
import torch, d2p

scores = torch.rand(2, 4, 5)
gap = torch.tensor(-1.0, requires_grad=True)
temperature = torch.tensor(0.5, requires_grad=True)

result = d2p.soft_sw(scores, gap=gap, temperature=temperature)
result.value.sum().backward()
gap.grad, temperature.grad     # populated
```

For a stateful layer, use the `nn.Module` wrappers, which register the chosen parameters. Note
the flag convention differs by family:

- **`SoftSW`** uses `learn_gap` and `learn_temperature`; **`SoftSWAffine`** uses `learn_gap_open`,
  `learn_gap_ext`, and `learn_temperature`.
- **Every other module** (`SoftNW`, `SoftDTW`, `SoftLevenshtein`, `SoftDamerau`, …) uses a single
  `learnable=True`.

```python
import torch, d2p

sw = d2p.SoftSW(gap=-1.0, temperature=1.0, learn_gap=True, learn_temperature=True)
list(sw.parameters())          # [gap, temperature]

nw = d2p.SoftNW(gap=-1.0, learnable=True)
list(nw.parameters())          # [gap, temperature]
```

### Why isn't the CPU backend using all my cores?

CPU parallelism is **across the batch**, not within a single DP. Each batch element's dynamic
program runs on its own thread via `at::parallel_for`, so throughput scales with the batch size
and with `torch.get_num_threads()`. A single-example call (`B == 1`) runs single-threaded, which
is expected. To use more cores, batch your inputs and/or raise the thread count:

```python
import torch, d2p

torch.set_num_threads(8)
scores = torch.rand(256, 64, 64)   # large B parallelizes across threads
result = d2p.soft_nw(scores)
```

### Does it work with `torch.compile` and autocast (AMP)?

`torch.compile`: yes (PyTorch 2.x), the operators are registered as custom ops and trace cleanly.
Autocast: the operators require **float32** inputs and are not autocast-eligible, so `torch.autocast`
does not downcast them and a `float16`/`bfloat16` input raises `scores must be float32`. Keep the d2p
inputs in float32 inside an autocast region (cast with `.float()` if an upstream op produced half
precision); the DP runs in FP32 and returns `float32`:

```python
import torch, d2p

with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    result = d2p.soft_nw(torch.rand(2, 4, 5).float())   # inputs must be float32
result.score.dtype    # torch.float32
```

### What dtype and shape are `lengths`?

`lengths` is optional and **`int32`**. It marks the valid extent of each example so padded
positions are excluded from the DP.

- **Pairwise operators** (`sw`, `sw_affine`, `nw`, `nw_affine`, `dtw`, `mas`, `levenshtein`,
  `lcs`, `osa`, `damerau`) take `lengths [B, 2]`: `(seq1_len, seq2_len)` per row.
- **Eisner** takes `lengths [B]`: the sentence length per row.

```python
import torch, d2p

# pairwise: [B, 2]
lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
d2p.soft_sw(torch.rand(2, 4, 5), lengths=lengths)

# eisner: [B]
lengths = torch.tensor([4, 3], dtype=torch.int32)
d2p.soft_eisner(torch.rand(2, 4, 4), lengths=lengths)
```

A wrong dtype is rejected with `lengths must be int32`; a wrong rank fails a shape check.

### What PyTorch, Python, and GPU versions are supported?

- **PyTorch** `>= 2.0` (the custom-op / `torch.compile` path needs 2.x).
- **Python** 3.10 - 3.14.
- **CUDA architectures** sm_75 (Turing) through sm_121 (Blackwell); the CPU backend has no GPU
  requirement.

d²p is released under Apache-2.0.

---

See also: [Usage Guide](usage.md) | [Algorithm guides](algorithms/) | [Examples](examples.md)
