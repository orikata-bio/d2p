# Smith-Waterman

d²p ships two local-alignment operators:

- `d2p.soft_sw(...)` for linear gaps
- `d2p.soft_sw_affine(...)` for affine gaps

Use Smith-Waterman when you want the best *local* subsequence match rather than an end-to-end global alignment. A short shared motif buried inside two otherwise dissimilar sequences is the canonical case: the operator finds where the sequences agree and ignores the flanks.

## What it computes

Given a pairwise similarity matrix `scores[b, i, j]` (how well position `i` of sequence 1 matches position `j` of sequence 2), Smith-Waterman finds the highest-scoring contiguous alignment between *sub*-ranges of the two sequences. Because the alignment can start and stop anywhere, mass is free to concentrate on a single well-matched block and skip everything around it. This is what makes it *local* alignment, in contrast to the global, corner-to-corner path of [Needleman-Wunsch](nw.md).

The differentiable ("soft") version returns a smooth alignment `value` per batch row and a `marginals` matrix whose entry `[i, j]` is the posterior probability that cell `(i, j)` lies on the alignment path. Both are differentiable in `scores` and in the gap/temperature hyperparameters.

## The soft recurrence

Classic Smith-Waterman fills a DP table with a hard maximum:

```
H[i, j] = max(0, H[i-1, j-1] + scores[i, j], H[i-1, j] + gap, H[i, j-1] + gap)
```

The leading `0` is the local-alignment escape hatch: whenever the running score would go negative the alignment restarts, so a poor prefix never drags down a good interior match.

d²p replaces the hard `max` with a temperature-scaled **log-sum-exp** (a soft, differentiable maximum). `value` is then the log-partition-style score over all local alignments, and `marginals` is exactly its gradient with respect to `scores` (the expected alignment under the induced distribution). `temperature` controls the softness:

- `T -> 0`: approaches the hard argmax path (peaky marginals).
- `T = 1`: the standard soft alignment.
- `T -> inf`: mass spreads toward uniform.

## Inputs

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `scores` | `[B, L1, L2]` float | required | Pairwise similarity; higher = better match. |
| `gap` *(soft_sw)* | float or Tensor | `-1.0` | Penalty per gap position (insertion or deletion). Keep it negative. |
| `gap_open` *(soft_sw_affine)* | float or Tensor | `-2.0` | Cost to *start* a gap run. |
| `gap_ext` *(soft_sw_affine)* | float or Tensor | `-0.5` | Cost per additional position once a gap is open. |
| `temperature` | float or Tensor | `1.0` | Softness of the max; see above. |
| `lengths` | `[B, 2]` int32 | `None` | Valid `(seq1_len, seq2_len)` per row; excludes padding. |
| `mask1` | `[B, L1]` bool | `None` | Keyword-only. Valid positions in sequence 1 (prefix mask). |
| `mask2` | `[B, L2]` bool | `None` | Keyword-only. Valid positions in sequence 2 (prefix mask). |

Any scalar hyperparameter (`gap`, `gap_open`, `gap_ext`, `temperature`) accepts a Python float **or** a `torch.Tensor` with `requires_grad=True`, which makes it learnable. `mask1`/`mask2` are keyword-only and specific to Smith-Waterman. Needleman-Wunsch does not take masks.

## Outputs

Both functions return an `SWResult` named tuple:

- `value`: soft alignment score, `[B]`.
- `marginals`: soft alignment matrix, `[B, L1, L2]`.

Backward-compatibility aliases point at the same tensors:

- `.score` -> `value`
- `.alignment` -> `marginals`
- `.posteriors` -> `marginals`

## Linear vs affine gaps

- `soft_sw` charges a flat `gap` per gap position.
- `soft_sw_affine` charges `gap_open` once to start a gap run and `gap_ext` for each subsequent position.

Affine gaps let one long gap cost far less than many short gaps, which better matches biological indels and most real-world "skip a stretch" situations. Reach for `soft_sw_affine` when gap *runs* are meaningful; use `soft_sw` when a flat per-position penalty is enough.

## Worked example

A shared motif `ACGTA` sits inside two different flanking contexts. Score `+1` for a character match and `-1` for a mismatch, then align:

```python
import torch
import d2p

# A shared motif "ACGTA" embedded in different flanks.
seq1 = "TTACGTAGG"   # motif at positions 2..6
seq2 = "CACGTATT"    # motif at positions 1..5

a = torch.tensor([ord(c) for c in seq1])
b = torch.tensor([ord(c) for c in seq2])
scores = (2 * (a[:, None] == b[None, :]).float() - 1).unsqueeze(0)  # +1 match, -1 mismatch

result = d2p.soft_sw(scores, gap=-1.0, temperature=0.5)
m = result.marginals[0]

for i, c in enumerate(seq1):
    j = int(m[i].argmax())
    print(f"seq1[{i}]={c} -> seq2[{j}]={seq2[j]}   mass={m[i, j]:.2f}")
```

```
seq1[0]=T -> seq2[0]=C   mass=0.01
seq1[1]=T -> seq2[0]=C   mass=0.10
seq1[2]=A -> seq2[1]=A   mass=0.87
seq1[3]=C -> seq2[2]=C   mass=0.97
seq1[4]=G -> seq2[3]=G   mass=0.98
seq1[5]=T -> seq2[4]=T   mass=0.97
seq1[6]=A -> seq2[5]=A   mass=0.87
seq1[7]=G -> seq2[6]=T   mass=0.11
seq1[8]=G -> seq2[7]=T   mass=0.01
```

Read the `marginals` as a soft alignment: the five motif positions (`seq1[2..6]`) map cleanly onto `seq2[1..5]` with near-1.0 mass along a diagonal, while the flanking `TT.../..GG` characters get almost none. That concentration on an interior block (with the ends carrying essentially zero mass) is the signature of local alignment.

## Learnable parameters

Pass the gap or temperature as a `requires_grad=True` tensor and gradients flow back to them, so an alignment penalty can be trained jointly with the rest of a model:

```python
import torch
import d2p

scores = torch.randn(4, 50, 60, requires_grad=True)
gap = torch.tensor(-1.5, requires_grad=True)
temperature = torch.tensor(0.7, requires_grad=True)

result = d2p.soft_sw(scores, gap=gap, temperature=temperature)
result.value.sum().backward()

print(gap.grad, temperature.grad)  # both populated

# The affine variant trains its two gap costs the same way:
gap_open = torch.tensor(-2.0, requires_grad=True)
gap_ext = torch.tensor(-0.5, requires_grad=True)
d2p.soft_sw_affine(scores, gap_open=gap_open, gap_ext=gap_ext).value.sum().backward()
print(gap_open.grad, gap_ext.grad)  # both populated
```

For a module-style API with managed parameters, `d2p.SoftSW` / `d2p.SoftSWAffine` wrap these calls (see the [Usage Guide](../usage.md)).

## Variable-length batching

Real batches mix sequence lengths. Pad `scores` to `[B, L1, L2]` and pass `lengths [B, 2]` (int32) giving the true `(seq1_len, seq2_len)` of each row; padded positions are then excluded from the alignment. Boolean prefix masks `mask1 [B, L1]` / `mask2 [B, L2]` are an equivalent way to mark valid positions:

```python
import torch
import d2p

scores = torch.randn(2, 10, 12)

# Option A: lengths [B, 2] int32
lengths = torch.tensor([[10, 12], [6, 7]], dtype=torch.int32)
padded = d2p.soft_sw(scores, lengths=lengths).marginals[1]
# Row 1 only aligns within its 6x7 prefix; padded cells carry no mass.
print(padded[:6, :7].sum().item(), (padded.sum() - padded[:6, :7].sum()).item())

# Option B: boolean prefix masks (Smith-Waterman only)
mask1 = torch.zeros(2, 10, dtype=torch.bool); mask1[:, :6] = True
mask2 = torch.zeros(2, 12, dtype=torch.bool); mask2[:, :7] = True
print(d2p.soft_sw(scores, mask1=mask1, mask2=mask2).value.shape)  # torch.Size([2])
```

## Common pitfalls

- **Sign of the gap.** `gap`, `gap_open`, and `gap_ext` are penalties; they should be negative. A positive gap rewards skipping and collapses the alignment.
- **`lengths` dtype.** `lengths` must be int32 and shaped `[B, 2]`. A `[B]` vector or a float tensor will not work.
- **Masks are Smith-Waterman only.** `mask1`/`mask2` are keyword-only here; `soft_nw` raises `TypeError` if you pass them.
- **Local, not global.** If you actually need both sequences aligned end to end, use [Needleman-Wunsch](nw.md). Smith-Waterman will happily ignore poorly-matched ends.
- **Score scale vs temperature.** Very large `scores` magnitudes act like a low temperature (peaky), very small ones like a high temperature (diffuse). Tune them together.

## Complexity

`O(L1 · L2)` time and memory per batch element for both variants; the affine version carries a small constant factor for its extra gap state. On CPU each batch element runs on its own thread via `at::parallel_for`, scaling with `torch.get_num_threads()`. The low-level `d2p.ops` API exposes matching `*_backward_full`, `*_hvp`, and `*_param_jacobian` surfaces for both variants. The underlying algorithm is the classic local-alignment method of Smith & Waterman (1981).

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
