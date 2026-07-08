# Needleman-Wunsch

d²p ships two global-alignment operators:

- `d2p.soft_nw(...)` for linear gaps
- `d2p.soft_nw_affine(...)` for affine gaps

Use Needleman-Wunsch when both sequences should be aligned *end to end*: every position of each sequence is either matched to a position of the other or charged as a gap. This is the right choice when the two inputs are variants of the same whole (two full sentences, two protein sequences of comparable length) rather than a short motif hidden in noise.

## What it computes

Given a pairwise similarity matrix `scores[b, i, j]`, Needleman-Wunsch finds the single best alignment path that runs from corner `(0, 0)` to corner `(L1-1, L2-1)`, threading through matches and gaps. Unlike [Smith-Waterman](sw.md), there is no local restart: the path must span the full matrix, so both sequences are consumed completely.

The differentiable ("soft") version returns a smooth alignment `score` per batch row and an `alignment` matrix whose entry `[i, j]` is the posterior probability that positions `i` and `j` are matched on the path. Both are differentiable in `scores` and in the gap/temperature hyperparameters.

## The soft recurrence

Classic Needleman-Wunsch fills a DP table with a hard maximum over the three ways to reach a cell: diagonal (match/mismatch), up (gap in one sequence), left (gap in the other).

```
H[i, j] = max(H[i-1, j-1] + scores[i, j], H[i-1, j] + gap, H[i, j-1] + gap)
```

There is no `0` term as in Smith-Waterman, which is exactly why the alignment is global: the score can go arbitrarily negative and the path is still forced to reach the far corner.

d²p replaces the hard `max` with a temperature-scaled **log-sum-exp** (a soft, differentiable maximum). `score` becomes the log-partition-style score over *all* global alignments, and `alignment` is its gradient with respect to `scores` (the expected match matrix under the induced distribution). `temperature` sets the softness:

- `T -> 0`: approaches the hard argmax path (peaky alignment).
- `T = 1`: the standard soft alignment.
- `T -> inf`: mass spreads toward uniform.

## Inputs

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `scores` | `[B, L1, L2]` float | required | Pairwise similarity; higher = better match. |
| `gap` *(soft_nw)* | float or Tensor | `-1.0` | Penalty per gap position (insertion or deletion). Keep it negative. |
| `gap_open` *(soft_nw_affine)* | float or Tensor | `-2.0` | Cost to *start* a gap run. |
| `gap_ext` *(soft_nw_affine)* | float or Tensor | `-0.5` | Cost per additional position once a gap is open. |
| `temperature` | float or Tensor | `1.0` | Softness of the max; see above. |
| `lengths` | `[B, 2]` int32 | `None` | Valid `(seq1_len, seq2_len)` per row; excludes padding. |

Any scalar hyperparameter (`gap`, `gap_open`, `gap_ext`, `temperature`) accepts a Python float **or** a `torch.Tensor` with `requires_grad=True`, which makes it learnable. Unlike [Smith-Waterman](sw.md), Needleman-Wunsch does **not** take `mask1`/`mask2`. Use `lengths` to handle padding.

## Outputs

Both functions return an `NWResult` named tuple:

- `score`: soft alignment score, `[B]`.
- `alignment`: soft alignment matrix, `[B, L1, L2]`.

(There are no extra aliases; the fields are exactly `score` and `alignment`.)

## Linear vs affine gaps

- `soft_nw` charges a flat `gap` per gap position.
- `soft_nw_affine` charges `gap_open` once to start a gap run and `gap_ext` for each subsequent position.

The affine variant is usually the better model when gap *runs* are meaningful: a single five-position indel should not cost five times a one-position indel. Use `soft_nw` when a flat per-position penalty is sufficient.

## Worked example

Two sequences that are identical except for one deleted character. Score `+1` for a match and `-1` for a mismatch, then align globally:

```python
import torch
import d2p

seq1 = "ACGTACGT"
seq2 = "ACGACGT"    # same sequence, one deleted "T"

a = torch.tensor([ord(c) for c in seq1])
b = torch.tensor([ord(c) for c in seq2])
scores = (2 * (a[:, None] == b[None, :]).float() - 1).unsqueeze(0)

result = d2p.soft_nw(scores, gap=-1.0, temperature=0.5)
m = result.alignment[0]

print("corners:", round(m[0, 0].item(), 2), round(m[-1, -1].item(), 2))
for i, c in enumerate(seq1):
    j = int(m[i].argmax())
    print(f"seq1[{i}]={c} -> seq2[{j}]={seq2[j]}   mass={m[i, j]:.2f}")
```

```
corners: 0.99 0.99
seq1[0]=A -> seq2[0]=A   mass=0.99
seq1[1]=C -> seq2[1]=C   mass=0.99
seq1[2]=G -> seq2[2]=G   mass=0.97
seq1[3]=T -> seq2[3]=A   mass=0.02
seq1[4]=A -> seq2[3]=A   mass=0.97
seq1[5]=C -> seq2[4]=C   mass=0.99
seq1[6]=G -> seq2[5]=G   mass=0.99
seq1[7]=T -> seq2[6]=T   mass=0.99
```

Both corner cells carry ~1.0 mass, confirming the path runs the full length of the matrix. The deleted character shows up as a gap: `seq1[3]="T"` has almost no mass (0.02) because it has no partner in `seq2`, and the alignment resumes at `seq1[4]="A" -> seq2[3]="A"`. Everything else lands on the main diagonal, exactly as expected for two near-identical sequences.

## Learnable parameters

Pass the gap or temperature as a `requires_grad=True` tensor to train the alignment penalty jointly with the rest of a model. This also shows `soft_nw_affine` with variable lengths:

```python
import torch
import d2p

scores = torch.randn(2, 9, 9, requires_grad=True)
gap_open = torch.tensor(-2.0, requires_grad=True)
gap_ext = torch.tensor(-0.5, requires_grad=True)
lengths = torch.tensor([[9, 9], [5, 4]], dtype=torch.int32)

result = d2p.soft_nw_affine(scores, gap_open=gap_open, gap_ext=gap_ext, lengths=lengths)
result.score.sum().backward()

print(gap_open.grad, gap_ext.grad)  # both populated
```

For a module-style API with managed parameters, `d2p.SoftNW` / `d2p.SoftNWAffine` wrap these calls (see the [Usage Guide](../usage.md)).

## Variable-length batching

Pad `scores` to a common `[B, L1, L2]` and pass `lengths [B, 2]` (int32) giving the true `(seq1_len, seq2_len)` of each row. The alignment then runs only to each row's real corner and padded positions carry no mass, as used in the example above. Because Needleman-Wunsch has no masks, `lengths` is the only padding mechanism.

## Common pitfalls

- **Sign of the gap.** `gap`, `gap_open`, and `gap_ext` are penalties; they should be negative. A positive gap rewards skipping and destroys the alignment.
- **`lengths` dtype.** `lengths` must be int32 and shaped `[B, 2]`. A `[B]` vector or a float tensor will not work.
- **No masks.** Passing `mask1`/`mask2` raises `TypeError`; use `lengths` instead.
- **Global, not local.** If dissimilar ends are dragging the alignment around, you may actually want [Smith-Waterman](sw.md). Needleman-Wunsch is obligated to align the ends.
- **Score scale vs temperature.** Very large `scores` magnitudes act like a low temperature (peaky), very small ones like a high temperature (diffuse). Tune them together.

## Complexity

`O(L1 · L2)` time and memory per batch element for both variants; the affine version carries a small constant factor for its extra gap state. On CPU each batch element runs on its own thread via `at::parallel_for`, scaling with `torch.get_num_threads()`. The low-level `d2p.ops` API exposes the same value-grad, `*_hvp`, `*_backward_full`, and `*_param_jacobian` surfaces as the Smith-Waterman family.

The underlying algorithm is the classic global-alignment method of Needleman & Wunsch (1970).

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
