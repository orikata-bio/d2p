# Edit Distance Families

d²p ships four edit-distance-style operators:

- `d2p.soft_levenshtein(...)`: insert / delete / substitute
- `d2p.soft_lcs(...)`: longest common subsequence (score-based)
- `d2p.soft_osa(...)`: Levenshtein plus *restricted* adjacent transposition
- `d2p.soft_damerau(...)`: Levenshtein plus *unrestricted* transposition

Reach for this family when the question is "how much editing turns one sequence into another?" rather than "where do these two sequences best align?" ([Needleman-Wunsch](nw.md) / [Smith-Waterman](sw.md) answer the latter). All four are pairwise operators over a `[B, L1, L2]` matrix and are differentiable end to end.

## What it computes

**Edit distance** is the minimum total cost to transform sequence 1 into sequence 2 using single-character edits: insertions, deletions, and substitutions. Levenshtein is the classic case with unit costs; OSA and Damerau add a fourth operation, transposing two characters. All three are *costs*, so they are minimized.

**LCS** is the odd one out: the longest subsequence common to both sequences (a subsequence keeps order but may skip positions). It is expressed as a *score* to maximize, not a cost to minimize, so long shared runs push it up.

The differentiable ("soft") version returns a smooth `distance` (or `score` for LCS) per batch row and an `alignment` matrix whose entry `[i, j]` is the posterior probability that `seq1[i]` and `seq2[j]` are matched/substituted on the optimal edit path. Both are differentiable in the input matrix and in the cost/temperature hyperparameters: for the cost operators `alignment` is the gradient of `distance` with respect to `sub_costs`, and for LCS it is the gradient of `score` with respect to `match_scores`.

## The soft recurrence

The cost operators fill a DP table `D[i, j]` = best cost to edit the length-`i` prefix of seq1 into the length-`j` prefix of seq2, over a hard minimum of the ways to reach each cell:

```
D[i, j] = min(
    D[i-1, j-1] + sub_costs[i, j],   # substitute (or match, if cost 0)
    D[i-1, j]   + del_cost,          # delete seq1[i]
    D[i,   j-1] + ins_cost,          # insert seq2[j]
    <transposition>                  # OSA / Damerau only
)
```

with base cases `D[0, 0] = 0`, `D[i, 0] = i·del_cost`, `D[0, j] = j·ins_cost`. LCS instead **maximizes** `L[i, j] = max(L[i-1, j-1] + match_scores[i, j], L[i-1, j], L[i, j-1])`.

d²p replaces the hard `min` (or `max` for LCS) with a temperature-scaled **log-sum-exp** (a soft, differentiable min/max). `distance`/`score` becomes the smooth optimum over all edit paths and `alignment` is its gradient. `temperature` sets the softness:

- `T -> 0`: approaches the hard argmin/argmax (peaky, converges to the integer distance).
- `T = 1`: the standard soft relaxation.
- `T -> inf`: mass spreads toward uniform.

## Levenshtein

`soft_levenshtein` is the base insert/delete/substitute distance.

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `sub_costs` | `[B, L1, L2]` float | required | Cost of substituting `seq1[i]` with `seq2[j]`; `0` = identical, positive = mismatch. |
| `ins_cost` | float or Tensor | `1.0` | Cost to insert one `seq2` character. |
| `del_cost` | float or Tensor | `1.0` | Cost to delete one `seq1` character. |
| `temperature` | float or Tensor | `1.0` | Softness of the min; see above. |
| `lengths` | `[B, 2]` int32 | `None` | Valid `(seq1_len, seq2_len)` per row; excludes padding. |

Returns `LevenshteinResult(distance, alignment)`: `distance` is `[B]`, `alignment` is `[B, L1, L2]`.

## LCS

`soft_lcs` scores the longest common subsequence. It takes a **reward** matrix, not a cost matrix, and maximizes.

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `match_scores` | `[B, L1, L2]` float | required | Reward for matching `seq1[i]` with `seq2[j]`; higher = more alike (e.g. `1` on match, `0` otherwise). |
| `temperature` | float or Tensor | `1.0` | Softness of the max. |
| `lengths` | `[B, 2]` int32 | `None` | Valid `(seq1_len, seq2_len)` per row. |

Returns `LCSResult(score, alignment)`. Note the field is **`score`**, not `distance` (LCS is a reward, not a cost). `score` is `[B]`, `alignment` is `[B, L1, L2]`. There are no insertion/deletion parameters: LCS only rewards matches.

## OSA

`soft_osa` (Optimal String Alignment) extends Levenshtein with *restricted* adjacent transposition (a swap of two neighbouring characters), drawn from the fixed two-back diagonal `D[i-2, j-2] + trans_cost`. "Restricted" means a transposed pair cannot be edited again.

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `sub_costs` | `[B, L1, L2]` float | required | Substitution cost, as in Levenshtein. |
| `trans_mask` | `[B, L1, L2]` float | required | **Same shape as `sub_costs`.** Nonzero where an adjacent swap ending at `(i, j)` is legal: `seq1[i]==seq2[j-1]` and `seq1[i-1]==seq2[j]`. |
| `trans_cost` | float or Tensor | `1.0` | Cost of one adjacent transposition. |

`ins_cost`, `del_cost`, `temperature`, and `lengths` behave exactly as in Levenshtein. Returns `OSAResult(distance, alignment)`.

## Damerau-Levenshtein

`soft_damerau` uses the *unrestricted* transposition formulation. Instead of a boolean mask over the fixed two-back diagonal, each cell names an arbitrary earlier source cell to transpose from.

| Parameter | Shape / type | Default | Meaning |
| --- | --- | --- | --- |
| `sub_costs` | `[B, L1, L2]` float | required | Substitution cost, as in Levenshtein. |
| `trans_src` | `[B, L1, L2, 2]` **int32** | required | Source index `(k, l)` a transposition ending at `(i, j)` jumps back from. `-1` disables it. |
| `trans_cost` | float or Tensor | `1.0` | Base transposition cost. |

`ins_cost`, `del_cost`, `temperature`, and `lengths` behave exactly as in Levenshtein. Returns `DamerauResult(distance, alignment)`. The transposition path costs `D[k, l] + trans_cost + (i-k-1)·del_cost + (j-l-1)·ins_cost`: the base cost plus the characters *spanned* between source and destination, charged as deletions/insertions.

## OSA vs Damerau

Both add transposition to Levenshtein; they differ in how the source is expressed:

- **OSA (restricted).** Only *adjacent* swaps, always sourced from the fixed two-back diagonal `D[i-2, j-2]`, gated by a boolean `trans_mask` shaped like `sub_costs`. A transposed pair can't be touched again. Cheaper to set up: the mask is just "is this an adjacent swap?".
- **Damerau (unrestricted).** The source `(k, l)` may be *any* earlier cell, supplied per position in the integer `trans_src` table, and the characters between source and destination are charged as ins/del. This can express longer-range transpositions the restricted form cannot, at the cost of precomputing the source indices.

## Worked example

Build `sub_costs` from a character mismatch matrix (`0` on matches, `1` on mismatches) and measure the classic `kitten` → `sitting` distance of 3 (substitute k→s, substitute e→i, insert g):

```python
import torch
import d2p

def mismatch_costs(a, b):
    a_ids = torch.tensor([ord(c) for c in a])
    b_ids = torch.tensor([ord(c) for c in b])
    return (a_ids[:, None] != b_ids[None, :]).float().unsqueeze(0)  # [1, L1, L2]

sub_costs = mismatch_costs("kitten", "sitting")
res = d2p.soft_levenshtein(sub_costs, ins_cost=1.0, del_cost=1.0, temperature=0.1)
print(res.distance)         # tensor([3.0000])
print(res.alignment.shape)  # torch.Size([1, 6, 7])

# LCS scores the longest shared subsequence "ittn" (length 4) from a MATCH matrix:
match_scores = 1.0 - mismatch_costs("kitten", "sitting")  # 1 on matches, 0 otherwise
lcs = d2p.soft_lcs(match_scores, temperature=0.1)
print(lcs.score)            # tensor([4.2198])  -> 4.0 as T -> 0
```

`temperature=0.1` sharpens the soft distance onto the integer answer 3; the LCS soft `score` sits just above the hard length 4 and converges down to 4 as `T -> 0` (log-sum-exp upper-bounds the max).

Now a single adjacent swap, `abcd` → `acbd`, where OSA pays off. `trans_mask` marks the cell where the swap ends:

```python
import torch
import d2p

def mismatch_costs(a, b):
    a_ids = torch.tensor([ord(c) for c in a])
    b_ids = torch.tensor([ord(c) for c in b])
    return (a_ids[:, None] != b_ids[None, :]).float().unsqueeze(0)  # [1, L1, L2]

seq1, seq2 = "abcd", "acbd"          # bc <-> cb, one adjacent transposition
sub_costs = mismatch_costs(seq1, seq2)

L1, L2 = len(seq1), len(seq2)
trans_mask = torch.zeros(1, L1, L2)  # same shape as sub_costs
for i in range(1, L1):
    for j in range(1, L2):
        if seq1[i] == seq2[j - 1] and seq1[i - 1] == seq2[j]:
            trans_mask[0, i, j] = 1.0

lev = d2p.soft_levenshtein(sub_costs, temperature=0.05)
osa = d2p.soft_osa(sub_costs, trans_mask, trans_cost=1.0, temperature=0.05)
print(lev.distance, osa.distance)    # tensor([1.9451]) tensor([1.0000])
```

Plain Levenshtein needs ~2 edits to reorder the adjacent `bc`; OSA recognizes the single transposition and reports distance 1. The Damerau form takes an integer `trans_src` table instead and is differentiable in the same way:

```python
import torch
import d2p

def mismatch_costs(a, b):
    a_ids = torch.tensor([ord(c) for c in a])
    b_ids = torch.tensor([ord(c) for c in b])
    return (a_ids[:, None] != b_ids[None, :]).float().unsqueeze(0)  # [1, L1, L2]

seq1, seq2 = "abcd", "acbd"
L1, L2 = len(seq1), len(seq2)

trans_src = torch.full((1, L1, L2, 2), -1, dtype=torch.int32)  # -1 = no transposition
for i in range(1, L1):
    for j in range(1, L2):
        if seq1[i] == seq2[j - 1] and seq1[i - 1] == seq2[j]:
            trans_src[0, i, j, 0] = i - 1
            trans_src[0, i, j, 1] = j - 1

sub_costs = mismatch_costs(seq1, seq2).requires_grad_(True)
dam = d2p.soft_damerau(sub_costs, trans_src, trans_cost=1.0, temperature=0.05)
dam.distance.sum().backward()
print(dam.distance, sub_costs.grad.shape)  # tensor([1.9451]) torch.Size([1, 4, 4])
```

Because Damerau also charges the characters spanned between source and destination, a *pure* 2-character adjacent swap here costs `trans_cost + del_cost + ins_cost` and does not undercut the plain edit path. Its advantage is structural, not this specific number: the source may be any earlier cell, letting it model transpositions OSA's fixed two-back diagonal cannot reach.

## Learnable parameters

Pass any of `ins_cost`, `del_cost`, `trans_cost`, or `temperature` as a `requires_grad=True` tensor and gradients flow back to them, so edit penalties can be trained jointly with the rest of a model:

```python
import torch
import d2p

def mismatch_costs(a, b):
    a_ids = torch.tensor([ord(c) for c in a])
    b_ids = torch.tensor([ord(c) for c in b])
    return (a_ids[:, None] != b_ids[None, :]).float().unsqueeze(0)  # [1, L1, L2]

ins_cost = torch.tensor(1.0, requires_grad=True)
del_cost = torch.tensor(1.0, requires_grad=True)
sub_costs = mismatch_costs("kitten", "sitting")

res = d2p.soft_levenshtein(sub_costs, ins_cost=ins_cost, del_cost=del_cost, temperature=0.5)
res.distance.sum().backward()
print(ins_cost.grad, del_cost.grad)  # both populated
```

For a module-style API with managed parameters, `d2p.SoftLevenshtein`, `d2p.SoftLCS`, `d2p.SoftOSA`, and `d2p.SoftDamerau` wrap these calls; each takes `learnable=True` to register its cost parameters (see the [Usage Guide](../usage.md)).

## Variable-length batching

Pad every matrix to a common `[B, L1, L2]` and pass `lengths [B, 2]` (int32) giving the true `(seq1_len, seq2_len)` of each row. The DP then runs only within each row's real prefix and padded positions carry no mass:

```python
import torch
import d2p

sub_costs = torch.rand(3, 10, 12)
lengths = torch.tensor([[10, 12], [6, 7], [8, 5]], dtype=torch.int32)
res = d2p.soft_levenshtein(sub_costs, lengths=lengths, temperature=0.5)
print(res.distance.shape)  # torch.Size([3])
```

The same `lengths [B, 2]` argument works for all four operators.

## Common pitfalls

- **OSA `trans_mask` shape.** It must be `[B, L1, L2]`, the *same* shape as `sub_costs`, nonzero where an adjacent swap is legal. A `[B, L1-1, L2-1]` tensor raises `RuntimeError`.
- **Damerau `trans_src` shape and dtype.** It must be `[B, L1, L2, 2]` and **int32**, with `-1` marking "no transposition" (`0` is a valid source index `(0, 0)`, not a disable). A float tensor or a `[B, L1, L2]` tensor raises `RuntimeError`. This is the single most common error in this family.
- **LCS uses `score`, not `distance`.** `soft_lcs(...).score` is the reward; accessing `.distance` raises `AttributeError`.
- **Costs are penalties.** `sub_costs`, `ins_cost`, `del_cost`, and `trans_cost` should be `>= 0`. Build `sub_costs` so matches are `0` and mismatches positive. LCS's `match_scores` are the opposite: higher = better.
- **`lengths` dtype.** Must be int32 and shaped `[B, 2]` `(seq1_len, seq2_len)`; a `[B]` vector or a float tensor will not work.
- **Temperature vs the integer distance.** At `T = 1` the softmin is a smooth relaxation and can fall *below* the hard integer distance (even slightly negative); lower `T` sharpens toward the true value.

## Complexity

`O(L1 · L2)` time and memory per batch element for all four operators; OSA and Damerau carry a small constant factor for the transposition transition. On CPU each batch element runs on its own thread via `at::parallel_for`, scaling with `torch.get_num_threads()`. The low-level `d2p.ops` API exposes the same `soft_X`, `soft_X_with_grads`, `soft_X_backward_full`, `soft_X_hvp`, and `soft_X_param_jacobian` surfaces for each operator.

The edit distance is due to Levenshtein (1966); the transposition-aware variant to Damerau (1964). OSA is the restricted ("optimal string alignment") form of Damerau-Levenshtein.

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
