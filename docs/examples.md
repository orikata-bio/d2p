# Examples

Six end-to-end, runnable examples: build an input tensor, make one d²p call, and read the result. Each snippet is self-contained: copy it and run. The `device` line selects CUDA when available, which assumes a CUDA-enabled d2p build; on a CPU-only build set `device = "cpu"`.

For the cross-cutting API (result objects, learnable parameters, batching, modules) see the [Usage Guide](usage.md); for per-operator inputs and outputs see the [algorithm guides](algorithms/).

## 1. Local sequence alignment (`soft_sw`)

Smith-Waterman finds the best-matching *local* subsequence. Here the score matrix is a dot-product between two sets of per-position embeddings; `.marginals` is the soft alignment, and gradients flow back to the embeddings.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

# Two batches of sequences, each position carrying a learned embedding.
B, L1, L2, dim = 2, 40, 55, 16
emb1 = torch.randn(B, L1, dim, device=device, requires_grad=True)   # query
emb2 = torch.randn(B, L2, dim, device=device, requires_grad=True)   # target

# Score matrix: dot-product similarity of every query/target position pair.
scores = torch.einsum("bik,bjk->bij", emb1, emb2)                   # [B, L1, L2]

result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)
align = result.marginals            # [B, L1, L2] soft local alignment (posteriors)
print(result.value.shape, align.shape)          # [2], [2, 40, 55]

# Backprop: pull the two sequences into better local agreement.
loss = -result.value.sum()          # maximize the local-alignment score
loss.backward()
print(emb1.grad.shape)                           # [2, 40, 16]
```

`marginals[b, i, j]` is the posterior that position `i` of the query aligns to position `j` of the target; the mass concentrates on the best-scoring local diagonal. `value` is the log-partition alignment score. Maximizing it (minimizing `-value`) tightens the match.

## 2. Global alignment with affine gaps (`soft_nw_affine`)

Needleman-Wunsch aligns two sequences *end to end*. The affine variant charges `gap_open` to start a gap and the smaller `gap_ext` for each additional step, so one long gap is cheaper than many short ones. The soft path is in `.alignment`.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

B, L1, L2, dim = 2, 30, 32, 16
emb1 = torch.randn(B, L1, dim, device=device)
emb2 = torch.randn(B, L2, dim, device=device)
scores = torch.einsum("bik,bjk->bij", emb1, emb2)                  # [B, L1, L2]
scores.requires_grad_(True)

# Affine gaps: opening a gap costs gap_open, extending it costs gap_ext each step.
result = d2p.soft_nw_affine(scores, gap_open=-2.0, gap_ext=-0.5, temperature=1.0)
align = result.alignment            # [B, L1, L2] soft end-to-end alignment
print(result.score.shape, align.shape)           # [2], [2, 30, 32]

# alignment[b, i, j] is the posterior that position i of seq1 aligns to j of seq2.
best_match = align[0].argmax(dim=-1)             # [L1] most-aligned seq2 index
print(best_match.shape)                          # [30]

result.score.sum().backward()
print(scores.grad.shape)                         # [2, 30, 32]
```

`alignment[b, i, j]` is the posterior that seq1 position `i` is matched to seq2 position `j`; `argmax` along the last axis reads off each position's most likely partner. `score` is the global log-partition value.

## 3. Differentiable time-series loss (`soft_dtw`)

Soft-DTW scores the similarity of two time series under optimal time warping. It consumes a **cost** matrix (lower = more similar). `torch.cdist` gives the pairwise distances, and `.cost` is directly usable as a training loss.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

B, T1, T2, feat = 4, 60, 48, 8
series_a = torch.randn(B, T1, feat, device=device, requires_grad=True)
series_b = torch.randn(B, T2, feat, device=device)

# DTW consumes a COST matrix (lower = more similar). cdist gives pairwise L2.
costs = torch.cdist(series_a, series_b)          # [B, T1, T2]

result = d2p.soft_dtw(costs, temperature=0.1, bandwidth=0)
print(result.cost.shape, result.alignment.shape)  # [4], [4, 60, 48]

# The soft-DTW cost is a differentiable similarity loss between the series.
loss = result.cost.mean()
loss.backward()
print(series_a.grad.shape)                         # [4, 60, 8]

# bandwidth > 0 applies a Sakoe-Chiba band, forbidding warps more than that
# many steps off the diagonal (rules out pathological warps):
banded = d2p.soft_dtw(costs.detach(), temperature=0.1, bandwidth=5)
print(banded.cost.shape)                           # [4]
```

`cost` is the soft-DTW distance per batch element; smaller means a better warp-aligned match, so `cost.mean()` is a drop-in loss. `alignment` is the soft warping path `[B, T1, T2]`. `bandwidth > 0` restricts warping to a Sakoe-Chiba band around the diagonal (`0` disables it).

## 4. Monotonic alignment for TTS (`soft_mas`)

Monotonic Alignment Search aligns `T` frames to `S` tokens under a strict left-to-right constraint (each frame stays on or advances to the next token) and requires **`T >= S`**. This is the alignment step used by TTS models such as Glow-TTS.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

B, T, S, dim = 2, 80, 20, 32        # T acoustic frames, S text tokens, T >= S
frame_emb = torch.randn(B, T, dim, device=device, requires_grad=True)
token_emb = torch.randn(B, S, dim, device=device, requires_grad=True)

# Frame/token affinity; MAS finds a monotonic frame -> token assignment.
scores = torch.einsum("btd,bsd->bts", frame_emb, token_emb)       # [B, T, S]

result = d2p.soft_mas(scores, temperature=1.0)
align = result.alignment            # [B, T, S] soft monotonic alignment
print(result.score.shape, align.shape)           # [2], [2, 80, 20]

# Each frame's row is a distribution over tokens (sums to 1); argmax reads the
# hard monotonic path back out.
frame_to_token = align[0].argmax(dim=-1)         # [T] token index per frame
print(frame_to_token.shape)                      # [80]

result.score.sum().backward()
print(frame_emb.grad.shape)                      # [2, 80, 32]
```

Each row of `alignment[b]` is a distribution over tokens for one frame (rows sum to 1); `argmax` along the token axis recovers the hard alignment, a non-decreasing token index as you move through the frames. Duplicated indices are the frames spent on one token (its predicted duration).

## 5. Structure marginals (`soft_cky`, `soft_eisner`)

Both parsers return **posteriors** over structure, not just a single best tree. CKY gives constituency spans; Eisner gives projective dependency arcs.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
B, N = 2, 8

# --- CKY: constituency structure ---------------------------------------------
merge_scores = torch.randn(B, N, N, N, device=device, requires_grad=True)
leaf_scores = torch.randn(B, N, device=device, requires_grad=True)

cky = d2p.soft_cky(merge_scores, leaf_scores, temperature=1.0)
merges = cky.marginals              # [B, N, N, N]; merges[b, i, k, j] = P(span (i,j) split at k)
spans = merges.sum(dim=2)           # [B, N, N]; spans[b, i, j] = P(span (i,j) is a constituent)
print(cky.score.shape, merges.shape, spans.shape)   # [2], [2,8,8,8], [2,8,8]
print(spans[0, 0, N - 1])           # root span (0, N-1): ~1.0 in every parse

# --- Eisner: projective dependency arcs --------------------------------------
arc_scores = torch.randn(B, N, N, device=device, requires_grad=True)  # [b, head, dep]
eis = d2p.soft_eisner(arc_scores, temperature=1.0)
arcs = eis.marginals                # [B, N, N]; arcs[b, h, d] = P(arc head h -> dep d)
print(eis.score.shape, arcs.shape)                  # [2], [2, 8, 8]

(cky.score.sum() + eis.score.sum()).backward()
print(merge_scores.grad.shape, arc_scores.grad.shape)  # [2,8,8,8], [2,8,8]
```

`soft_cky.marginals` is the 4-D merge field `[B, N, N, N]`, where `marginals[b, i, k, j]` is the posterior of building span `(i, j)` by splitting at `k`; sum out the split axis to get span posteriors `spans[b, i, j]` (the full-sentence span `(0, N-1)` always has posterior ~1). `soft_eisner.marginals[b, h, d]` is the posterior of a dependency arc from head `h` to dependent `d` (position `0` is the ROOT). Both fields are fully differentiable w.r.t. their scores.

## 6. Fuzzy string matching (`soft_levenshtein`, `soft_osa`)

The edit-distance family turns discrete string operations into differentiable losses. Build `sub_costs` from a character-equality matrix (0 where characters match, 1 where they differ) and read `.distance`. A low `temperature` approaches the exact integer distance.

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

def char_ids(s):
    return torch.tensor([ord(c) for c in s], dtype=torch.long, device=device)

a, b = "kitten", "sitting"
ida, idb = char_ids(a), char_ids(b)             # [L1], [L2]

# Substitution cost: 0 where characters match, 1 where they differ.
equal = (ida[:, None] == idb[None, :]).float()  # [L1, L2]
sub_costs = (1.0 - equal).unsqueeze(0)          # [1, L1, L2]
sub_costs.requires_grad_(True)

# temperature -> 0 approaches the exact (hard) edit distance.
result = d2p.soft_levenshtein(sub_costs, ins_cost=1.0, del_cost=1.0, temperature=0.1)
print(result.distance, result.alignment.shape)  # ~[3.0], [1, 6, 7]

# .distance is a differentiable fuzzy-match loss (kitten -> sitting costs 3).
result.distance.sum().backward()
print(sub_costs.grad.shape)                      # [1, 6, 7]
```

`distance` is the soft edit distance; here `kitten → sitting` gives `3.0`, matching the classic Levenshtein distance, and it is differentiable w.r.t. `sub_costs` (and the `ins_cost`/`del_cost` scalars, if you pass them as tensors).

`soft_osa` adds restricted adjacent transpositions. Its `trans_mask` has the **same shape as `sub_costs`** (`[B, L1, L2]`) and is nonzero where swapping two adjacent characters is valid, so a real transposition costs one edit instead of two substitutions:

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

def char_ids(s):
    return torch.tensor([ord(c) for c in s], dtype=torch.long, device=device)

a, b = "ca", "ac"
ida, idb = char_ids(a), char_ids(b)
sub_costs = (1.0 - (ida[:, None] == idb[None, :]).float()).unsqueeze(0)  # [1, L1, L2]
sub_costs.requires_grad_(True)

# trans_mask is [B, L1, L2] (SAME shape as sub_costs), nonzero where an adjacent
# swap is valid: a[i] == b[j-1] and a[i-1] == b[j].
trans_mask = torch.zeros_like(sub_costs)
swap = (ida[1:, None] == idb[None, :-1]) & (ida[:-1, None] == idb[None, 1:])
trans_mask[:, 1:, 1:] = swap.float()

osa = d2p.soft_osa(sub_costs, trans_mask, ins_cost=1.0, del_cost=1.0,
                   trans_cost=1.0, temperature=0.1)
lev = d2p.soft_levenshtein(sub_costs.detach(), temperature=0.1)
print(osa.distance, lev.distance)   # OSA ~1.0 (one transposition) vs Levenshtein ~1.9
osa.distance.sum().backward()
print(sub_costs.grad.shape)                      # [1, 2, 2]
```

For `ca → ac`, OSA reports `~1.0` (a single transposition) versus `~1.9` for plain Levenshtein, which must pay ~2 for two substitutions. Note the transposition shape convention: `soft_osa` takes `trans_mask [B, L1, L2]`, whereas `soft_damerau` takes `trans_src [B, L1, L2, 2]` (`int32`).

---

See also: [Usage Guide](usage.md) | [Algorithm guides](algorithms/) | [Performance](performance.md)
