# Eisner

`d2p.soft_eisner(...)` provides differentiable projective dependency parsing. Use it when you want soft arc marginals over the set of projective trees, the temperature-controlled analogue of running Eisner's algorithm and reading off which head → dependent arcs the model believes in.

## What it computes

A dependency parse links every token to a single head; position `0` is the artificial ROOT. A tree is **projective** when no two arcs cross once the tokens are laid out in order. Eisner's algorithm (Eisner, 1996) sums or optimizes over all projective trees in `O(N³)` by combining *complete* and *incomplete* spans instead of enumerating trees.

`d²p` runs the **soft inside pass** of that recurrence: every span combination is scored with a temperature-scaled `logsumexp` rather than a `max`. The outputs are:

- `score`: the log partition function over all projective trees, and
- `marginals`: the posterior probability of each arc, recovered as the gradient of `score` (inside-outside).

## Soft inside recurrence

Eisner's chart holds two item types per span and direction: *incomplete* items, where the two endpoints are joined by one arc and the dependent side may still take further dependents outward, and *complete* items, where the head at one end has gathered every dependent inside the span. With temperature `T`, items are combined by a `logsumexp` over split points, adding `arc_scores[h, d]` whenever an arc is introduced:

```
# ATTACH  (complete + complete -> incomplete), introduces arc h -> d:
incomplete[i, j] = arc_scores[h, d] + T · logsumexp_{i ≤ k < j} ( complete[i, k] + complete[k+1, j] )
# COMPLETE (incomplete + complete -> complete):
complete[i, j]   = T · logsumexp_k ( incomplete[i, k] + complete[k, j] )
score            = complete item spanning ROOT … N-1
```

`T → 0` recovers the single best (Viterbi) projective tree; `T = 1` is the standard partition function; large `T` flattens the arc distribution.

## Inputs

- `arc_scores`: `[B, N, N]`. `arc_scores[b, h, d]` is the score for an arc from head `h` to dependent `d`. Position `0` is the ROOT by convention, so `arc_scores[b, 0, d]` scores "`d` attaches to ROOT". Self-arcs (`h == d`) and arcs into ROOT (`d == 0`) are not part of any tree and are ignored.
- `temperature`: scalar (or 1-element tensor).
- `lengths`: optional `[B]` int32 tensor of valid positions per sentence (ROOT included). Positions beyond `lengths[b]` are treated as padding, excluded from the chart, and receive zero marginal.

## Outputs

`soft_eisner` returns `EisnerResult`:

- `score`: log partition function `[B]`.
- `marginals`: arc marginals `[B, N, N]`. `marginals[b, h, d]` is the posterior probability of the arc `h → d`, and equals `∂ score / ∂ arc_scores`.

Because every non-ROOT token has exactly one head in any tree, the incoming arc mass for each dependent sums to `1`:

```python
import torch
import d2p

arc_scores = torch.randn(2, 11, 11)
result = d2p.soft_eisner(arc_scores, temperature=1.0)

result.marginals.sum(dim=1)    # [B, N]; == 1 for every real dependent, 0 for ROOT
```

## Example

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

# 10-word sentence + ROOT at position 0
arc_scores = torch.randn(2, 11, 11, device=device, requires_grad=True)

result = d2p.soft_eisner(arc_scores, temperature=1.0)
result.score.sum().backward()
# result.score      -> [2]
# result.marginals  -> [2, 11, 11]
```

## Worked example

```python
import torch
import d2p

torch.manual_seed(3)
N = 6                                  # ROOT + 5 tokens
arc_scores = torch.randn(1, N, N)

result = d2p.soft_eisner(arc_scores, temperature=1.0)
m = result.marginals[0]                # [N, N], m[h, d] = P(arc h -> d)

print(m.sum(dim=0).round(decimals=2))  # incoming mass per dependent
# tensor([0., 1., 1., 1., 1., 1.])
print(m[:, 1:].argmax(dim=0).tolist()) # most likely head of tokens 1..5
# [0, 1, 2, 0, 0]
```

Each real token (`d = 1…5`) collects exactly `1.0` of incoming arc mass, while ROOT (`d = 0`) collects none, since it is nobody's dependent. The `argmax` row reads off the single most probable head per token: tokens `1`, `4`, and `5` attach to ROOT (head `0`), token `2` attaches to token `1`, and token `3` to token `2`, giving a plausible projective skeleton (`ROOT → 1 → 2 → 3`, plus `ROOT → 4` and `ROOT → 5`). A structured loss would push on the full `marginals` matrix, not just its argmax.

## lengths and padding

Pad a batch to a common `N` and pass per-sentence lengths so the shorter parses ignore their padding:

```python
import torch
import d2p

arc_scores = torch.randn(2, 8, 8, requires_grad=True)
lengths = torch.tensor([8, 5], dtype=torch.int32)   # 2nd sentence: ROOT + 4 tokens
result = d2p.soft_eisner(arc_scores, temperature=1.0, lengths=lengths)
print(result.marginals[1].sum(dim=0).round(decimals=2).tolist())
# [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]   # positions 5,6,7 are padding
```

## Learnable temperature

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
arc_scores = torch.randn(2, 8, 8, requires_grad=True)

eisner = d2p.SoftEisner(temperature=1.0, learnable=True).to(device)
out = eisner(arc_scores.to(device))
out.score.sum().backward()
print(eisner.temperature.grad)         # temperature receives a gradient
```

## Common pitfalls

- **Head/dependent order.** `arc_scores[b, h, d]` and `marginals[b, h, d]` are indexed head-first. To get each token's distribution over *heads*, index or sum along the head axis (`dim=1` on the batched tensor, `dim=0` on a single `[N, N]` matrix).
- **ROOT is position 0.** Keep ROOT in slot `0`; the operator forces zero mass on self-arcs and on arcs into ROOT, so scores placed there are silently dropped.
- **Projective only.** Crossing (non-projective) arcs are not in the support; if your data contains them, Eisner marginals cannot model them.
- **`lengths` counts ROOT** and must be int32. `lengths[b] = k` means positions `0 … k-1` are valid (ROOT plus `k-1` tokens).

## Complexity

`O(N³)` time and `O(N²)` space, the standard cost of Eisner's span-combination chart: `O(N²)` spans, each combined over up to `N` split points, with the chart itself held in `O(N²)` memory.

## Background

See Eisner (1996), *Three New Probabilistic Models for Dependency Parsing*, for the projective span-combination recurrence; `d²p` computes its soft (log-sum-exp) inside pass and reads arc marginals off the gradient of the log partition function.

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
