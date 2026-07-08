# CKY

`d2p.soft_cky(...)` provides differentiable inside-style CKY parsing for binary constituency structure. It replaces the hard search over parse trees with a temperature-controlled `logsumexp`, so both the log partition function and the full field of chart marginals are smooth, differentiable functions of the input scores.

## What it computes

CKY (Cocke-Kasami-Younger) is the classic chart parser for a context-free grammar in binary (Chomsky) normal form. Over a length-`N` sequence there are `O(N²)` contiguous spans `(i, j)`; each span of length `≥ 2` is built by choosing a split point `k` that divides it into a left child `(i, k)` and a right child `(k+1, j)`.

`d²p` computes the **soft inside pass**: instead of taking the single best split for each span, it accumulates the exponentiated scores of *every* split. The outputs are:

- `score`: the log partition function, i.e. the (temperature-scaled) log-sum-exp over all binary trees spanning the sequence, and
- `marginals`: the posterior weight of every merge rule, recovered as the gradient of `score` (the inside-outside marginals).

Intuition: `score` measures how much total tree mass the chart supports, and the marginals tell you which constituents that mass flows through.

## Soft inside recurrence

With temperature `T`, the inside score of a span is, schematically:

```
inside[i, i] = leaf_scores[i]
inside[i, j] = T · logsumexp_{i ≤ k < j} ( ( merge_scores[i, k, j]
                                             + inside[i, k]
                                             + inside[k+1, j] ) / T )
score        = inside[0, N-1]
```

As `T → 0` the `logsumexp` becomes a hard `max` (Viterbi / single best tree); at `T = 1` it is the standard log partition; as `T → ∞` the distribution over splits flattens toward uniform.

## Inputs

- `merge_scores`: `[B, N, N, N]`. `merge_scores[b, i, k, j]` is the score for building span `(i, j)` by merging `(i, k)` and `(k+1, j)`. Only entries with `i ≤ k < j` are read; all other cells (the "invalid" splits) are ignored, so you may leave them at any value.
- `leaf_scores`: `[B, N]`. The terminal score for each position as a length-1 span.
- `temperature`: a scalar, or a per-span tensor `[B, N, N]` for position-dependent sharpening.

## Outputs

`soft_cky` returns `CKYResult`:

- `score`: log partition function `[B]`.
- `marginals`: merge (split-point) marginals `[B, N, N, N]`. `marginals[b, i, k, j]` is the posterior probability (expected count) that span `(i, j)` is built by the split at `k`. This field mirrors the shape of `merge_scores` and equals `∂ score / ∂ merge_scores`.

To get **span posteriors** (the probability that constituent `(i, j)` appears at all, regardless of where it splits), sum over the split axis:

```python
import torch
import d2p

merge_scores = torch.randn(2, 8, 8, 8)
leaf_scores  = torch.randn(2, 8)
result = d2p.soft_cky(merge_scores, leaf_scores, temperature=1.0)

span_post = result.marginals.sum(dim=2)   # [B, N, N]; span_post[b, i, j] = P(span (i, j))
```

The full-sentence span `(0, N-1)` always has posterior `1.0`, and every length-1 leaf is present in every tree (leaf marginals are trivially `1`).

## Example

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"

batch, length = 2, 8
merge_scores = torch.randn(batch, length, length, length, device=device, requires_grad=True)
leaf_scores  = torch.randn(batch, length, device=device, requires_grad=True)

result = d2p.soft_cky(merge_scores, leaf_scores, temperature=1.0)
result.score.sum().backward()          # gradients flow to both score tensors
# result.score      -> [2]
# result.marginals  -> [2, 8, 8, 8]
```

## Worked example

```python
import torch
import d2p

torch.manual_seed(0)
N = 5
merge_scores = torch.randn(1, N, N, N)   # invalid splits are ignored
leaf_scores  = torch.randn(1, N)

result = d2p.soft_cky(merge_scores, leaf_scores, temperature=1.0)
span_post = result.marginals.sum(dim=2)[0]   # [N, N] span posteriors
print(span_post.round(decimals=2))
# tensor([[0.0000, 0.0600, 0.1700, 0.2200, 1.0000],
#         [0.0000, 0.0000, 0.4200, 0.3100, 0.7000],
#         [0.0000, 0.0000, 0.0000, 0.5300, 0.4300],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.1500],
#         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
```

Read `span_post[i, j]` as *how confident the chart is that tokens `i…j` form one constituent*. Row `0`, column `4` is the root span `(0, 4)` at posterior `1.0`: the whole sentence is always a constituent. After the root the chart most favors `(1, 4)` at `0.70` (tokens 1 to 4 grouped, i.e. the root most likely splits right after token 0) and the nested `(2, 3)` at `0.53`. These posteriors are exactly the brackets a structured loss would push on. Everything on and below the diagonal is `0`: spans with `j ≤ i` are not constituents in this field.

## Learnable temperature

Temperature can be optimized jointly with the rest of a model. Either pass a tensor with `requires_grad=True` as `temperature`, or use the module wrapper:

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
merge_scores = torch.randn(2, 8, 8, 8, device=device)
leaf_scores  = torch.randn(2, 8, device=device)

cky = d2p.SoftCKY(temperature=1.0, learnable=True).to(device)
out = cky(merge_scores, leaf_scores)
out.score.sum().backward()
print(cky.temperature.grad)            # temperature receives a gradient
```

## Common pitfalls

- **`marginals` is `[B, N, N, N]`, not `[B, N, N]`.** The raw field is the per-*split* (merge) marginal, matching the shape of `merge_scores`. Sum over the split axis (`marginals.sum(dim=2)`) to recover `[B, N, N]` span posteriors.
- **Split convention.** `k` is the last index of the *left* child: children are `(i, k)` and `(k+1, j)`, and only `i ≤ k < j` is valid. Off-by-one here silently reshapes the chart.
- **`leaf_scores` do not change `marginals`.** Every tree over `N` tokens uses all `N` leaves exactly once, so `leaf_scores` shift `score` by a constant (the sum of the leaf scores) but cancel out of the tree distribution. Put structural signal in `merge_scores`.
- **Temperature must be positive**, and a per-span temperature must be exactly `[B, N, N]`.

## Complexity

`O(N³)` time (one `logsumexp` over up to `N` split points for each of `O(N²)` spans) and `O(N²)` DP working memory (the inside chart). Note the `merge_scores` input itself is `O(N³)` to materialize.

## Background

See any treatment of CKY chart parsing and the inside-outside algorithm for context-free grammars; `d²p` computes the soft (log-sum-exp) inside pass and recovers marginals by differentiating the log partition function.

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
