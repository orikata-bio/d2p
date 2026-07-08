# Concepts

d²p turns classic dynamic-programming algorithms (sequence alignment, parsing, edit distance) into differentiable PyTorch operators you can drop into a neural network and train end to end. This page is the mental model behind them: what "differentiable dynamic programming" means, what every operator gives back, how temperature shapes it, and how to choose an operator. It is background, not a reference. For the concrete API and per-operator signatures see the [Usage Guide](usage.md) and the [algorithm guides](algorithms/).

## Differentiable dynamic programming

Dynamic programming (DP) solves a structured combinatorial problem (the best alignment of two sequences, the highest-scoring parse tree, the minimum edit distance) by filling a table with a recurrence that takes a `max` (or `min`) over a few predecessors at each cell. The output is a single optimal structure and its score.

That structured score is often exactly what you want *inside* a larger model. If a network emits a matrix of pairwise similarities, an alignment DP turns it into a trainable signal: "how well do these two sequences match, and which parts line up?"

The trouble is that classic DP is not differentiable: `max` and `argmax` have zero or undefined gradients, so you cannot backpropagate through the recurrence to teach the network that produced the scores.

Differentiable DP fixes this by *smoothing* the recurrence, so the whole computation (from input scores to output score and structure) becomes a differentiable function. You can then use a DP value as a loss, or a soft alignment as an attention-like intermediate, and let gradients flow back into the upstream network. That learnable structured objective is the whole point.

Concretely: suppose a network embeds two sequences and you form a score matrix from pairwise similarities of the embeddings. Feeding that matrix to `soft_sw` yields a single differentiable alignment score. Maximising it over training data pushes the embeddings to agree exactly where the two sequences should line up, and it does so without ever needing a hand-labelled alignment, because the operator sums over all alignments rather than committing to one. The same idea underlies structured losses across bioinformatics, speech, and NLP.

## Hard DP vs soft DP

The only change is at the `max`. Hard DP propagates the single best predecessor at each cell:

- value: `max` over the candidate transitions
- decision: `argmax`, a hard 0/1 selection

Soft DP replaces both with temperature-scaled smooth analogues:

- value: `T · log-sum-exp(candidates / T)`, a soft maximum
- decision: `softmax(candidates / T)`, a soft, fractional selection

`log-sum-exp` is a smooth upper bound on `max`, and `softmax` is its gradient. As a feel for the difference: given two candidate scores `3` and `1`, hard DP keeps `3` and discards the rest; soft DP at `T = 1` returns `log(e³ + e¹) ≈ 3.13`, a value that still registers the runner-up, and whose gradient splits the credit roughly `88% / 12%` between the two. This particular smoothing is the natural one: it is the only choice whose gradient yields a coherent probability distribution over the candidate transitions, and it collapses back to ordinary `max` as `T → 0`. Applied at every cell of the recurrence, it makes the entire DP a smooth, differentiable function of the input scores.

Cost-based operators (lower is better) use the mirror image, a soft *minimum* built from the same `log-sum-exp`, yielding a smooth distance instead of a smooth score. The original hard structure is recovered in the low-temperature limit.

## What every operator returns

Every d²p operator returns two things.

1. **A scalar per batch element**: the smooth value of the DP, shape `[B]`. For score-based operators this is a **log-partition** value: a soft maximum (`log-sum-exp`) over the scores of *all* valid structures, not just the single best one. The result field is named for its meaning: `value` / `score` for score-based operators, `cost` / `distance` for cost-based ones.

2. **A soft assignment matrix**: the `marginals` / `alignment`. Its shape is `[B, L1, L2]` for the pairwise operators, `[B, N, N]` for Eisner arcs, and `[B, N, N, N]` for CKY merges (sum out the split axis for `[B, N, N]` span posteriors). Each entry says how much a given cell (a matched pair, a tree span, a dependency arc) participates across the smoothed distribution over structures.

The two are tightly linked, and this is the fact worth internalising:

> The soft matrix is exactly the gradient of the scalar value with respect to the input scores.

This is the standard log-partition ⇒ marginals identity, and in d²p it holds exactly:

```python
import torch, d2p

scores = torch.randn(2, 40, 50, requires_grad=True)
result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)

grad, = torch.autograd.grad(result.value.sum(), scores)
torch.allclose(grad, result.marginals)   # True
```

So the marginals are not a heuristic side-product for visualisation. They *are* the expected soft structure under the model: the posterior weight on each alignment cell, tree span, or dependency arc. Because of the identity above, computing them amounts to one backward pass through the recurrence, but you never have to run it yourself: the forward call returns both the value and the marginals together.

The smoothing is exact, not an approximation: the value and its marginals come from the same closed-form recurrence, so the gradient is exact to floating-point precision and needs only a single backward pass: no sampling of discrete structures and no high-variance estimator.

## Two ways to use the output

The two return values map onto the two ways operators tend to be used in a model, and you can use either or both:

- **The scalar as a differentiable structured objective.** Because the value is a smooth, differentiable score (or distance) summarising the *entire* space of structures, you can add it straight to a loss and backpropagate, teaching the upstream network to produce scores whose best alignment, parse, or edit is the one you want. This is the "structured loss" pattern.
- **The soft matrix as a differentiable alignment.** Because the marginals are the expected soft structure, they behave like a learned, structure-aware attention map: use them to pool one sequence against another, to supervise an alignment, or to read out which parts correspond. This is the "soft alignment / attention" pattern.

Both flow gradients back to the same input scores, so a single operator call can serve as loss, as attention, or as both at once.

## Temperature

A single `temperature` (`T`) controls how soft the smoothing is:

| Temperature | Value behaves like | Marginals |
| --- | --- | --- |
| `T → 0` | hard `max` / `min` | collapse toward the single best structure (hard argmax path) |
| `T = 1` | standard soft DP | calibrated soft posteriors |
| `T → ∞` | smoothest (adds a temperature-scaled entropy term) | spread out toward uniform |

Low temperature gives sharp, decisive marginals but stiff gradients; high temperature gives a smoother, easier optimisation landscape at the cost of a fuzzier structure.

`T = 1` is a sound default; a common training recipe starts warmer and anneals toward a lower temperature so the structure sharpens as learning settles. `temperature` is itself differentiable. Pass a tensor with `requires_grad=True` (or set a module's learn flag) to learn it alongside the network.

## The operator families

d²p groups its operators into three families. All of them expose the same batched, temperature-controlled, differentiable interface described above (a `[B, ...]` input matrix in, a `[B]` value plus a soft matrix out) and differ only in the shape of structure they search over.

**Alignment**: line up two sequences against each other. Each consumes a `[B, L1, L2]` similarity-or-cost matrix relating the two sequences and returns a soft alignment of the same shape.

- **Smith-Waterman** (`soft_sw`, `soft_sw_affine`): best *local* subsequence match; alignment mass can start and stop anywhere inside the matrix.
- **Needleman-Wunsch** (`soft_nw`, `soft_nw_affine`): *global* alignment covering both sequences end to end.
- **Dynamic Time Warping** (`soft_dtw`): monotone warping between sequences that advance at different speeds (time series, frame signals); cost-based.
- **Monotonic Alignment Search** (`soft_mas`): strictly monotonic frame→token alignment (TTS/ASR); requires frames ≥ tokens.

**Parsing**: recover latent tree structure over a single sequence. Each scores the pieces of that structure (span splits for CKY, head→dependent arcs for Eisner) and returns posteriors over them.

- **CKY** (`soft_cky`): binary *constituency* parsing; marginals are per-split (merge) posteriors `[B, N, N, N]` (sum the split axis for span posteriors).
- **Eisner** (`soft_eisner`): projective *dependency* parsing; marginals are arc posteriors.

**Edit distance**: the cost of transforming one string into another. Each takes a substitution-or-match matrix plus scalar per-operation costs (with an extra transposition cost for OSA and Damerau); all but LCS are cost-based.

- **Levenshtein** (`soft_levenshtein`): insertions, deletions, substitutions.
- **LCS** (`soft_lcs`): longest common subsequence; score-based (match scores, higher is better).
- **OSA** (`soft_osa`): Levenshtein plus *restricted* adjacent transpositions.
- **Damerau** (`soft_damerau`): Levenshtein plus *unrestricted* transpositions.

## Choosing an operator

A few axes decide most cases.

- **Local vs global alignment.** Do you want the single best matching sub-region (Smith-Waterman) or a full end-to-end alignment of both sequences (Needleman-Wunsch)? Reach for the affine gap variants when a long gap should cost less than many short ones.
- **Monotonic vs free.** If the alignment must advance strictly forward with no reordering (audio, video, time series), use DTW or MAS. If the transform is insertions, deletions, substitutions, and (for OSA and Damerau) transpositions, use the edit-distance family.
- **Constituency vs dependency.** For nested phrase structure use CKY; for head→dependent arcs use Eisner (projective trees only).
- **Cost vs score conventions.** Score-based operators (SW, NW, MAS, CKY, Eisner, LCS) treat *higher as better* and return a `value`/`score`; cost-based operators (DTW, Levenshtein, OSA, Damerau) treat *lower as better* and return a `cost`/`distance`. Feed them the matching sign. For example, negate a similarity matrix to use it as a cost.

Every operator shares the same result shape, temperature semantics, differentiability, and `nn.Module` wrapper, and most accept a `lengths` argument for batching variable-length inputs, so switching between them is mostly a matter of swapping the call. See the [Usage Guide](usage.md) for those cross-cutting details.

---

See also: [Usage Guide](usage.md) | [Algorithm guides](algorithms/) | [Examples](examples.md)
