# Monotonic Alignment Search

`d2p.soft_mas(...)` is a differentiable Monotonic Alignment Search (MAS) operator for strictly monotonic sequence-to-sequence alignment.

## What it computes

MAS aligns a short symbolic sequence of `S` tokens to a longer sequence of `T` frames under a hard monotonicity constraint: as you move forward in time (frames), the token index never decreases, and every token is used by at least one frame. Concretely, each frame is assigned to exactly one token, consecutive frames stay on the same token or advance to the next one, and the tokens are consumed strictly in order. This is the alignment TTS models such as Glow-TTS use to map text tokens to mel-spectrogram frames without an external forced aligner: the per-token frame counts become predicted *durations*.

Because a token can never be skipped and frames are consumed one per step, a valid alignment exists only when there are at least as many frames as tokens, so **`T >= S` is required**. The hard search takes an `argmax` over monotonic paths; `soft_mas` replaces it with a temperature-controlled log-sum-exp, making the returned `score` and `alignment` differentiable so the scores that drive the alignment can be learned end to end.

## The soft recurrence

Working over a monotone lattice indexed by frame `t` and token `s`, the forward variable `A[t, s]` accumulates the soft score of all monotonic alignments that place frame `t` on token `s`:

```
A[t, s] = scores[t, s] + softmax_T( A[t-1, s], A[t-1, s-1] )
softmax_T(a, b) = T * log( exp(a/T) + exp(b/T) )
```

The two incoming terms are *stay on token `s`* and *advance to token `s`*. Then:

- `score` is the soft-combined value in the final `(t = T-1, s = S-1)` corner, a temperature-smoothed log-sum-exp over the scores of all valid monotonic alignments. As `T -> 0` it approaches the score of the single best monotonic alignment.
- `alignment[b, t, s]` is the posterior probability that frame `t` is assigned to token `s`. Every frame's row sums to `1` (soft one-token-per-frame assignment), and each token's column sum is its expected duration in frames.

## Inputs

- `scores`: affinity tensor `[B, T, S]`, float, where **higher = better** (frame `t` matches token `s`). Requires `T >= S`.
- `temperature`: log-sum-exp temperature (default `1.0`). Lower is sharper (closer to hard MAS); higher is smoother. A Python float or a `torch.Tensor` with `requires_grad=True` (learnable).
- `lengths`: optional `[B, 2]` int32 tensor of true `(T, S)` (frames then tokens) per batch item, so padded positions are excluded.

In the common TTS/ASR setting, `T` is the frame (acoustic) axis and `S` is the text-token axis.

## Outputs

`soft_mas` returns a `MASResult` named tuple:

- `score`: soft alignment score `[B]`.
- `alignment`: soft monotonic alignment `[B, T, S]`, a posterior over token assignments per frame.

## Example

Align `S` tokens to `T` frames from an affinity between token and frame embeddings:

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

T, S = 40, 6                                  # 40 frames, 6 tokens (needs T >= S)
tokens = torch.randn(1, S, 8, device=device)  # token embeddings
frames = torch.randn(1, T, 8, device=device)  # frame embeddings

# scores[b, t, s] = affinity of frame t to token s (HIGHER = better).
scores = torch.einsum("btd,bsd->bts", frames, tokens)  # [1, 40, 6]

result = d2p.soft_mas(scores, temperature=0.3)
print(result.score)                           # [1] soft alignment score
print(result.alignment.shape)                 # torch.Size([1, 40, 6])

# Each frame is softly assigned to exactly one token (its row sums to 1),
# so the column sums are the expected per-token durations:
durations = result.alignment[0].sum(dim=0)    # [6], sums to T
print(durations)
```

`durations` reads out how many frames each token occupies (the entries sum to `T`), which is exactly the target a TTS duration predictor is trained to reproduce. Lowering `temperature` sharpens the alignment toward a hard, block-diagonal path; raising it blurs the token boundaries.

## Learnable temperature

`temperature` may be a tensor so it can be optimized with the model:

```python
import torch
import d2p

torch.manual_seed(0)
scores = torch.randn(1, 40, 6)                # [B, T, S] with T >= S

temperature = torch.tensor(0.5, requires_grad=True)
out = d2p.soft_mas(scores, temperature=temperature)
out.score.sum().backward()
print(temperature.grad)                       # gradient flows into temperature
```

The `SoftMAS(temperature=..., learnable=True)` module wraps the same behaviour as an `nn.Module`; see the usage guide for the module API and AMP.

## Common pitfalls

- **`T >= S` is required.** If you pass fewer frames than tokens, `soft_mas` (and `SoftMAS`) raises `ValueError`; when you pass `lengths`, each row must satisfy `T_len >= S_len`. Always ensure the frame axis is at least as long as the token axis.
- **Scores, not costs.** Higher must mean a better match, the opposite of DTW costs. Negate a cost matrix before passing it.
- **`lengths` order and dtype.** `lengths` is int32 `[B, 2]` ordered `(frames, tokens)`; per item the frame length must still be `>=` the token length.

## Reading out the alignment

Because each frame's row is a distribution over tokens, two common readouts fall straight out of `alignment`:

- `result.alignment.argmax(dim=-1)`: the hard token id assigned to each frame `[B, T]`.
- `result.alignment.sum(dim=1)`: the expected per-token durations `[B, S]`, the target for a TTS duration predictor.

## Complexity

The dynamic program is `O(B · T · S)` in time and memory. On CPU each batch element runs on its own thread; on CUDA the batch runs in parallel.

## Low-level API

`d2p.ops` exposes the differentiable internals, all mirroring the `soft_mas` arguments:

- `soft_mas_float` / `soft_mas_with_grads`: forward, and forward with cached gradients.
- `soft_mas_backward_full`: full backward through both `score` and `alignment`.
- `soft_mas_hvp`: Hessian-vector products.
- `soft_mas_param_jacobian`: parameter Jacobians of the posterior field, e.g. its sensitivity to temperature.

## Background

MAS was introduced in Glow-TTS (Kim et al., 2020) as a hard, monotonic text-to-speech aligner; this is the differentiable soft relaxation of that search.

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
