# Dynamic Time Warping

`d2p.soft_dtw(...)` is a differentiable Dynamic Time Warping (DTW) operator with an optional Sakoe-Chiba band.

## What it computes

DTW measures the distance between two sequences that advance through time at different, non-uniform rates. Instead of comparing frame `i` of one sequence to frame `i` of the other, it finds a monotonic *warping path* from corner `(0, 0)` to corner `(L1-1, L2-1)` of the cost matrix that pairs up frames so the total cost along the path is minimal. The path may move right, down, or diagonally, so one frame of the first sequence can align to several consecutive frames of the second (many-to-one warping) and vice versa. This is what makes DTW *elastic*: it absorbs local stretching and compression of the time axis.

Classic DTW takes a hard `min` at each step, which is not differentiable. `soft_dtw` replaces `min` with a temperature-controlled `softmin`, so both the returned `cost` and the `alignment` field are smooth functions of the input costs and can be backpropagated through. You can learn the feature extractor that produces the costs, or the temperature itself.

## The soft recurrence

Let `D[i, j]` be the accumulated soft cost of the best warp reaching cell `(i, j)`. With softmin at temperature `T`:

```
D[i, j] = costs[i, j] + softmin_T( D[i-1, j-1], D[i-1, j], D[i, j-1] )
softmin_T(a, b, c) = -T * log( exp(-a/T) + exp(-b/T) + exp(-c/T) )
```

- `cost` is `D[L1-1, L2-1]`, the soft cost of the full warping path.
- `alignment[b, i, j]` is the gradient of that cost with respect to `costs[b, i, j]`, the soft occupancy of each cell, i.e. how much the optimal warp passes through `(i, j)`.
- As `T -> 0`, `softmin` approaches `min`, `cost` approaches the hard DTW cost, and `alignment` concentrates on the single least-cost path. Larger `T` spreads mass across competing paths.

## Inputs

- `costs`: cost matrix `[B, L1, L2]`, float, where **lower = better** match. This is a distance, not a similarity, typically a pairwise distance such as `torch.cdist`. If you have a similarity, negate it first.
- `temperature`: softmin temperature (default `1.0`). Lower is sharper (closer to hard DTW); higher is smoother. A Python float or a `torch.Tensor` with `requires_grad=True` (learnable).
- `bandwidth`: Sakoe-Chiba band width (default `0`). `0` disables the constraint and scores the full matrix. A positive value restricts the warp to a diagonal window, forbidding cells too far off the diagonal, which prevents pathological warps and prunes the transitions considered.
- `lengths`: optional `[B, 2]` int32 tensor of true `(L1, L2)` per batch item, so padded positions are excluded.

## Outputs

`soft_dtw` returns a `DTWResult` named tuple:

- `cost`: soft DTW cost `[B]`.
- `alignment`: soft alignment matrix `[B, L1, L2]`, the per-cell occupancy of the warping path.

## Example

Align two feature trajectories of different lengths using a `cdist` cost matrix:

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

# Two feature trajectories sampled at different rates.
x = torch.randn(1, 12, 4, device=device)   # 12 frames, 4-d features
y = torch.randn(1, 16, 4, device=device)   # 16 frames, 4-d features

# Pairwise cost matrix: LOWER = better match (here, Euclidean distance).
costs = torch.cdist(x, y)                   # [1, 12, 16]

result = d2p.soft_dtw(costs, temperature=0.1, bandwidth=0)
print(result.cost)                          # [1] soft warping cost
print(result.alignment.shape)               # torch.Size([1, 12, 16])

# One slow frame of x can be matched by several frames of y (many-to-one warp):
occupancy = result.alignment[0].sum(dim=1)  # mass absorbed per x-frame
print(occupancy)                            # entries > 1 -> stretched in time
```

`result.cost` is the elastic distance between the two trajectories: use it directly as a training loss to pull matching sequences together. The `occupancy` vector shows where the warp stretches: an entry near `2` means that frame of `x` was matched by roughly two frames of `y`.

## Learnable temperature

`temperature` may be a tensor, so it can be optimized alongside the rest of a model:

```python
import torch
import d2p

costs = torch.cdist(torch.randn(1, 12, 4), torch.randn(1, 16, 4))  # [1, 12, 16]

temperature = torch.tensor(0.5, requires_grad=True)
result = d2p.soft_dtw(costs, temperature=temperature, bandwidth=8)
result.cost.sum().backward()
print(temperature.grad)                     # gradient flows into temperature
```

The `SoftDTW(temperature=..., bandwidth=..., learnable=True)` module wraps the same behaviour as an `nn.Module`; see the usage guide for the module API and AMP.

## Interpreting the alignment

`alignment` is the soft occupancy of the warping path, not a per-row distribution. A value near `1` means the warp passes through that cell, and a first-sequence frame whose row sums to more than `1` was held in place while the second sequence advanced, a local stretch. As `temperature -> 0` the mass collapses onto the single least-cost monotonic path; each cell on that path has occupancy near `1`, so a row or column can sum to more than `1` where the warp stretches (a held frame). `result.alignment[b].argmax(dim=0)` then gives, for each second-sequence frame, its highest-occupancy first-sequence index: a lossy readout of the path, not the full path.

## Common pitfalls

- **Costs, not similarities.** Lower must mean *more* similar. Feeding a similarity aligns the least-similar frames; negate it first.
- **Band width trade-off.** A positive `bandwidth` is faster and rejects wild warps, but if it is narrower than `|L1 - L2|` (plus slack) it can exclude the true path and inflate the cost. Use `0` for exact full-matrix DTW; widen the band until the cost stops changing.
- **Cost scale sets the effective temperature.** `softmin` acts on the ratio `costs / temperature`, so rescaling the cost matrix sharpens or softens the warp just like changing `temperature`. Normalise costs (or scale `temperature` with them) for comparable behaviour across batches.
- **`lengths` dtype.** Pass `lengths` as int32 `[B, 2]` when batching padded sequences.

## Complexity

The dynamic program is `O(B · L1 · L2)` in time and memory. A Sakoe-Chiba `bandwidth = w` restricts the warp to a diagonal window and prunes the transitions considered, but the current kernels still allocate the full `O(B · L1 · L2)` tables, so memory is unchanged and the practical speedup is implementation-dependent. On CPU each batch element runs on its own thread; on CUDA the batch runs in parallel.

## Low-level API

`d2p.ops` exposes the differentiable internals for second-order and sensitivity work, all mirroring the `soft_dtw` arguments:

- `soft_dtw_float` / `soft_dtw_with_grads`: forward, and forward with cached gradients.
- `soft_dtw_backward_full`: full backward through both `cost` and `alignment`.
- `soft_dtw_hvp`: Hessian-vector products.
- `soft_dtw_param_jacobian`: parameter Jacobians, e.g. `dP/dT`, the sensitivity of the alignment field to temperature.

## Background

DTW and the Sakoe-Chiba band come from Sakoe & Chiba (1978); the differentiable softmin formulation used here is soft-DTW (Cuturi & Blondel, 2017).

---

See also: [Usage Guide](../usage.md) (batching, masks, learnable parameters, modules, AMP, temperature) | [Performance](../performance.md).
