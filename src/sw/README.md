# Soft Smith-Waterman (Linear Gap)

Differentiable local sequence alignment with linear gap penalty.

## Algorithm

Smith-Waterman finds the optimal **local alignment** between two sequences,
meaning it finds the best matching subsequence that can start and end anywhere.

### Recurrence

```
alpha[i,j] = LSE_T(
    alpha[i-1,j-1] + scores[i,j],   // align: match positions
    alpha[i-1,j] + gap,              // up: gap in sequence 2
    alpha[i,j-1] + gap,              // left: gap in sequence 1
    0                                 // sky: start new alignment
)
```

The "sky" option (starting fresh with score 0) is what makes this **local**
alignment - the algorithm can abandon a poor alignment and restart.

### Partition Function

```
S = LSE_T(alpha[i,j] for all i,j)
```

The partition function is the logsumexp over all cells, representing the
soft maximum alignment score across all possible local alignments.

## Files

| File | Description |
|------|-------------|
| `kernels.cu` | CUDA kernels with wavefront parallelization |
| `kernels.cuh` | CUDA kernel declarations and algorithm documentation |
| `kernels_cpu.cpp` | CPU kernels with Kahan summation |
| `kernels_cpu.h` | CPU kernel declarations |

## Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `forward` | Compute alpha table and partition function | O(L1 * L2) |
| `backward` | Compute posteriors (dS/dscores), dS/dgap, dS/dT | O(L1 * L2) |
| `hvp` | Hessian-vector product d^2S/dscores^2 * V | O(L1 * L2) |
| `param_grad` | Parameter Jacobian dP/d{gap,T} | O(L1 * L2) |

## Memory Layout

```
Alpha table: [B, (L1+1) * (L2+1)] flattened row-major
  Index: alpha[b, i, j] = alpha[b * stride + i * (L2+1) + j]

Scores: [B, L1, L2] standard row-major
  Index: scores[b, i, j] = scores[b * L1 * L2 + i * L2 + j]
```

## CUDA Parallelization

Uses wavefront (anti-diagonal) parallelization:

```
    j=0  j=1  j=2  j=3
  +----+----+----+----+
  | d0 | d1 | d2 | d3 |  i=0
  +----+----+----+----+
  | d1 | d2 | d3 | d4 |  i=1
  +----+----+----+----+
  | d2 | d3 | d4 | d5 |  i=2
  +----+----+----+----+
```

Cells on the same anti-diagonal are independent and computed in parallel.

## Usage

```python
import d2p

# High-level API
result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)
# result.value: [B] soft alignment scores
# result.marginals: [B, L1, L2] soft alignment matrix

# Low-level API
value, marginals = d2p.sw.soft_sw_forward(scores, gap, temp, lengths)
```

## See Also

- `../sw_affine/` - Affine gap penalty version (3-state DP)
- `../common/` - Shared numerical utilities
