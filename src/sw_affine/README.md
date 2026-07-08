# Soft Smith-Waterman (Affine Gap)

Differentiable local sequence alignment with affine gap penalty.

## Algorithm

Affine gap Smith-Waterman extends linear-gap SW with separate costs for
opening and extending gaps. This better models biological sequences where
starting a gap is more costly than extending one.

### Gap Cost Model

```
Linear:  cost = gap * length
Affine:  cost = gap_open + gap_ext * (length - 1)
```

### State Machine

Three states track whether we're in a gap:

```
M[i,j] = score ending with seq1[i] aligned to seq2[j]
I[i,j] = score ending with gap in seq2 (insertion)
D[i,j] = score ending with gap in seq1 (deletion)

           +--gap_ext--+
           v           |
  +---+  gap_open  +---+---+
  | M |----------->| I |   |
  +---+<-----------+---+   |
    |    (free)            |
    |                      |
    | gap_open             |
    v                      |
  +---+<-------------------+
  | D |----gap_ext---------+
  +---+
```

### Recurrence

```
M[i,j] = scores[i,j] + LSE_T(M[i-1,j-1], I[i-1,j-1], D[i-1,j-1], 0)

I[i,j] = LSE_T(M[i-1,j] + gap_open, I[i-1,j] + gap_ext)

D[i,j] = LSE_T(M[i,j-1] + gap_open, D[i,j-1] + gap_ext)
```

### Partition Function

```
S = LSE_T(M[i,j], I[i,j], D[i,j] for all i,j)
```

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
| `forward` | Compute 3-state alpha tables and partition function | O(L1 * L2) |
| `backward` | Compute posteriors, dS/dgap_open, dS/dgap_ext, dS/dT | O(L1 * L2) |
| `hvp` | Hessian-vector product d^2S/dscores^2 * V | O(L1 * L2) |
| `param_grad` | Parameter Jacobian dP/d{gap_open,gap_ext,T} | O(L1 * L2) |

## Memory Layout

```
Alpha table: [B, 3*(L1+1)*(L2+1)] with 3 states

State indexing:
  cell_stride = (L1+1) * (L2+1)
  M[b,i,j] = alpha[b * 3 * cell_stride + 0 * cell_stride + i*(L2+1) + j]
  I[b,i,j] = alpha[b * 3 * cell_stride + 1 * cell_stride + i*(L2+1) + j]
  D[b,i,j] = alpha[b * 3 * cell_stride + 2 * cell_stride + i*(L2+1) + j]
```

## Usage

```python
import d2p

# High-level API
result = d2p.soft_sw_affine(scores, gap_open=-2.0, gap_ext=-0.5, temperature=1.0)
# result.value: [B] soft alignment scores
# result.marginals: [B, L1, L2] soft alignment matrix

# Low-level API
value, marginals = d2p.sw_affine.soft_sw_affine_forward(
    scores, gap_open, gap_ext, temp, lengths
)
```

## Comparison with Linear Gap

| Aspect | Linear Gap | Affine Gap |
|--------|------------|------------|
| Parameters | 1 (gap) | 2 (gap_open, gap_ext) |
| States | 1 | 3 (M, I, D) |
| Memory | O(L1 * L2) | O(3 * L1 * L2) |
| Biological realism | Lower | Higher |

## See Also

- `../sw/` - Linear gap penalty version (1-state DP)
- `../common/` - Shared numerical utilities
