# Performance

## Complexity

Per batch element, in terms of the pairwise input lengths `L1`, `L2`, the parser sequence length `N`, or the MAS frame/token counts `T`/`S`:

| Operator | Time | Space |
| --- | --- | --- |
| `soft_sw`, `soft_sw_affine` | `O(L1·L2)` | `O(L1·L2)` |
| `soft_nw`, `soft_nw_affine` | `O(L1·L2)` | `O(L1·L2)` |
| `soft_dtw` | `O(L1·L2)` | `O(L1·L2)` |
| `soft_levenshtein`, `soft_lcs`, `soft_osa`, `soft_damerau` | `O(L1·L2)` | `O(L1·L2)` |
| `soft_mas` | `O(T·S)` | `O(T·S)` |
| `soft_cky` | `O(N³)` | `O(N³)` |
| `soft_eisner` | `O(N³)` | `O(N²)` |

Space counts the DP working memory plus the returned field; CKY's `marginals` is `[B, N, N, N]`, so its public-surface memory is `O(N³)`.

The full batch is processed in parallel (across CUDA blocks on GPU, across threads on CPU), so wall-clock time is dominated by the per-element cost above rather than the batch size `B`, up to available hardware parallelism.

## CPU execution

- **Batch-parallel:** each batch element's dynamic program runs on its own thread via PyTorch's `at::parallel_for`. Throughput scales with the intra-op thread count (`torch.get_num_threads()`, `OMP_NUM_THREADS`).
- A single sequence (`B = 1`) runs single-threaded: the parallelism is *across* the batch, not within one DP. Batch your work to use all cores.
- Inside a DataLoader worker (or with threads pinned to 1) the operators run serially, as expected.

## GPU execution

- CUDA device code covers NVIDIA architectures from **Turing (`sm_75`)** through **Blackwell (`sm_121`)**, selectable at build time (a fat build ships several; a native build targets just your GPU; see [source-build.md](source-build.md)).
- Every operator sets a CUDA device guard, so multi-GPU use dispatches to the input tensor's device correctly.

## Numerical stability and safety

- Inputs must be **float32**; the operators are not autocast-eligible (a `float16`/`bfloat16` input raises), so keep d2p inputs in float32 inside an autocast region.
- The DP recurrences accumulate in **FP32 internally**, even under `torch.autocast`, for stability across the soft-max/log-sum-exp steps.
- Kernels are memory-safety hardened: length/shape/device validation at every op boundary, `size_t`-widened indexing to avoid overflow on large inputs, and shared-reduction hazards fixed.

## Tips

- **Batch aggressively**: it is the primary source of parallelism on both CPU and GPU.
- **Constrain when you can**: `soft_dtw(..., bandwidth=w)` (Sakoe-Chiba band) prunes the transitions considered and rules out pathological warps (memory still `O(L1·L2)`).
- **Pick temperature deliberately**: very small temperatures sharpen marginals but stiffen gradients; see [usage.md](usage.md#temperature).
- **Parsers are cubic**: `soft_cky`/`soft_eisner` are `O(N³)`; keep `N` modest or budget accordingly.
