<p align="center">
  <img src="assets/d2p_logo.svg" width="200"><br>
  <strong>Differentiable Dynamic Programming for PyTorch</strong><br>
  CPU and CUDA operators for alignment, parsing, and edit-distance style workloads.
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=ZPtK8LKcvi"><img src="https://img.shields.io/badge/Paper-ICML%202026-b31b1b.svg" alt="Paper: ICML 2026"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg" alt="PyTorch 2.0+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

d2p provides differentiable dynamic programming operators for PyTorch across sequence alignment, parsing, monotonic alignment, and edit-distance families. The public API is high-level first (`d2p.soft_sw`, `d2p.soft_dtw`, `d2p.soft_cky`, ...), with matching `nn.Module` wrappers, `torch.compile` support, and low-level operators under `d2p.ops`.

## Installation

> **d2p is not yet published to PyPI or conda.** For now, build from source (below). Prebuilt binary wheels (`d2p-cpu` / `d2p-cu128` / `d2p-cu130` lanes) are planned for a future release.

Install a matching PyTorch build first, then build d2p from source. The extension compiles against your installed PyTorch, so build with `--no-build-isolation` (after installing the build backend):

```bash
pip install meson-python meson ninja        # build backend
git clone https://github.com/orikata-bio/d2p.git
cd d2p
pip install . --no-build-isolation --no-cache-dir   # builds against your installed PyTorch
```

`--no-build-isolation` is required, not optional. The extension registers ops
through `TORCH_LIBRARY` and is locked to the exact PyTorch it compiles against,
because `libc10`/`libtorch` symbol versions change every PyTorch minor release.
Under PEP 517 build isolation pip installs its *own* (latest) PyTorch into a
temporary build environment and compiles against that instead of yours; the
install then reports success and the mismatch only appears later as an
`undefined symbol` error on `import d2p`.

`--no-cache-dir` matters for the same reason: pip keys its wheel cache on the
source distribution, not on the PyTorch it was built against, so an incompatible
wheel built earlier is silently reused even after you correct the command.

Requirements:

- Python `3.10` through `3.14`
- PyTorch `2.0+` (install the CPU or CUDA build for your platform first)
- CUDA builds also need a CUDA toolkit matching your PyTorch CUDA version

CUDA builds compile device code for NVIDIA architectures from Turing (`sm_75`) through Blackwell (`sm_121`), selectable at build time. For editable installs, architecture selection, and custom toolchains, see [docs/source-build.md](docs/source-build.md).

## Quick Start

```python
import torch
import d2p

device = "cuda" if torch.cuda.is_available() else "cpu"
scores = torch.randn(2, 100, 120, device=device, requires_grad=True)

result = d2p.soft_sw(scores, gap=-1.0, temperature=1.0)

print(result.value.shape)      # [2]
print(result.marginals.shape)  # [2, 100, 120]

loss = result.value.sum()
loss.backward()
```

The same high-level calls work under `torch.compile` on supported PyTorch `2.x` builds.

## Algorithm Guides

d2p ships 12 operators across seven guide families:

| Family | Public APIs | Guide |
| --- | --- | --- |
| Smith-Waterman | `soft_sw`, `soft_sw_affine` | [docs/algorithms/sw.md](docs/algorithms/sw.md) |
| Needleman-Wunsch | `soft_nw`, `soft_nw_affine` | [docs/algorithms/nw.md](docs/algorithms/nw.md) |
| Dynamic Time Warping | `soft_dtw` | [docs/algorithms/dtw.md](docs/algorithms/dtw.md) |
| CKY | `soft_cky` | [docs/algorithms/cky.md](docs/algorithms/cky.md) |
| Monotonic Alignment Search | `soft_mas` | [docs/algorithms/mas.md](docs/algorithms/mas.md) |
| Eisner | `soft_eisner` | [docs/algorithms/eisner.md](docs/algorithms/eisner.md) |
| Edit distance | `soft_levenshtein`, `soft_lcs`, `soft_osa`, `soft_damerau` | [docs/algorithms/edit-distance.md](docs/algorithms/edit-distance.md) |

**More docs:** [Concepts](docs/concepts.md) (differentiable-DP background) | [Usage Guide](docs/usage.md) (batching, masks, learnable params, modules, AMP, temperature) | [Examples](docs/examples.md) (end-to-end tutorials) | [Performance](docs/performance.md) (complexity, threading, arch) | [FAQ](docs/faq.md) | [Source builds](docs/source-build.md)

## What You Get

- High-level result objects with named fields such as `.value`, `.marginals`, `.score`, `.alignment`, or `.distance`, depending on family.
- `nn.Module` wrappers such as `d2p.SoftSW`, `d2p.SoftDTW`, and `d2p.SoftCKY`.
- Low-level operator access through `d2p.ops` plus the legacy `torch.ops.d2p.*` surface.
- Gradients through inputs, scalar hyperparameters, and Hessian-vector products for the shipped families.
- First-order parameter-Jacobian APIs where the family exposes them.
- Multithreaded CPU kernels: each batch element's dynamic program runs in parallel across PyTorch's intra-op threads (`torch.set_num_threads`).
- Memory-safety-hardened CUDA and CPU kernels: length/shape/device validation at every op boundary, `size_t`-widened indexing, and multi-GPU device guards.

## API Levels

- High-level functions: `d2p.soft_sw(...)`, `d2p.soft_dtw(...)`, `d2p.soft_cky(...)`, ...
- Module wrappers: `d2p.SoftSW(...)`, `d2p.SoftMAS(...)`, `d2p.SoftLevenshtein(...)`, ...
- Low-level ops: `d2p.ops` for structured access, plus legacy compatibility re-exports from `d2p._ops`

See the [Usage Guide](docs/usage.md) for each level in depth: learnable parameters, `nn.Module` wrappers, AMP, temperature, and the low-level derivative surfaces (`_with_grads`, `_hvp`, `_param_jacobian`).

## Source Builds

For editable checkouts, per-architecture CUDA builds, or a custom CUDA toolchain, see [docs/source-build.md](docs/source-build.md).

## Scope Notes

`0.1.0` ships the stable public package surface: forward ops, autograd, HVPs, and first-order parameter Jacobians where exposed by the family API. Some broader paper-side or mixed-derivative surfaces are intentionally not part of the `0.1.0` public API yet.

## Citation

d2p accompanies the paper **[d²p: Structured Soft Attention Is All You Need](https://openreview.net/forum?id=ZPtK8LKcvi)** (ICML 2026). If you use d2p in your research, please cite it.

- OpenReview: <https://openreview.net/forum?id=ZPtK8LKcvi>
- ICML 2026 poster: <https://icml.cc/virtual/2026/poster/63172>

```bibtex
@inproceedings{mogilevsky2026d2p,
  title     = {{d\textsuperscript{2}p: Structured Soft Attention Is All You Need}},
  author    = {Mogilevsky, Casey Sumagaysay and Liang, Kimberly},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  year      = {2026},
  publisher = {PMLR},
  url       = {https://openreview.net/forum?id=ZPtK8LKcvi},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
