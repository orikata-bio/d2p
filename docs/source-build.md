# Source Builds

Building from source is currently the only way to install `d2p`, as prebuilt wheels are not published yet. Use these steps for the CPU-only or CUDA lane, an editable checkout, or a custom toolchain.

## Prerequisites

- Python `3.10` through `3.14`
- PyTorch `2.0+` installed in the active environment before building `d2p`
- A normal C++ toolchain
- `nvcc` only if you are building the CUDA lane
- The build backend when you use `--no-build-isolation` (below): `pip install meson-python meson ninja`

`d2p` builds its extension against the active PyTorch installation, so the PyTorch lane in your environment is the contract that matters.

## CPU-only source build

No CUDA toolkit is required for the CPU-only lane.

```bash
git clone https://github.com/orikata-bio/d2p.git
cd d2p
python -m pip install . --no-build-isolation --config-settings=setup-args=-Dcuda=disabled
```

## CUDA source build

For CUDA builds, install a matching PyTorch CUDA lane first, then make `nvcc` available from either a system CUDA toolkit or an environment-local package such as `cuda-nvcc`.

```bash
git clone https://github.com/orikata-bio/d2p.git
cd d2p
python -m pip install . --no-build-isolation --config-settings=setup-args=-Dcuda=enabled
```

## CUDA architectures

By default a CUDA build targets only your machine's GPU (`d2p_cuda_arch_mode=native`), which is the fastest to compile. Select a different policy with meson setup args:

```bash
python -m pip install . --no-build-isolation \
  --config-settings=setup-args=-Dcuda=enabled \
  --config-settings=setup-args=-Dd2p_cuda_arch_mode=release-cu130
```

- `d2p_cuda_arch_mode`: `native` (default, your GPU only), `release-cu128`, `release-cu130` (fat builds spanning sm_75 through sm_121), or `custom`.
- `d2p_cuda_gencode`: with `-Dd2p_cuda_arch_mode=custom`, a semicolon-separated NVCC gencode list, for example `arch=compute_80,code=sm_80`.
- The `D2P_CUDA_ARCH_LIST` (or `TORCH_CUDA_ARCH_LIST`) environment variable overrides the arch list at build time.

## Editable installs

Editable installs use the same lane selection:

```bash
python -m pip install -e . --no-build-isolation --config-settings=setup-args=-Dcuda=disabled
python -m pip install -e . --no-build-isolation --config-settings=setup-args=-Dcuda=enabled
```

## Notes

- CPU-only artifacts import without CUDA libraries present, but they do not provide CUDA tensor dispatch.
- CUDA builds link against the active PyTorch CUDA runtime; `nvcc` is the extra build-time requirement.
- Prebuilt wheels are not published yet. When they ship, the lane-specific names will be `d2p-cpu`, `d2p-cu128`, and `d2p-cu130`; the plain `d2p` name is the source distribution and source-build path.
