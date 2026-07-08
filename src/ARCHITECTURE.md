# Source Architecture

This document describes the modular architecture for D2P operators and serves as the pattern for adding new operators.

## Directory Structure

Each operator lives in its own self-contained directory:

```
src/
├── ARCHITECTURE.md          # This file
├── common/                  # Shared utilities (all operators use these)
│   ├── numerics.cuh         # CUDA numerical utilities (LSE, safe_exp, etc.)
│   ├── numerics.h           # CPU numerical utilities
│   ├── softmax.cuh          # CUDA softmax primitives
│   ├── reduce.cuh           # CUDA reduction primitives
│   ├── torch_utils.h        # PyTorch tensor validation macros
│   └── cuda_utils.h         # CUDA error checking, stream utilities
│
├── sw/                      # Smith-Waterman (linear gap) - REFERENCE IMPLEMENTATION
│   ├── README.md            # Algorithm documentation
│   ├── registry.cpp         # Operator schema definitions (m.def)
│   ├── kernels.cu           # CUDA kernel implementations
│   ├── kernels.cuh          # CUDA kernel declarations
│   ├── kernels_cpu.cpp      # CPU kernel implementations
│   ├── kernels_cpu.h        # CPU kernel declarations
│   ├── torch_cuda.cpp       # CUDA PyTorch bindings + autograd
│   └── torch_cpu.cpp        # CPU PyTorch bindings + autograd
│
├── sw_affine/               # Smith-Waterman (affine gap) - same structure
│   ├── README.md
│   ├── registry.cpp
│   ├── kernels.cu
│   ├── kernels.cuh
│   ├── kernels_cpu.cpp
│   ├── kernels_cpu.h
│   ├── torch_cuda.cpp
│   └── torch_cpu.cpp
│
└── <other_operator>/        # Future operators follow same pattern
```

## File Responsibilities

### `registry.cpp` - Operator Schema Definitions

Defines the public API contract using `TORCH_LIBRARY_FRAGMENT`:

```cpp
#include <torch/extension.h>

#ifdef USE_TORCH_LIBRARY

TORCH_LIBRARY_FRAGMENT(d2p, m) {
    // Core operators
    m.def("soft_<op>(Tensor scores, ...) -> Tensor[]");
    m.def("soft_<op>_float(Tensor scores, float param, ...) -> Tensor[]");
    m.def("soft_<op>_with_grads(...) -> (Tensor, ...)");
    m.def("soft_<op>_hvp(...) -> Tensor");
    m.def("soft_<op>_param_jacobian(...) -> Tensor");
    m.def("soft_<op>_backward_full(...) -> (Tensor, ...)");

    // Namespaced API (cleaner names)
    m.def("<op>_forward(...) -> Tensor[]");
    m.def("<op>_forward_t(...) -> Tensor[]");  // tensor params version
    m.def("<op>_marginals_backward(...) -> (Tensor, ...)");
    m.def("<op>_marginals_hvp(...) -> Tensor");
}

#endif
```

### `kernels.cu` / `kernels.cuh` - CUDA Kernels

Pure CUDA implementation with no PyTorch dependencies:

```cpp
// kernels.cuh - declarations
void <op>_forward(const float* scores, float* alpha, float* partition,
                  const int* lengths, int B, int L1, int L2,
                  float param, float temperature);

void <op>_backward(const float* alpha, const float* scores, ...);

void <op>_hvp(const float* alpha, const float* scores, const float* tangent, ...);

void <op>_param_grad(const float* alpha, ..., int param_type);
```

### `kernels_cpu.cpp` / `kernels_cpu.h` - CPU Kernels

CPU implementation mirroring CUDA interface:

```cpp
// kernels_cpu.h - declarations (same signatures as CUDA)
void <op>_forward_cpu(...);
void <op>_backward_cpu(...);
void <op>_hvp_cpu(...);
void <op>_param_grad_cpu(...);
```

### `torch_cuda.cpp` - CUDA PyTorch Bindings

Connects CUDA kernels to PyTorch's autograd and dispatcher:

```cpp
#include <torch/extension.h>
#include "common/torch_utils.h"
#include "common/cuda_utils.h"
#include "<op>/kernels.cuh"

// 1. Autograd Function class
class Soft<Op>CUDAFunction : public torch::autograd::Function<Soft<Op>CUDAFunction> {
    static tensor_list forward(AutogradContext* ctx, ...);
    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};

// 2. Python interface functions
std::vector<torch::Tensor> soft_<op>_cuda(...);
std::vector<torch::Tensor> soft_<op>_cuda_float(...);
// ... other functions ...

// 3. Namespaced API wrappers
std::vector<torch::Tensor> <op>_forward_cuda(...);
// ... other wrappers ...

// 4. Dispatcher registration
#ifdef USE_TORCH_LIBRARY
TORCH_LIBRARY_IMPL(d2p, CUDA, m) {
    m.impl("soft_<op>", soft_<op>_cuda);
    m.impl("<op>_forward", <op>_forward_cuda);
    // ...
}

TORCH_LIBRARY_IMPL(d2p, AutogradCUDA, m) {
    m.impl("soft_<op>", soft_<op>_cuda);
    // ...
}
#endif
```

### `torch_cpu.cpp` - CPU PyTorch Bindings

Same structure as CUDA, using CPU kernels:

```cpp
#include "common/torch_utils.h"
#include "<op>/kernels_cpu.h"

// Same pattern: Autograd class, interface functions, registration
TORCH_LIBRARY_IMPL(d2p, CPU, m) { ... }
TORCH_LIBRARY_IMPL(d2p, AutogradCPU, m) { ... }
```

### `README.md` - Algorithm Documentation

Documents the algorithm, recurrence relations, and usage:

```markdown
# Soft <Operator Name>

Brief description.

## Algorithm

Recurrence relations with mathematical notation.

## Files

| File | Description |
|------|-------------|
| ... | ... |

## Operations

| Operation | Description | Complexity |
|-----------|-------------|------------|
| ... | ... | ... |

## Usage

\`\`\`python
import d2p
result = d2p.soft_<op>(...)
\`\`\`
```

## Shared Utilities (`common/`)

All operators should use shared utilities for consistency:

### `torch_utils.h`
- `D2P_CHECK_INPUT_CUDA(x)` - Validate CUDA tensor
- `D2P_CHECK_INPUT_CPU(x)` - Validate CPU tensor
- `D2P_CHECK_CONTIGUOUS(x)` - Check contiguity
- `make_default_lengths_2d(B, L1, L2, device)` - Create default lengths tensor

### `numerics.cuh` / `numerics.h`
- `NINF` - Negative infinity constant (-1e30f)
- `safe_exp(x)` - Clamped exponential
- `lse2(a, b)` - Two-argument log-sum-exp
- `lse3(a, b, c)` - Three-argument log-sum-exp
- `lse4(a, b, c, d)` - Four-argument log-sum-exp
- `lse_T(T, ...)` - Temperature-scaled log-sum-exp

### `cuda_utils.h`
- `CUDA_CHECK(expr)` - CUDA error checking
- Stream and device utilities

## Adding a New Operator

1. **Create directory**: `src/<op>/`

2. **Implement kernels**:
   - `kernels.cu` + `kernels.cuh` (CUDA)
   - `kernels_cpu.cpp` + `kernels_cpu.h` (CPU)

3. **Create registry**: `registry.cpp` with schema definitions

4. **Create bindings**:
   - `torch_cuda.cpp` (CUDA bindings + autograd)
   - `torch_cpu.cpp` (CPU bindings + autograd)

5. **Document**: `README.md` with algorithm details

6. **Update build**: Add files to `meson.build`:
   ```meson
   cpp_sources = files(
     # <op> module (fully self-contained)
     'src/<op>/registry.cpp',
     'src/<op>/torch_cuda.cpp',
     'src/<op>/torch_cpu.cpp',
     'src/<op>/kernels_cpu.cpp',
   )

   cuda_sources = files(
     'src/<op>/kernels.cu',
   )
   ```

7. **Add Python API**: `d2p/<op>.py` and `d2p/ops/<op>.py`

## Design Principles

1. **Self-contained modules**: Each operator directory contains everything needed
2. **Consistent naming**: Same file names across all operators
3. **Shared utilities**: Common code in `common/`, not duplicated
4. **Clean separation**: Kernels have no PyTorch deps, bindings have no kernel logic
5. **Full differentiability**: Support gradients through all parameters
6. **Dual backend**: Every CUDA op has a CPU equivalent

## PyTorch Integration

Uses `TORCH_LIBRARY_FRAGMENT` and `TORCH_LIBRARY_IMPL` for:
- **torch.compile()** compatibility
- **Multi-backend dispatch** (CUDA/CPU automatic selection)
- **TorchScript** export support
- **Autograd** integration

Each module registers to the same `d2p` namespace, which PyTorch merges at runtime.
