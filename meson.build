project('d2p', 'cpp',
  version : '0.1.0',
  default_options : [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=2',
  ]
)

# Find Python and PyTorch
# pure: false ensures all files are installed to platlib (with the .so)
python = import('python').find_installation('python3', pure: false)
python_dep = python.dependency()

cuda_option = get_option('cuda')
cuda_enabled = false
if not cuda_option.disabled()
  cuda_enabled = add_languages('cuda', required : cuda_option.enabled())
endif

if cuda_enabled
  message('d2p CUDA build: enabled')
else
  message('d2p CUDA build: disabled')
endif

torch_path_kind = 'cpu'
if cuda_enabled
  torch_path_kind = 'cuda'
endif

# Get PyTorch paths from Python
torch_include_result = run_command(python, '-c',
  '''
from torch.utils.cpp_extension import include_paths

kind = '@0@'
try:
    paths = include_paths(kind)
except Exception:
    if kind != 'cpu':
        raise
    paths = include_paths()

print(';'.join(paths))
'''.format(torch_path_kind),
  check : true
)
torch_lib_result = run_command(python, '-c',
  '''
from torch.utils.cpp_extension import library_paths

kind = '@0@'
try:
    paths = library_paths(kind)
except Exception:
    if kind != 'cpu':
        raise
    paths = library_paths()

print(';'.join(paths))
'''.format(torch_path_kind),
  check : true
)
torch_abi_result = run_command(python, '-c',
  'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))',
  check : true
)

torch_includes = torch_include_result.stdout().strip().split(';')
torch_lib_dirs = torch_lib_result.stdout().strip().split(';')
torch_abi = torch_abi_result.stdout().strip()

message('PyTorch include paths: ' + torch_include_result.stdout().strip())
message('PyTorch library paths: ' + torch_lib_result.stdout().strip())
message('PyTorch CXX11 ABI: ' + torch_abi)

# Create include directories.
# Skip reported include paths that do not actually exist: conda-forge's torch
# advertises a bogus torch/include/torch/csrc/api/include (the real C++ API headers
# live under the env prefix, which it also reports), and meson hard-fails on a
# missing include dir. No-op for pip's torch, where every reported path exists.
fs = import('fs')
torch_inc_dirs = []
foreach inc : torch_includes
  if inc != '' and fs.is_dir(inc)
    torch_inc_dirs += include_directories(inc, is_system : true)
  endif
endforeach

# Add local src directory for common/ headers
# Allows: #include "common/numerics.cuh"
src_inc = include_directories('src')

# Find torch libraries
cpp = meson.get_compiler('cpp')
torch_lib = cpp.find_library('torch', dirs : torch_lib_dirs, required : true)
torch_cpu_lib = cpp.find_library('torch_cpu', dirs : torch_lib_dirs, required : true)
c10_lib = cpp.find_library('c10', dirs : torch_lib_dirs, required : true)

torch_deps = [torch_lib, torch_cpu_lib, c10_lib]
if cuda_enabled
  torch_cuda_lib = cpp.find_library('torch_cuda', dirs : torch_lib_dirs, required : true)
  c10_cuda_lib = cpp.find_library('c10_cuda', dirs : torch_lib_dirs, required : true)
  torch_deps += [torch_cuda_lib, c10_cuda_lib]
endif

# Compiler flags
cxx_abi_flag = '-D_GLIBCXX_USE_CXX11_ABI=' + torch_abi
cxx_args = ['-O3', cxx_abi_flag, '-DUSE_TORCH_LIBRARY']
cuda_args = ['-O3', '-std=c++17', '-DUSE_TORCH_LIBRARY', '--expt-relaxed-constexpr']

# Enable OpenMP so ATen's at::parallel_for actually threads the CPU kernels (each
# batch element's DP runs on a torch/ATen worker thread). Without -fopenmp on the
# host C++ compiles, at::parallel_for compiles to its SERIAL fallback (measured:
# ~1x with vs ~7.6x on 8 threads once enabled). Applied to the host C++ sources
# only (cxx_args / link), never to nvcc's cuda_args. Guarded so toolchains that do
# not accept -fopenmp still build (they fall back to the current serial behavior).
openmp_link_args = []
if cpp.has_argument('-fopenmp')
  cxx_args += '-fopenmp'
  openmp_link_args = ['-fopenmp']
endif

cuda_arch_mode = get_option('d2p_cuda_arch_mode')
cuda_gencode_args = []

if cuda_enabled
  message('d2p CUDA arch mode: ' + cuda_arch_mode)
  if cuda_arch_mode == 'native'
    native_cuda_gencode_result = run_command(python, '-c',
      '''
import os
import re

def expand_arch_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []
    wants_ptx = False
    if token.endswith("+PTX"):
        wants_ptx = True
        token = token[:-4]
    token = token.replace(".", "")
    if not token.isdigit():
        raise SystemExit(f"unsupported CUDA arch token: {token!r}")
    flags = [f"-gencode=arch=compute_{token},code=sm_{token}"]
    if wants_ptx:
        flags.append(f"-gencode=arch=compute_{token},code=compute_{token}")
    return flags

arch_list = os.environ.get("D2P_CUDA_ARCH_LIST") or os.environ.get("TORCH_CUDA_ARCH_LIST") or ""
flags: list[str] = []

if arch_list:
    for token in re.split(r"[;, ]+", arch_list):
        if token:
            flags.extend(expand_arch_token(token))
else:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        token = f"{major}{minor}"
        flags.append(f"-gencode=arch=compute_{token},code=sm_{token}")

print(chr(10).join(flags))
''',
      check : true
    )
    native_cuda_gencode = native_cuda_gencode_result.stdout().strip()
    if native_cuda_gencode != ''
      foreach flag : native_cuda_gencode.split('\n')
        if flag != ''
          cuda_gencode_args += flag
        endif
      endforeach
    endif
  elif cuda_arch_mode == 'release-cu128'
    cuda_gencode_args += [
      '-gencode=arch=compute_75,code=sm_75',
      '-gencode=arch=compute_80,code=sm_80',
      '-gencode=arch=compute_86,code=sm_86',
      '-gencode=arch=compute_89,code=sm_89',
      '-gencode=arch=compute_90,code=sm_90',
      '-gencode=arch=compute_100,code=sm_100',
      '-gencode=arch=compute_101,code=sm_101',
      '-gencode=arch=compute_120,code=sm_120',
      '-gencode=arch=compute_120,code=compute_120',
    ]
  elif cuda_arch_mode == 'release-cu130'
    cuda_gencode_args += [
      '-gencode=arch=compute_75,code=sm_75',
      '-gencode=arch=compute_80,code=sm_80',
      '-gencode=arch=compute_86,code=sm_86',
      '-gencode=arch=compute_89,code=sm_89',
      '-gencode=arch=compute_90,code=sm_90',
      '-gencode=arch=compute_100,code=sm_100',
      '-gencode=arch=compute_103,code=sm_103',
      # CUDA 13.0 renumbered SM101 as SM110.
      '-gencode=arch=compute_110,code=sm_110',
      '-gencode=arch=compute_120,code=sm_120',
      '-gencode=arch=compute_121,code=sm_121',
      '-gencode=arch=compute_121,code=compute_121',
    ]
  elif cuda_arch_mode == 'custom'
    custom_cuda_gencode = get_option('d2p_cuda_gencode')
    if custom_cuda_gencode == ''
      error('d2p_cuda_arch_mode=custom requires -Dd2p_cuda_gencode=arch=compute_XX,code=sm_XX;...')
    endif
    foreach entry : custom_cuda_gencode.split(';')
      if entry != ''
        cuda_gencode_args += '-gencode=' + entry
      endif
    endforeach
  endif
endif

if cuda_enabled and cuda_gencode_args.length() > 0
  cuda_args += cuda_gencode_args
endif

# Source files - Soft Smith-Waterman (regular + affine), Soft CKY, Soft DTW, Soft NW, Soft MAS, Soft Eisner, Soft Levenshtein, Soft LCS, Soft OSA, and Soft Damerau
cpp_sources = files(
  # CKY module (fully self-contained)
  'src/cky/registry.cpp',
  'src/cky/torch_cpu.cpp',
  'src/cky/kernels_cpu.cpp',
  # SW module (fully self-contained)
  'src/sw/registry.cpp',
  'src/sw/torch_cpu.cpp',
  # SW Affine module (fully self-contained)
  'src/sw_affine/registry.cpp',
  'src/sw_affine/torch_cpu.cpp',
  'src/sw/kernels_cpu.cpp',
  'src/sw_affine/kernels_cpu.cpp',
  # NW module (fully self-contained)
  'src/nw/registry.cpp',
  'src/nw/torch_cpu.cpp',
  'src/nw/kernels_cpu.cpp',
  # NW Affine module (fully self-contained)
  'src/nw_affine/registry.cpp',
  'src/nw_affine/torch_cpu.cpp',
  'src/nw_affine/kernels_cpu.cpp',
  # DTW module (fully self-contained)
  'src/dtw/registry.cpp',
  'src/dtw/torch_cpu.cpp',
  'src/dtw/kernels_cpu.cpp',
  # Levenshtein module (fully self-contained)
  'src/lev/registry.cpp',
  'src/lev/torch_cpu.cpp',
  'src/lev/kernels_cpu.cpp',
  # LCS module (fully self-contained)
  'src/lcs/registry.cpp',
  'src/lcs/torch_cpu.cpp',
  'src/lcs/kernels_cpu.cpp',
  # OSA module (fully self-contained)
  'src/osa/registry.cpp',
  'src/osa/torch_cpu.cpp',
  'src/osa/kernels_cpu.cpp',
  # Damerau module (fully self-contained)
  'src/damerau/registry.cpp',
  'src/damerau/torch_cpu.cpp',
  'src/damerau/kernels_cpu.cpp',
  # MAS module (fully self-contained)
  'src/mas/registry.cpp',
  'src/mas/torch_cpu.cpp',
  'src/mas/kernels_cpu.cpp',
  # Eisner module (fully self-contained)
  'src/eisner/registry.cpp',
  'src/eisner/torch_cpu.cpp',
  'src/eisner/kernels_cpu.cpp',
)

extension_sources = cpp_sources

if cuda_enabled
  extension_sources += files(
    'src/cky/torch_cuda.cpp',
    'src/sw/torch_cuda.cpp',
    'src/sw_affine/torch_cuda.cpp',
    'src/nw/torch_cuda.cpp',
    'src/nw_affine/torch_cuda.cpp',
    'src/dtw/torch_cuda.cpp',
    'src/lev/torch_cuda.cpp',
    'src/lcs/torch_cuda.cpp',
    'src/osa/torch_cuda.cpp',
    'src/damerau/torch_cuda.cpp',
    'src/mas/torch_cuda.cpp',
    'src/eisner/torch_cuda.cpp',
    'src/cky/kernels.cu',
    'src/sw/kernels.cu',
    'src/sw_affine/kernels.cu',
    'src/nw/kernels.cu',
    'src/nw_affine/kernels.cu',
    'src/dtw/kernels.cu',
    'src/lev/kernels.cu',
    'src/lcs/kernels.cu',
    'src/osa/kernels.cu',
    'src/damerau/kernels.cu',
    'src/mas/kernels.cu',
    'src/eisner/kernels.cu',
  )
endif

# Build Python extension module
if cuda_enabled
  python.extension_module('_C',
    extension_sources,
    include_directories : [src_inc] + torch_inc_dirs,
    dependencies : [python_dep] + torch_deps,
    cpp_args : cxx_args,
    cuda_args : cuda_args,
    link_args : openmp_link_args,
    link_language : 'cpp',
    install : true,
    subdir : 'd2p',
  )
else
  python.extension_module('_C',
    extension_sources,
    include_directories : [src_inc] + torch_inc_dirs,
    dependencies : [python_dep] + torch_deps,
    cpp_args : cxx_args,
    link_args : openmp_link_args,
    link_language : 'cpp',
    install : true,
    subdir : 'd2p',
  )
endif

# Install Python source files
python.install_sources(
  'd2p/__init__.py',
  'd2p/_compile.py',
  'd2p/_ops.py',
  'd2p/_pt2_ops.py',
  'd2p/_pt2_utils.py',
  'd2p/cky.py',
  'd2p/dtw.py',
  'd2p/edit_distance.py',
  'd2p/eisner.py',
  'd2p/mas.py',
  'd2p/nw.py',
  'd2p/sw.py',
  subdir : 'd2p',
)

# Install d2p.ops subpackage
python.install_sources(
  'd2p/ops/__init__.py',
  'd2p/ops/sw.py',
  'd2p/ops/sw_affine.py',
  subdir : 'd2p/ops',
)
