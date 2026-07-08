"""Regression for the shared CUDA reduction primitive ``block_reduce_sum``
(``src/common/reduce.cuh``).

``block_reduce_sum`` combines per-warp partial sums through a reused
``__shared__ float shared[32]`` scratch buffer. When a kernel calls it more than
once in the same block, the next call's write to ``shared[]`` can race the previous
call's read (write-after-read) unless a ``__syncthreads()`` separates them. The fix
adds a trailing barrier inside ``block_reduce_sum`` so the buffer is free-on-return.

The primitive is a device function with no Python binding, so it is exercised through
the operators whose CUDA backward kernels issue back-to-back ``block_reduce_sum``
calls:

  * nw_affine — ``nw_affine_backward_diag_kernel`` (kernels.cu:429-430) and
    ``nw_affine_boundary_gap_grad_kernel`` (kernels.cu:480-481)
  * sw_affine — ``sw_affine_backward_diag_kernel`` (kernels.cu:457-458)

All launch 256-thread (8-warp) blocks, so the reduction is genuinely multi-warp and
the WAR is a cross-warp hazard. Both back-to-back pairs accumulate ``grad_open`` /
``grad_ext``; a corrupted reduction overwrites a whole warp's partial sum, which
shows up as (a) gap gradients that disagree with the race-free CPU backend, or
(b) run-to-run drift on CUDA. Both are checked, on a large ragged (masked) batch that
keeps every warp active so the race window is as wide as possible.

Note: a green run does not by itself prove the race is gone — races are timing
dependent. The authoritative proof is ``compute-sanitizer --tool racecheck`` (run
separately); these tests are behavioral regression guards.
"""

import pytest
import torch

try:
    import d2p
    D2P_AVAILABLE = True
except ImportError:
    D2P_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="block_reduce_sum WAR regression needs the CUDA d2p backend",
)

# Affine operators whose CUDA backward path issues back-to-back block_reduce_sum
# calls. Each ``*_with_grads`` returns (score/partition, posteriors, grad_open,
# grad_ext, grad_T); grad_open/grad_ext (indices 2/3) flow through the reduction.
AFFINE_WITH_GRADS = {
    "nw_affine": "soft_nw_affine_with_grads",
    "sw_affine": "soft_sw_affine_with_grads",
}

GAP_OPEN, GAP_EXT, TEMPERATURE = -1.7, -0.4, 1.0


def _max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def _make_masked_batch():
    """Large ragged batch: 8 warps active, lengths spanning short->full, and
    boundary-adjacent scores pushed low so the boundary gap-grad kernel (the second
    back-to-back reduction site) accumulates real masked partial sums."""
    B, max_L1, max_L2 = 8, 40, 44
    torch.manual_seed(20260707)
    scores = torch.randn(B, max_L1, max_L2)
    # Force posterior/beta mass toward the leading-gap boundary states.
    scores[:, 0, :] -= 2.5
    scores[:, :, 0] -= 2.5
    lengths = torch.tensor(
        [
            [max_L1, max_L2],
            [max_L1 - 7, max_L2 - 9],
            [max_L1 - 20, max_L2 - 4],
            [max_L1 - 3, max_L2 - 30],
            [12, 9],
            [max_L1, max_L2 - 1],
            [max_L1 - 15, max_L2 - 15],
            [5, 7],
        ],
        dtype=torch.int32,
    )
    return scores, lengths


@pytest.mark.parametrize("op_name", list(AFFINE_WITH_GRADS))
def test_gap_grads_match_cpu_backend(op_name):
    """CUDA gap grads (via back-to-back block_reduce_sum) must match the race-free
    CPU backend. A WAR corruption of the reduction yields a gross grad mismatch."""
    with_grads = getattr(d2p, AFFINE_WITH_GRADS[op_name])
    scores, lengths = _make_masked_batch()

    _, post_cpu, grad_open_cpu, grad_ext_cpu, grad_T_cpu = with_grads(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths
    )
    _, post_cuda, grad_open_cuda, grad_ext_cuda, grad_T_cuda = with_grads(
        scores.cuda(), GAP_OPEN, GAP_EXT, TEMPERATURE, lengths.cuda()
    )

    # Tolerances match the established nw_affine CPU/CUDA parity invariant
    # (tests/test_soft_nw_affine.py::TestCPUCUDA::test_cuda_with_grads_cpu_parity).
    assert torch.allclose(post_cpu, post_cuda.cpu(), rtol=1e-3, atol=1e-4), \
        f"{op_name} posteriors CPU/CUDA mismatch: max diff = {_max_diff(post_cpu, post_cuda)}"
    assert torch.allclose(grad_open_cpu, grad_open_cuda.cpu(), rtol=1e-2, atol=2e-3), \
        f"{op_name} grad_open CPU/CUDA mismatch: max diff = {_max_diff(grad_open_cpu, grad_open_cuda)}"
    assert torch.allclose(grad_ext_cpu, grad_ext_cuda.cpu(), rtol=1e-2, atol=2e-3), \
        f"{op_name} grad_ext CPU/CUDA mismatch: max diff = {_max_diff(grad_ext_cpu, grad_ext_cuda)}"
    assert torch.allclose(grad_T_cpu, grad_T_cuda.cpu(), rtol=1e-2, atol=2e-3), \
        f"{op_name} grad_T CPU/CUDA mismatch: max diff = {_max_diff(grad_T_cpu, grad_T_cuda)}"


@pytest.mark.parametrize("op_name", list(AFFINE_WITH_GRADS))
def test_gap_grads_repeatable_cuda(op_name):
    """Repeated CUDA evaluations of the same inputs must agree. A timing-dependent
    WAR race in the shared reduction buffer surfaces as run-to-run drift in the gap
    gradients (which a single CPU/CUDA comparison could miss if the race is flaky)."""
    with_grads = getattr(d2p, AFFINE_WITH_GRADS[op_name])
    scores, lengths = _make_masked_batch()
    scores_c, lengths_c = scores.cuda(), lengths.cuda()

    _, _, grad_open0, grad_ext0, grad_T0 = with_grads(
        scores_c, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths_c
    )
    for run in range(1, 12):
        _, _, grad_open_i, grad_ext_i, grad_T_i = with_grads(
            scores_c, GAP_OPEN, GAP_EXT, TEMPERATURE, lengths_c
        )
        assert torch.allclose(grad_open0, grad_open_i), \
            f"{op_name} grad_open run {run} drift: max diff = {_max_diff(grad_open0, grad_open_i)}"
        assert torch.allclose(grad_ext0, grad_ext_i), \
            f"{op_name} grad_ext run {run} drift: max diff = {_max_diff(grad_ext0, grad_ext_i)}"
        assert torch.allclose(grad_T0, grad_T_i), \
            f"{op_name} grad_T run {run} drift: max diff = {_max_diff(grad_T0, grad_T_i)}"
