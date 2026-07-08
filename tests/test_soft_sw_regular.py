"""
Correctness tests for Soft Smith-Waterman (regular/linear gap).
"""

import contextlib

import pytest
import torch

from reference import soft_sw_regular_forward_naive, soft_sw_regular_naive

try:
    import d2p
    D2P_AVAILABLE = True
except ImportError:
    D2P_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def assert_padded_region_zero(name, tensor, lengths):
    tensor_cpu = tensor.cpu()
    lengths_cpu = lengths.cpu()
    max_L1 = tensor_cpu.size(1)
    max_L2 = tensor_cpu.size(2)

    for batch, (l1, l2) in enumerate(lengths_cpu.tolist()):
        if l1 < max_L1:
            assert torch.count_nonzero(tensor_cpu[batch, l1:, :]).item() == 0, \
                f"{name} batch {batch} wrote past active L1 length"
        if l2 < max_L2:
            assert torch.count_nonzero(tensor_cpu[batch, :, l2:]).item() == 0, \
                f"{name} batch {batch} wrote past active L2 length"


def finite_diff_param_jacobian(scores, lengths, param_type, gap, temperature, eps):
    if param_type == 0:
        plus = d2p.soft_sw_float(scores, gap + eps, temperature, lengths)[1]
        minus = d2p.soft_sw_float(scores, gap - eps, temperature, lengths)[1]
    else:
        plus = d2p.soft_sw_float(scores, gap, temperature + eps, lengths)[1]
        minus = d2p.soft_sw_float(scores, gap, temperature - eps, lengths)[1]
    return (plus - minus) / (2 * eps)


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, gap, temperature):
    partition, posteriors, grad_gap, grad_T = d2p.soft_sw_with_grads(scores, gap, temperature, None)
    hvp = d2p.soft_sw_hvp(scores, tangent, gap, temperature, None)
    dP_dgap = d2p.soft_sw_param_jacobian(scores, 0, gap, temperature, None)
    dP_dT = d2p.soft_sw_param_jacobian(scores, 1, gap, temperature, None)
    return {
        "partition": partition,
        "posteriors": posteriors,
        "grad_gap": grad_gap,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dgap": dP_dgap,
        "dP_dT": dP_dT,
    }


def assert_threaded_sw_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_gap_ref = reference_outputs["grad_gap"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_gap_ref, outputs["grad_gap"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_gap mismatch: max diff = {max_diff(grad_gap_ref, outputs['grad_gap'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=1e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(8, 10), (16, 16), (5, 20)])
def seq_lengths(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_partition(self, batch_size, seq_lengths, temperature, device):
        """Test that partition functions match."""
        L1, L2 = seq_lengths
        gap = -1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        partition_ref, _ = soft_sw_regular_forward_naive(scores, gap, temperature)
        partition_d2p = d2p.soft_sw_float(scores, gap, temperature, None)[0]

        assert allclose(partition_ref, partition_d2p), \
            f"Partition mismatch: max diff = {max_diff(partition_ref, partition_d2p)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_partition_large_batch_warp_dispatch(self):
        """Exercise the warp-forward CUDA path with a larger batch."""
        B, L1, L2 = 16, 9, 11
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(7)
        scores = torch.randn(B, L1, L2, device='cuda')

        partition_ref, _ = soft_sw_regular_forward_naive(scores, gap, temperature)
        partition_d2p = d2p.soft_sw_float(scores, gap, temperature, None)[0]

        assert allclose(partition_ref, partition_d2p), \
            f"Warp partition mismatch: max diff = {max_diff(partition_ref, partition_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        gap = -1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        posteriors_ref = soft_sw_regular_naive(scores, gap, temperature)
        posteriors_d2p = d2p.soft_sw_float(scores, gap, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_posteriors_large_batch_warp_dispatch(self):
        """Exercise the warp-backward CUDA path with a larger batch."""
        B, L1, L2 = 16, 9, 11
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(11)
        scores = torch.randn(B, L1, L2, device='cuda')

        posteriors_ref = soft_sw_regular_naive(scores, gap, temperature)
        posteriors_d2p = d2p.soft_sw_float(scores, gap, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Warp posterior mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        gap = -0.5

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device, requires_grad=True)

        posteriors = d2p.soft_sw_float(scores, gap, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = soft_sw_regular_naive(scores_ref, gap, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_temperature_gradient(self, device):
        """Test dS/dT against finite differences."""
        B, L1, L2 = 3, 5, 6
        gap = -0.75
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2, device=device)

        grad_T_d2p = d2p.soft_sw_with_grads(scores, gap, temperature, None)[3]

        partition_plus, _ = soft_sw_regular_forward_naive(scores, gap, temperature + eps)
        partition_minus, _ = soft_sw_regular_forward_naive(scores, gap, temperature - eps)
        grad_T_fd = (partition_plus - partition_minus) / (2 * eps)

        assert allclose(grad_T_fd, grad_T_d2p, rtol=1e-2, atol=2e-3), \
            f"Temperature gradient mismatch: max diff = {max_diff(grad_T_fd, grad_T_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        gap = -1.0
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_sw_hvp(scores, V, gap, temperature, None)

        posteriors_plus = soft_sw_regular_naive(scores + eps * V, gap, temperature)
        posteriors_minus = soft_sw_regular_naive(scores - eps * V, gap, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=1e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLengths:

    def test_variable_lengths_match_naive(self, device):
        """Variable-length batches should match the naive per-example result."""
        B = 4
        max_L1, max_L2 = 10, 12
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, max_L1, max_L2, device=device)
        lengths = torch.tensor(
            [[8, 10], [10, 12], [6, 8], [9, 11]],
            device=device,
            dtype=torch.int32,
        )

        partition, posteriors = d2p.soft_sw_float(scores, gap, temperature, lengths)
        assert_padded_region_zero("variable_length_posteriors", posteriors, lengths)

        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            partition_ref, _ = soft_sw_regular_forward_naive(scores_b, gap, temperature)
            posteriors_ref = soft_sw_regular_naive(scores_b, gap, temperature)

            assert allclose(partition_ref, partition[b:b+1]), \
                f"Partition mismatch for batch {b}: {partition_ref.item()} vs {partition[b].item()}"
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestParamGradients:

    def test_param_jacobian_variable_lengths_matches_finite_diff(self):
        """CPU and CUDA param_jacobians must match finite differences on masked batches."""
        B = 3
        max_L1, max_L2 = 6, 7
        gap = -1.4
        temperature = 0.9
        eps = 1e-4

        torch.manual_seed(20260409)
        scores = torch.randn(B, max_L1, max_L2)
        scores[0, 0, :] -= 2.5
        scores[1, :, 0] -= 2.0
        scores[2, 0, 0] -= 3.0
        lengths = torch.tensor([[6, 7], [4, 5], [5, 3]], dtype=torch.int32)

        backends = [("cpu", scores, lengths)]
        if CUDA_AVAILABLE:
            backends.append(("cuda", scores.cuda(), lengths.cuda()))

        for backend, backend_scores, backend_lengths in backends:
            for param_type, name in ((0, "gap"), (1, "temperature")):
                actual = d2p.soft_sw_param_jacobian(
                    backend_scores, param_type, gap, temperature, backend_lengths
                )
                fd = finite_diff_param_jacobian(
                    backend_scores, backend_lengths, param_type, gap, temperature, eps
                )

                assert allclose(actual, fd, rtol=2e-2, atol=5e-3), \
                    f"{backend} dP/d{name} mismatch: max diff = {max_diff(actual, fd)}"
                assert_padded_region_zero(f"{backend} dP/d{name}", actual, backend_lengths)

    def test_backward_full_param_grads_variable_lengths_match_finite_diff(self):
        """Backward-full param grads must match masked finite differences."""
        B = 3
        max_L1, max_L2 = 6, 7
        gap = -1.4
        temperature = 0.9
        eps = 1e-4

        torch.manual_seed(20260409)
        scores = torch.randn(B, max_L1, max_L2)
        scores[0, 0, :] -= 2.5
        scores[1, :, 0] -= 2.0
        scores[2, 0, 0] -= 3.0
        lengths = torch.tensor([[6, 7], [4, 5], [5, 3]], dtype=torch.int32)
        grad_alignment = torch.randn_like(scores)

        backends = [("cpu", scores, lengths, grad_alignment)]
        if CUDA_AVAILABLE:
            backends.append(
                ("cuda", scores.cuda(), lengths.cuda(), grad_alignment.cuda())
            )

        for backend, backend_scores, backend_lengths, backend_grad_alignment in backends:
            grad_scores, grad_gap, grad_T = d2p.soft_sw_backward_full(
                backend_scores, backend_grad_alignment, gap, temperature, backend_lengths
            )

            fd_gap = (
                finite_diff_param_jacobian(
                    backend_scores, backend_lengths, 0, gap, temperature, eps
                ) * backend_grad_alignment
            ).sum()
            fd_T = (
                finite_diff_param_jacobian(
                    backend_scores, backend_lengths, 1, gap, temperature, eps
                ) * backend_grad_alignment
            ).sum()

            assert_padded_region_zero(
                f"{backend} backward_full_grad_scores", grad_scores, backend_lengths
            )
            assert allclose(grad_gap, fd_gap.reshape_as(grad_gap), rtol=2e-2, atol=5e-3), \
                f"{backend} backward_full grad_gap mismatch: max diff = {max_diff(grad_gap, fd_gap.reshape_as(grad_gap))}"
            assert allclose(grad_T, fd_T.reshape_as(grad_T), rtol=2e-2, atol=5e-3), \
                f"{backend} backward_full grad_T mismatch: max diff = {max_diff(grad_T, fd_T.reshape_as(grad_T))}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestValidation:

    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [[-1, 5]], r"lengths\[0,0\] must be between 0 and 4"),
            ("cpu", [[4, 6]], r"lengths\[0,1\] must be between 0 and 5"),
            pytest.param(
                "cuda",
                [[-1, 5]],
                r"lengths\[0,0\] must be between 0 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 6]],
                r"lengths\[0,1\] must be between 0 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise(self, device_type, lengths, match):
        """Explicit lengths must stay within the padded score shape."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            d2p.soft_sw_float(scores, -1.0, 1.0, lengths_t)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
    def test_cuda_lengths_must_match_scores_device(self):
        """Explicit CUDA lengths must live on the same device as scores."""
        scores = torch.randn(1, 4, 5, device="cuda:0")
        lengths_t = torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:1")

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            d2p.soft_sw_float(scores, -1.0, 1.0, lengths_t)

    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_hvp_shape_mismatch_raises(self, device_type):
        """HVP inputs must stay shape-aligned with scores."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        tangent = torch.randn(1, 4, 4, device=device)

        with pytest.raises(RuntimeError, match=r"scores and tangent must have same shape"):
            d2p.soft_sw_hvp(scores, tangent, -1.0, 1.0, None)

    @pytest.mark.parametrize(
        "device_type",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_backward_full_shape_mismatch_raises(self, device_type):
        """Explicit backward inputs must match the padded score shape."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        grad_alignment = torch.randn(1, 4, 4, device=device)

        with pytest.raises(RuntimeError, match=r"grad_alignment must have same shape as scores"):
            d2p.soft_sw_backward_full(scores, grad_alignment, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        gap = -0.75
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            partition_ref, _ = soft_sw_regular_forward_naive(scores, gap, temperature)
            posteriors_ref = soft_sw_regular_naive(scores, gap, temperature)

            partition_gap_plus, _ = soft_sw_regular_forward_naive(scores, gap + eps, temperature)
            partition_gap_minus, _ = soft_sw_regular_forward_naive(scores, gap - eps, temperature)
            grad_gap_ref = (partition_gap_plus - partition_gap_minus) / (2 * eps)

            partition_temp_plus, _ = soft_sw_regular_forward_naive(scores, gap, temperature + eps)
            partition_temp_minus, _ = soft_sw_regular_forward_naive(scores, gap, temperature - eps)
            grad_T_ref = (partition_temp_plus - partition_temp_minus) / (2 * eps)

            posteriors_plus = soft_sw_regular_naive(scores + eps * tangent, gap, temperature)
            posteriors_minus = soft_sw_regular_naive(scores - eps * tangent, gap, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

        reference_outputs = {
            "partition": partition_ref,
            "posteriors": posteriors_ref,
            "grad_gap": grad_gap_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, gap, temperature)
            assert_threaded_sw_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, L1, L2 = 2, 8, 10
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = d2p.soft_sw_float(scores_cpu, gap, temperature, None)[1]
        posteriors_cuda = d2p.soft_sw_float(scores_cuda, gap, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"


# --- Memory-safety regression tests (merged from test_sw_{cpp,cuda}_memsafety.py) ---

INT32_MAX = torch.iinfo(torch.int32).max
OVERFLOW_RE = r"SW CPU DP table size exceeds supported int32 range"


def _zero_storage_overflow_scores():
    try:
        scores = torch.empty((1, 0, INT32_MAX), dtype=torch.float32)
    except RuntimeError as exc:
        pytest.skip(f"PyTorch cannot construct zero-storage overflow shape: {exc}")

    assert scores.numel() == 0
    assert scores.is_contiguous()
    return scores


def _zero_lengths():
    return torch.tensor([[0, 0]], dtype=torch.int32)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestSWCppOverflowGuard:
    @pytest.mark.parametrize(
        "op_name",
        [
            "soft_sw_float",
            "soft_sw_with_grads",
            "soft_sw_hvp",
            "soft_sw_param_jacobian",
            "soft_sw_backward_full",
        ],
    )
    def test_cpu_dp_table_overflow_rejected_before_workspace_allocation(self, op_name):
        scores = _zero_storage_overflow_scores()
        lengths = _zero_lengths()

        with pytest.raises(RuntimeError, match=OVERFLOW_RE):
            if op_name == "soft_sw_float":
                d2p.soft_sw_float(scores, -1.0, 1.0, lengths)
            elif op_name == "soft_sw_with_grads":
                d2p.soft_sw_with_grads(scores, -1.0, 1.0, lengths)
            elif op_name == "soft_sw_hvp":
                d2p.soft_sw_hvp(scores, torch.empty_like(scores), -1.0, 1.0, lengths)
            elif op_name == "soft_sw_param_jacobian":
                d2p.soft_sw_param_jacobian(scores, 0, -1.0, 1.0, lengths)
            elif op_name == "soft_sw_backward_full":
                d2p.soft_sw_backward_full(scores, torch.empty_like(scores), -1.0, 1.0, lengths)
            else:
                raise AssertionError(f"unhandled op_name {op_name}")


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestSWCppValidInputs:
    def test_valid_cpu_forward_matches_reference(self):
        torch.manual_seed(20260707)
        scores = torch.randn(2, 4, 5)
        lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
        gap = -0.75
        temperature = 1.0

        partition, posteriors = d2p.soft_sw_float(scores, gap, temperature, lengths)

        for b, (l1, l2) in enumerate(lengths.tolist()):
            scores_b = scores[b : b + 1, :l1, :l2]
            partition_ref, _ = soft_sw_regular_forward_naive(scores_b, gap, temperature)
            posteriors_ref = soft_sw_regular_naive(scores_b, gap, temperature)

            assert torch.allclose(partition[b : b + 1], partition_ref, rtol=1e-4, atol=1e-5)
            assert torch.allclose(
                posteriors[b : b + 1, :l1, :l2],
                posteriors_ref,
                rtol=1e-3,
                atol=1e-4,
            )

    def test_valid_cpu_auxiliary_paths_still_run(self):
        torch.manual_seed(20260707)
        scores = torch.randn(2, 3, 4)
        tangent = torch.randn_like(scores)
        lengths = torch.tensor([[3, 4], [2, 3]], dtype=torch.int32)
        gap = -1.0
        temperature = 0.9

        partition, posteriors, grad_gap, grad_T = d2p.soft_sw_with_grads(
            scores, gap, temperature, lengths
        )
        hvp = d2p.soft_sw_hvp(scores, tangent, gap, temperature, lengths)
        dP_dgap = d2p.soft_sw_param_jacobian(scores, 0, gap, temperature, lengths)
        dP_dT = d2p.soft_sw_param_jacobian(scores, 1, gap, temperature, lengths)
        grad_scores, backward_gap, backward_T = d2p.soft_sw_backward_full(
            scores, tangent, gap, temperature, lengths
        )

        assert partition.shape == (2,)
        assert posteriors.shape == scores.shape
        assert grad_gap.shape == (2,)
        assert grad_T.shape == (2,)
        assert hvp.shape == scores.shape
        assert dP_dgap.shape == scores.shape
        assert dP_dT.shape == scores.shape
        assert grad_scores.shape == scores.shape
        assert backward_gap.shape == (1,)
        assert backward_T.shape == (1,)


def cuda_kernel_names(call):
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:
        pytest.skip(f"CUDA profiler unavailable: {exc}")

    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            call()
            torch.cuda.synchronize()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "cupti" in message or "kineto" in message or "profiler" in message:
            pytest.skip(f"CUDA profiler failed: {exc}")
        raise

    return {
        getattr(event, "name", "") or getattr(event, "key", "")
        for event in prof.events()
    }


def assert_cuda_kernel_seen(kernel_names, token):
    if any(token in name for name in kernel_names):
        return
    relevant = sorted(name for name in kernel_names if "sw_regular_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; SW kernels seen: {relevant[:20]}"
    )


def assert_allclose(a, b, rtol=1e-4, atol=1e-5):
    assert torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol), (
        f"max diff = {(a.cpu() - b.cpu()).abs().max().item()}"
    )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestSWCudaMemSafety:
    def test_oversized_dp_table_rejected_before_cuda_allocation(self):
        scores = torch.empty((0, 46341, 46341), device="cuda")
        tangent = torch.empty_like(scores)
        lengths = torch.empty((0, 2), dtype=torch.int32, device="cuda")
        gap = torch.tensor([-1.0], device="cuda")
        temperature = torch.tensor([1.0], device="cuda")

        entrypoints = (
            lambda: d2p.soft_sw(scores, gap, temperature, lengths),
            lambda: d2p.soft_sw_float(scores, -1.0, 1.0, None),
            lambda: d2p.soft_sw_with_grads(scores, -1.0, 1.0, None),
            lambda: d2p.soft_sw_hvp(scores, tangent, -1.0, 1.0, None),
            lambda: d2p.soft_sw_param_jacobian(scores, 0, -1.0, 1.0, None),
            lambda: d2p.soft_sw_backward_full(scores, tangent, -1.0, 1.0, None),
        )

        for call in entrypoints:
            with pytest.raises(RuntimeError, match="SW CUDA DP table is too large"):
                call()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
    def test_entrypoints_honor_scores_device_when_current_device_differs(self):
        scores = torch.randn(4, 4, 5, device="cuda:1")
        tangent = torch.randn_like(scores)
        lengths = torch.tensor([[4, 5], [3, 5], [4, 4], [2, 3]], dtype=torch.int32, device="cuda:1")

        previous_device = torch.cuda.current_device()
        try:
            torch.cuda.set_device(0)
            value, marginals = d2p.soft_sw_float(scores, -1.0, 1.0, lengths)
            assert value.device == scores.device
            assert marginals.device == scores.device

            d2p.soft_sw_with_grads(scores, -1.0, 1.0, lengths)
            d2p.soft_sw_hvp(scores, tangent, -1.0, 1.0, lengths)
            d2p.soft_sw_param_jacobian(scores, 0, -1.0, 1.0, lengths)
            d2p.soft_sw_param_jacobian(scores, 1, -1.0, 1.0, lengths)
            d2p.soft_sw_backward_full(scores, tangent, -1.0, 1.0, lengths)

            scores_req = scores.detach().clone().requires_grad_(True)
            value_req, _ = d2p.soft_sw_float(scores_req, -1.0, 1.0, lengths)
            value_req.sum().backward()
            assert scores_req.grad is not None
            assert scores_req.grad.device == scores_req.device
            torch.cuda.synchronize(scores.device)
        finally:
            torch.cuda.set_device(previous_device)

    def test_value_backward_skips_unused_marginal_hvp_and_param_jacobians(self):
        B, L1, L2 = 1, 6, 7

        def value_only_backward():
            torch.manual_seed(8675309)
            scores = torch.randn(B, L1, L2, device="cuda", requires_grad=True)
            value, _ = d2p.soft_sw_float(scores, -1.0, 1.0, None)
            value.sum().backward()

        def explicit_zero_marginal_backward():
            torch.manual_seed(8675309)
            scores = torch.randn(B, L1, L2, device="cuda", requires_grad=True)
            value, marginals = d2p.soft_sw_float(scores, -1.0, 1.0, None)
            (value.sum() + 0.0 * marginals.sum()).backward()

        value_only_kernels = cuda_kernel_names(value_only_backward)
        assert_cuda_kernel_seen(value_only_kernels, "sw_regular_forward_diag_kernel")
        assert not any("sw_regular_hvp_" in name for name in value_only_kernels)
        assert not any("sw_regular_param_grad_" in name for name in value_only_kernels)

        explicit_zero_kernels = cuda_kernel_names(explicit_zero_marginal_backward)
        assert_cuda_kernel_seen(explicit_zero_kernels, "sw_regular_hvp_forward_diag_kernel")
        assert_cuda_kernel_seen(explicit_zero_kernels, "sw_regular_param_grad_forward_diag_kernel")

    def test_valid_inputs_match_cpu(self):
        torch.manual_seed(1234)
        scores_cpu = torch.randn(2, 5, 6)
        lengths_cpu = torch.tensor([[5, 6], [3, 4]], dtype=torch.int32)
        scores_cuda = scores_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        value_cpu, marginals_cpu = d2p.soft_sw_float(scores_cpu, -1.0, 1.0, lengths_cpu)
        value_cuda, marginals_cuda = d2p.soft_sw_float(scores_cuda, -1.0, 1.0, lengths_cuda)

        assert_allclose(value_cuda, value_cpu, rtol=1e-4, atol=1e-5)
        assert_allclose(marginals_cuda, marginals_cpu, rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
