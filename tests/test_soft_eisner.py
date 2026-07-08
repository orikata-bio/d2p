"""
Correctness tests for Soft Eisner (Projective Dependency Parsing).
"""

import contextlib

import pytest
import torch

from reference import soft_eisner_forward_naive, soft_eisner_naive

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


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(arc_scores, tangent, temperature):
    partition, marginals, grad_T = d2p.soft_eisner_backward_full(arc_scores, temperature, None)
    hvp = d2p.soft_eisner_hvp(arc_scores, tangent, temperature, None)
    dP_dT = d2p.soft_eisner_param_jacobian(arc_scores, temperature, None)
    return {
        "partition": partition,
        "marginals": marginals,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_eisner_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    marginals_ref = reference_outputs["marginals"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(marginals_ref, outputs["marginals"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread marginals mismatch: max diff = {max_diff(marginals_ref, outputs['marginals'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=1e-2, atol=1e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8, 12])
def seq_length(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_partition(self, batch_size, seq_length, temperature, device):
        """Test that partition functions match."""
        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
        partition_d2p, _ = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        assert allclose(partition_ref, partition_d2p), \
            f"Partition mismatch: max diff = {max_diff(partition_ref, partition_d2p)}"

    def test_partition_positive_scores(self, batch_size, device):
        """Test with positive arc scores."""
        seq_length = 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device).abs()

        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
        partition_d2p, _ = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        assert allclose(partition_ref, partition_d2p), \
            f"Partition mismatch with positive scores: max diff = {max_diff(partition_ref, partition_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_arc_marginals(self, batch_size, seq_length, temperature, device):
        """Test that arc marginals match."""
        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        marginals_ref = soft_eisner_naive(arc_scores, temperature)
        _, marginals_d2p = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        assert allclose(marginals_ref, marginals_d2p, rtol=1e-3, atol=1e-4), \
            f"Arc marginals mismatch: max diff = {max_diff(marginals_ref, marginals_d2p)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft Eisner."""
        seq_length = 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)
        arc_scores.requires_grad_(True)

        # Use soft_eisner_float which has autograd support
        partition = d2p.soft_eisner_float(arc_scores, temperature, None)
        loss = partition.sum()
        loss.backward()
        grad_d2p = arc_scores.grad.clone()

        arc_scores_ref = arc_scores.detach().clone().requires_grad_(True)
        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores_ref, temperature)
        loss_ref = partition_ref.sum()
        loss_ref.backward()
        grad_ref = arc_scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_marginal_sum(self, batch_size, seq_length, device):
        """Test that arc marginals for each dependent sum to ~1."""
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(batch_size, seq_length, seq_length, device=device)

        _, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        # For each word j (except root at 0), sum of incoming arcs should be ~1
        # marginals[:, :, j].sum(dim=1) ~= 1 for j > 0
        for j in range(1, seq_length):
            incoming_sum = marginals[:, :, j].sum(dim=1)
            expected = torch.ones(batch_size, device=device)
            assert allclose(incoming_sum, expected, rtol=1e-2, atol=1e-2), \
                f"Incoming arc sum for word {j} should be ~1, got {incoming_sum}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, n = 2, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)
        V = torch.randn(B, n, n, device=device)

        hvp_d2p = d2p.soft_eisner_hvp(arc_scores, V, temperature, None)

        marginals_plus = soft_eisner_naive(arc_scores + eps * V, temperature)
        marginals_minus = soft_eisner_naive(arc_scores - eps * V, temperature)
        hvp_fd = (marginals_plus - marginals_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_various_temps(self, device):
        """Test HVP with various temperatures."""
        B, n = 2, 6
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            arc_scores = torch.randn(B, n, n, device=device)
            V = torch.randn(B, n, n, device=device)

            hvp_d2p = d2p.soft_eisner_hvp(arc_scores, V, temperature, None)

            marginals_plus = soft_eisner_naive(arc_scores + eps * V, temperature)
            marginals_minus = soft_eisner_naive(arc_scores - eps * V, temperature)
            hvp_fd = (marginals_plus - marginals_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
                f"HVP mismatch for temp={temperature}: max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, n = 8, 7
        temperature = 1.0
        hvp_eps = 1e-4
        param_eps = 1e-3
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        arc_scores = torch.randn(B, n, n)
        tangent = torch.randn(B, n, n)

        with torch_num_threads(1):
            partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
            marginals_ref = soft_eisner_naive(arc_scores, temperature)

            partition_plus, _, _ = soft_eisner_forward_naive(arc_scores, temperature + hvp_eps)
            partition_minus, _, _ = soft_eisner_forward_naive(arc_scores, temperature - hvp_eps)
            grad_T_ref = (partition_plus - partition_minus) / (2 * hvp_eps)

            marginals_plus = soft_eisner_naive(arc_scores + hvp_eps * tangent, temperature)
            marginals_minus = soft_eisner_naive(arc_scores - hvp_eps * tangent, temperature)
            hvp_ref = (marginals_plus - marginals_minus) / (2 * hvp_eps)

            marginals_temp_plus = soft_eisner_naive(arc_scores, temperature + param_eps)
            marginals_temp_minus = soft_eisner_naive(arc_scores, temperature - param_eps)
            dP_dT_ref = (marginals_temp_plus - marginals_temp_minus) / (2 * param_eps)

        reference_outputs = {
            "partition": partition_ref,
            "marginals": marginals_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(arc_scores, tangent, temperature)
            assert_threaded_eisner_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores_cpu = torch.randn(B, n, n)
        arc_scores_cuda = arc_scores_cpu.cuda()

        _, marginals_cpu = d2p.soft_eisner_with_grads(arc_scores_cpu, temperature, None)
        _, marginals_cuda = d2p.soft_eisner_with_grads(arc_scores_cuda, temperature, None)

        assert allclose(marginals_cpu, marginals_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(marginals_cpu, marginals_cuda)}"

    def test_consistency_partition(self):
        """Test CPU vs CUDA partition functions match."""
        B, n = 2, 10
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores_cpu = torch.randn(B, n, n)
        arc_scores_cuda = arc_scores_cpu.cuda()

        partition_cpu, _ = d2p.soft_eisner_with_grads(arc_scores_cpu, temperature, None)
        partition_cuda, _ = d2p.soft_eisner_with_grads(arc_scores_cuda, temperature, None)

        assert allclose(partition_cpu, partition_cuda), \
            f"CPU/CUDA partition mismatch: max diff = {max_diff(partition_cpu, partition_cuda)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sentence lengths in batch."""
        B = 4
        max_n = 10
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, max_n, max_n, device=device)

        # Variable lengths
        lengths = torch.tensor([8, 10, 6, 9], device=device, dtype=torch.int32)

        partition, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            n = lengths[b].item()
            arc_scores_b = arc_scores[b:b+1, :n, :n]

            partition_ref, _, _ = soft_eisner_forward_naive(arc_scores_b, temperature)
            marginals_ref = soft_eisner_naive(arc_scores_b, temperature)

            assert allclose(partition_ref, partition[b:b+1]), \
                f"Partition mismatch for batch {b}: {partition_ref.item()} vs {partition[b].item()}"

            assert allclose(marginals_ref, marginals[b:b+1, :n, :n], rtol=1e-3, atol=1e-4), \
                f"Marginals mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestValidation:

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
    def test_non_square_arc_scores_raise(self, device_type):
        """Eisner charts must stay square."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 5, device=device)

        with pytest.raises(RuntimeError, match=r"arc_scores must be \[B, n, n\]"):
            d2p.soft_eisner_float(arc_scores, 1.0, None)

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
    def test_zero_length_chart_raises(self, device_type):
        """An empty Eisner chart must be rejected before kernel launch."""
        device = torch.device(device_type)
        arc_scores = torch.empty(1, 0, 0, dtype=torch.float32, device=device)

        with pytest.raises(RuntimeError, match=r"soft_eisner requires n > 0"):
            d2p.soft_eisner_float(arc_scores, 1.0, None)

    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [0], r"lengths\[0\] must be between 1 and 4"),
            ("cpu", [5], r"lengths\[0\] must be between 1 and 4"),
            pytest.param(
                "cuda",
                [0],
                r"lengths\[0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [5],
                r"lengths\[0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise(self, device_type, lengths, match):
        """Explicit lengths must stay within the padded chart shape."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 4, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            d2p.soft_eisner_float(arc_scores, 1.0, lengths_t)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_lengths_must_match_scores_device(self):
        """Lengths must stay on the same device as the score chart."""
        cuda_scores = torch.randn(1, 4, 4, device="cuda")
        cpu_lengths = torch.tensor([4], dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            d2p.soft_eisner_float(cuda_scores, 1.0, cpu_lengths)

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
        """The HVP tangent must stay shape-aligned with arc_scores."""
        device = torch.device(device_type)
        arc_scores = torch.randn(1, 4, 4, device=device)
        tangent = torch.randn(1, 4, 5, device=device)

        with pytest.raises(RuntimeError, match=r"tangent must have same shape as arc_scores"):
            d2p.soft_eisner_hvp(arc_scores, tangent, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_two_words(self, device):
        """Test with 2-word sentence (simplest non-trivial case)."""
        arc_scores = torch.tensor([[[0.0, 0.5], [-0.3, 0.0]]], device=device)
        temperature = 1.0

        partition, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)
        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
        marginals_ref = soft_eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), \
            f"Two words partition wrong: {partition.item()} vs {partition_ref.item()}"
        assert allclose(marginals, marginals_ref), \
            "Two words marginals mismatch"

    def test_three_words(self, device):
        """Test with 3-word sentence."""
        torch.manual_seed(42)
        arc_scores = torch.randn(1, 3, 3, device=device)
        temperature = 1.0

        partition, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)
        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
        marginals_ref = soft_eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), \
            "Three words partition mismatch"
        assert allclose(marginals, marginals_ref, rtol=1e-3), \
            "Three words marginals mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard parsing)."""
        B, n = 2, 6
        temperature = 0.01

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        # With low temperature, marginals should be close to 0 or 1
        assert marginals.min() >= -0.1, "Low temp marginals should be >= 0"
        assert marginals.max() <= 1.1, "Low temp marginals should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, n = 2, 6
        temperature = 10.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)
        marginals_ref = soft_eisner_naive(arc_scores, temperature)

        assert allclose(marginals_ref, marginals, rtol=1e-3, atol=1e-4), \
            "High temperature marginals mismatch"

    def test_diagonal_zeros(self, device):
        """Test that self-loop marginals are zero."""
        B, n = 2, 6
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        _, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        # Self-loops (diagonal) should have zero probability
        diag_marginals = torch.diagonal(marginals, dim1=1, dim2=2)
        assert allclose(diag_marginals, torch.zeros_like(diag_marginals), atol=1e-5), \
            f"Self-loop marginals should be 0, got max {diag_marginals.abs().max().item()}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test soft_eisner_with_grads returns correct values."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        partition, marginals = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        # Compare with reference
        partition_ref, _, _ = soft_eisner_forward_naive(arc_scores, temperature)
        marginals_ref = soft_eisner_naive(arc_scores, temperature)

        assert allclose(partition, partition_ref), "with_grads partition mismatch"
        assert allclose(marginals, marginals_ref, rtol=1e-3), "with_grads marginals mismatch"

    def test_backward_full_output(self, device):
        """Test soft_eisner_backward_full returns correct values."""
        B, n = 2, 8
        temperature = 1.0

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n, device=device)

        partition, marginals, grad_T = d2p.soft_eisner_backward_full(arc_scores, temperature, None)

        # Compare with soft_eisner_with_grads
        partition_ref, marginals_ref = d2p.soft_eisner_with_grads(arc_scores, temperature, None)

        assert allclose(partition, partition_ref), "backward_full partition mismatch"
        assert allclose(marginals, marginals_ref), "backward_full marginals mismatch"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestParamJacobian:

    def test_param_jacobian_temperature(self):
        """Test parameter Jacobian for temperature against finite differences (CPU)."""
        B, n = 2, 6
        temperature = 1.0
        eps = 1e-3

        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n)

        # dP/dT via param_jacobian
        dP_dT = d2p.soft_eisner_param_jacobian(arc_scores, temperature, None)

        # Finite diff validation
        marginals_plus = soft_eisner_naive(arc_scores, temperature + eps)
        marginals_minus = soft_eisner_naive(arc_scores, temperature - eps)
        dP_dT_fd = (marginals_plus - marginals_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=1e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_various_temps(self):
        """Test parameter Jacobian at various temperatures (CPU)."""
        B, n = 2, 6
        eps = 1e-3

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            arc_scores = torch.randn(B, n, n)

            dP_dT = d2p.soft_eisner_param_jacobian(arc_scores, temperature, None)

            marginals_plus = soft_eisner_naive(arc_scores, temperature + eps)
            marginals_minus = soft_eisner_naive(arc_scores, temperature - eps)
            dP_dT_fd = (marginals_plus - marginals_minus) / (2 * eps)

            assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=1e-3), \
                f"Param Jacobian mismatch for temp={temperature}: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_shape(self):
        """Test that param_jacobian output shape matches arc_scores."""
        B, n = 3, 8
        torch.manual_seed(42)
        arc_scores = torch.randn(B, n, n)

        dP_dT = d2p.soft_eisner_param_jacobian(arc_scores, 1.0, None)
        assert dP_dT.shape == arc_scores.shape, \
            f"Shape mismatch: {dP_dT.shape} vs {arc_scores.shape}"


# --- Memory-safety regression tests (merged from test_eisner_{cpp,cuda}_memsafety.py) ---

import re
from pathlib import Path


BAD_N = 46341
MAX_SAFE_N = 46340
OVERFLOW_MATCH = rf"soft_eisner requires n <= {MAX_SAFE_N}"


def _huge_logical_cpu_scores():
    """A logical [1, 46341, 46341] tensor backed by one float of storage."""
    storage = torch.empty(1, dtype=torch.float32)
    scores = torch.as_strided(storage, (1, BAD_N, BAD_N), (0, 0, 0))
    assert scores.numel() > 2**31
    assert storage.numel() == 1
    return scores


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.parametrize(
    "entrypoint",
    [
        lambda scores: d2p.soft_eisner(scores, 1.0, None),
        lambda scores: d2p.soft_eisner_float(scores, 1.0, None),
        lambda scores: d2p.soft_eisner_with_grads(scores, 1.0, None),
        lambda scores: d2p.soft_eisner_backward_full(scores, 1.0, None),
        lambda scores: d2p.soft_eisner_param_jacobian(scores, 1.0, None),
        lambda scores: d2p.soft_eisner_hvp(scores, _huge_logical_cpu_scores(), 1.0, None),
    ],
)
def test_cpu_entrypoints_reject_int_overflow_chart_before_kernel(entrypoint):
    scores = _huge_logical_cpu_scores()

    with pytest.raises(RuntimeError, match=OVERFLOW_MATCH):
        entrypoint(scores)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_valid_cpu_inputs_still_run():
    torch.manual_seed(20260707)
    scores = torch.randn(2, 5, 5)
    tangent = torch.randn_like(scores)
    lengths = torch.tensor([5, 3], dtype=torch.int32)

    partition_float = d2p.soft_eisner_float(scores, 1.0, lengths)
    partition, marginals = d2p.soft_eisner_with_grads(scores, 1.0, lengths)
    partition_full, marginals_full, grad_T = d2p.soft_eisner_backward_full(scores, 1.0, lengths)
    hvp = d2p.soft_eisner_hvp(scores, tangent, 1.0, lengths)
    dP_dT = d2p.soft_eisner_param_jacobian(scores, 1.0, lengths)

    assert torch.allclose(partition_float, partition)
    assert torch.allclose(partition, partition_full)
    assert torch.allclose(marginals, marginals_full)
    assert partition.shape == (2,)
    assert marginals.shape == scores.shape
    assert grad_T.shape == (2,)
    assert hvp.shape == scores.shape
    assert dP_dT.shape == scores.shape


def test_eisner_cpu_kernel_cell_offsets_are_widened():
    source = Path(__file__).resolve().parents[1] / "src" / "eisner" / "kernels_cpu.cpp"
    text = source.read_text()

    for forbidden in ("i * n +", "j * n +", "k * n +", "(k + 1) * n +", "0 * n +"):
        assert forbidden not in text
    assert "static_cast<size_t>(row) * static_cast<size_t>(n)" in text


CUDA_RUNTIME_REQUIRED = pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="Eisner CUDA memory-safety regression needs the CUDA d2p backend",
)
MULTI_CUDA_REQUIRED = pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE and torch.cuda.device_count() >= 2),
    reason="Eisner CUDA device-guard regression needs at least two CUDA devices",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TORCH_CUDA_CPP = REPO_ROOT / "src" / "eisner" / "torch_cuda.cpp"
KERNELS_CU = REPO_ROOT / "src" / "eisner" / "kernels.cu"


def _read_source(path):
    return path.read_text(encoding="utf-8")


def _call_eisner_entrypoint(name, scores, lengths=None):
    tangent = torch.randn_like(scores)
    temp_t = torch.tensor([1.0], dtype=torch.float32, device=scores.device)
    calls = {
        "soft_eisner": lambda: torch.ops.d2p.soft_eisner(scores, temp_t, lengths),
        "soft_eisner_float": lambda: d2p.soft_eisner_float(scores, 1.0, lengths),
        "soft_eisner_with_grads": lambda: d2p.soft_eisner_with_grads(scores, 1.0, lengths),
        "soft_eisner_hvp": lambda: d2p.soft_eisner_hvp(scores, tangent, 1.0, lengths),
        "soft_eisner_backward_full": lambda: d2p.soft_eisner_backward_full(scores, 1.0, lengths),
        "soft_eisner_param_jacobian": lambda: d2p.soft_eisner_param_jacobian(scores, 1.0, lengths),
    }
    return calls[name]()


def _flatten_tensors(result):
    if isinstance(result, (tuple, list)):
        return result
    return (result,)


def test_cuda_autograd_disables_materialized_marginals_grad():
    source = _read_source(TORCH_CUDA_CPP)

    assert "ctx->set_materialize_grads(false);" in source


def test_cuda_wrappers_guard_arc_scores_device():
    source = _read_source(TORCH_CUDA_CPP)

    # forward, backward, and the five direct CUDA entrypoints all need a guard
    # whose lifetime covers allocation, recordStream, and kernel dispatch.
    assert source.count("D2P_CUDA_GUARD(arc_scores);") >= 7


def test_cuda_chart_indices_use_size_t_helper():
    source = _read_source(KERNELS_CU)

    assert "__device__ __forceinline__ size_t chart_index" in source
    assert "return (size_t)row * (size_t)n + (size_t)col;" in source
    assert not re.search(r"\b[ijk]\s*\*\s*n\s*\+", source)
    assert not re.search(r"\(k\s*\+\s*1\)\s*\*\s*n\s*\+", source)


@CUDA_RUNTIME_REQUIRED
@pytest.mark.parametrize(
    "entrypoint",
    [
        "soft_eisner",
        "soft_eisner_float",
        "soft_eisner_with_grads",
        "soft_eisner_hvp",
        "soft_eisner_backward_full",
        "soft_eisner_param_jacobian",
    ],
)
def test_empty_cuda_batch_rejected_before_launch(entrypoint):
    scores = torch.empty((0, 4, 4), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match=r"soft_eisner requires B > 0"):
        _call_eisner_entrypoint(entrypoint, scores)


@MULTI_CUDA_REQUIRED
@pytest.mark.parametrize(
    "entrypoint",
    [
        "soft_eisner",
        "soft_eisner_float",
        "soft_eisner_with_grads",
        "soft_eisner_hvp",
        "soft_eisner_backward_full",
        "soft_eisner_param_jacobian",
    ],
)
def test_cuda_entrypoints_follow_arc_scores_device(entrypoint):
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(1, 4, 4, dtype=torch.float32, device="cuda:1")
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda:1")

        result = _call_eisner_entrypoint(entrypoint, scores, lengths)

        for tensor in _flatten_tensors(result):
            assert tensor.device == scores.device
    finally:
        torch.cuda.set_device(previous_device)


@MULTI_CUDA_REQUIRED
def test_cuda_autograd_backward_follows_arc_scores_device():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(1, 4, 4, dtype=torch.float32, device="cuda:1", requires_grad=True)
        lengths = torch.tensor([4], dtype=torch.int32, device="cuda:1")
        temp = torch.tensor([1.0], dtype=torch.float32, device="cuda:1")

        partition, _ = torch.ops.d2p.soft_eisner(scores, temp, lengths)
        partition.sum().backward()

        assert scores.grad is not None
        assert scores.grad.device == scores.device
    finally:
        torch.cuda.set_device(previous_device)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
