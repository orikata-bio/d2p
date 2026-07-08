"""
Correctness tests for Soft Needleman-Wunsch (Global Alignment with Linear Gap).
"""

import contextlib
from pathlib import Path

import pytest
import torch

from reference import soft_nw_forward_naive, soft_nw_naive

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


def run_cpu_representative_outputs(scores, tangent, gap, temperature):
    score, posteriors, grad_gap, grad_T = d2p.soft_nw_with_grads(scores, gap, temperature, None)
    hvp = d2p.soft_nw_hvp(scores, tangent, gap, temperature, None)
    dP_dgap = d2p.soft_nw_param_jacobian(scores, 0, gap, temperature, None)
    dP_dT = d2p.soft_nw_param_jacobian(scores, 1, gap, temperature, None)
    return {
        "score": score,
        "posteriors": posteriors,
        "grad_gap": grad_gap,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dgap": dP_dgap,
        "dP_dT": dP_dT,
    }


def assert_threaded_nw_correctness(outputs, reference_outputs, thread_count):
    score_ref = reference_outputs["score"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_gap_ref = reference_outputs["grad_gap"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dgap_ref = reference_outputs["dP_dgap"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(score_ref, outputs["score"]), \
        f"{thread_count}-thread score mismatch: max diff = {max_diff(score_ref, outputs['score'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_gap_ref, outputs["grad_gap"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_gap mismatch: max diff = {max_diff(grad_gap_ref, outputs['grad_gap'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dgap_ref, outputs["dP_dgap"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dgap mismatch: max diff = {max_diff(dP_dgap_ref, outputs['dP_dgap'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


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


@pytest.fixture(params=[-1.0, -0.5, -2.0])
def gap(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_score(self, batch_size, seq_lengths, gap, temperature, device):
        """Test that NW scores match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        score_d2p = d2p.soft_nw_float(scores, gap, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch: max diff = {max_diff(score_ref, score_d2p)}"

    def test_score_positive_scores(self, batch_size, gap, device):
        """Test NW with positive match scores."""
        L1, L2 = 10, 12
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device).abs()

        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        score_d2p = d2p.soft_nw_float(scores, gap, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch with positive scores: max diff = {max_diff(score_ref, score_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, gap, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        posteriors_ref = soft_nw_naive(scores, gap, temperature)
        posteriors_d2p = d2p.soft_nw_float(scores, gap, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        posteriors = d2p.soft_nw_float(scores, gap, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = soft_nw_naive(scores_ref, gap, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_score_gradients(self, batch_size, device):
        """Test gradients through the score output."""
        L1, L2 = 6, 8
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        score = d2p.soft_nw_float(scores, gap, temperature, None)[0]
        loss = score.sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _ = soft_nw_forward_naive(scores_ref, gap, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-3, atol=1e-4), \
            f"Score gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"


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

        hvp_d2p = d2p.soft_nw_hvp(scores, V, gap, temperature, None)

        posteriors_plus = soft_nw_naive(scores + eps * V, gap, temperature)
        posteriors_minus = soft_nw_naive(scores - eps * V, gap, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_various_gaps(self, device):
        """Test HVP with various gap penalties."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0
        eps = 1e-4

        for gap in [-0.5, -1.0, -2.0]:
            torch.manual_seed(42)
            scores = torch.randn(B, L1, L2, device=device)
            V = torch.randn(B, L1, L2, device=device)

            hvp_d2p = d2p.soft_nw_hvp(scores, V, gap, temperature, None)

            posteriors_plus = soft_nw_naive(scores + eps * V, gap, temperature)
            posteriors_minus = soft_nw_naive(scores - eps * V, gap, temperature)
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=2e-3), \
                f"HVP mismatch for gap={gap}: max diff = {max_diff(hvp_fd, hvp_d2p)}"


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
            score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
            posteriors_ref = soft_nw_naive(scores, gap, temperature)

            score_gap_plus, _ = soft_nw_forward_naive(scores, gap + eps, temperature)
            score_gap_minus, _ = soft_nw_forward_naive(scores, gap - eps, temperature)
            grad_gap_ref = (score_gap_plus - score_gap_minus) / (2 * eps)

            score_temp_plus, _ = soft_nw_forward_naive(scores, gap, temperature + eps)
            score_temp_minus, _ = soft_nw_forward_naive(scores, gap, temperature - eps)
            grad_T_ref = (score_temp_plus - score_temp_minus) / (2 * eps)

            posteriors_gap_plus = soft_nw_naive(scores, gap + eps, temperature)
            posteriors_gap_minus = soft_nw_naive(scores, gap - eps, temperature)
            dP_dgap_ref = (posteriors_gap_plus - posteriors_gap_minus) / (2 * eps)

            posteriors_temp_plus = soft_nw_naive(scores, gap, temperature + eps)
            posteriors_temp_minus = soft_nw_naive(scores, gap, temperature - eps)
            dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

            posteriors_plus = soft_nw_naive(scores + eps * tangent, gap, temperature)
            posteriors_minus = soft_nw_naive(scores - eps * tangent, gap, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

        reference_outputs = {
            "score": score_ref,
            "posteriors": posteriors_ref,
            "grad_gap": grad_gap_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dgap": dP_dgap_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, gap, temperature)
            assert_threaded_nw_correctness(outputs, reference_outputs, thread_count)
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

        posteriors_cpu = d2p.soft_nw_float(scores_cpu, gap, temperature, None)[1]
        posteriors_cuda = d2p.soft_nw_float(scores_cuda, gap, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_score(self):
        """Test CPU vs CUDA scores match."""
        B, L1, L2 = 2, 10, 12
        gap = -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        score_cpu = d2p.soft_nw_float(scores_cpu, gap, temperature, None)[0]
        score_cuda = d2p.soft_nw_float(scores_cuda, gap, temperature, None)[0]

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"

    def test_cuda_boundary_gap_gradient(self):
        """CUDA grad_gap must include boundary beta contributions (r26 regression)."""
        B, L1, L2 = 3, 5, 7
        gap = -1.0
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(7)
        scores = torch.randn(B, L1, L2)
        # Push boundary-adjacent scores low to force beta mass onto boundaries
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5

        scores_cuda = scores.cuda()
        _, _, grad_gap_cuda, grad_T_cuda = d2p.soft_nw_with_grads(
            scores_cuda, gap, temperature, None
        )

        # Finite-difference reference (CPU naive)
        score_plus, _ = soft_nw_forward_naive(scores, gap + eps, temperature)
        score_minus, _ = soft_nw_forward_naive(scores, gap - eps, temperature)
        grad_gap_ref = (score_plus - score_minus) / (2 * eps)

        score_T_plus, _ = soft_nw_forward_naive(scores, gap, temperature + eps)
        score_T_minus, _ = soft_nw_forward_naive(scores, gap, temperature - eps)
        grad_T_ref = (score_T_plus - score_T_minus) / (2 * eps)

        assert allclose(grad_gap_ref, grad_gap_cuda, rtol=1e-2, atol=2e-3), \
            f"CUDA grad_gap boundary mismatch: max diff = {max_diff(grad_gap_ref, grad_gap_cuda)}"
        assert allclose(grad_T_ref, grad_T_cuda, rtol=1e-2, atol=2e-3), \
            f"CUDA grad_T mismatch (boundary-heavy): max diff = {max_diff(grad_T_ref, grad_T_cuda)}"

    def test_cuda_with_grads_cpu_parity(self):
        """CUDA with_grads outputs must match CPU with_grads (r26 regression)."""
        B, L1, L2 = 4, 6, 8
        gap = -0.75
        temperature = 1.0

        torch.manual_seed(99)
        scores = torch.randn(B, L1, L2)

        score_cpu, post_cpu, grad_gap_cpu, grad_T_cpu = d2p.soft_nw_with_grads(
            scores, gap, temperature, None
        )
        scores_cuda = scores.cuda()
        score_cuda, post_cuda, grad_gap_cuda, grad_T_cuda = d2p.soft_nw_with_grads(
            scores_cuda, gap, temperature, None
        )

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"
        assert allclose(post_cpu, post_cuda, rtol=1e-3, atol=1e-4), \
            f"CPU/CUDA posteriors mismatch: max diff = {max_diff(post_cpu, post_cuda)}"
        assert allclose(grad_gap_cpu, grad_gap_cuda, rtol=1e-2, atol=2e-3), \
            f"CPU/CUDA grad_gap mismatch: max diff = {max_diff(grad_gap_cpu, grad_gap_cuda)}"
        assert allclose(grad_T_cpu, grad_T_cuda, rtol=1e-2, atol=2e-3), \
            f"CPU/CUDA grad_T mismatch: max diff = {max_diff(grad_T_cpu, grad_T_cuda)}"

    def test_cuda_param_jacobian_boundary_finite_diff(self):
        """CUDA param_jacobians must include boundary-U initialization (r43 regression)."""
        B, L1, L2 = 2, 5, 7
        gap = -1.0
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2)
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5
        scores_cuda = scores.cuda()

        dP_dgap_cuda = d2p.soft_nw_param_jacobian(scores_cuda, 0, gap, temperature, None)
        dP_dT_cuda = d2p.soft_nw_param_jacobian(scores_cuda, 1, gap, temperature, None)

        posteriors_gap_plus = soft_nw_naive(scores, gap + eps, temperature)
        posteriors_gap_minus = soft_nw_naive(scores, gap - eps, temperature)
        dP_dgap_ref = (posteriors_gap_plus - posteriors_gap_minus) / (2 * eps)

        posteriors_temp_plus = soft_nw_naive(scores, gap, temperature + eps)
        posteriors_temp_minus = soft_nw_naive(scores, gap, temperature - eps)
        dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

        assert allclose(dP_dgap_ref, dP_dgap_cuda, rtol=1e-2, atol=5e-3), \
            f"CUDA dP/dgap mismatch: max diff = {max_diff(dP_dgap_ref, dP_dgap_cuda)}"
        assert allclose(dP_dT_ref, dP_dT_cuda, rtol=1e-2, atol=5e-3), \
            f"CUDA dP/dT mismatch: max diff = {max_diff(dP_dT_ref, dP_dT_cuda)}"

    def test_cuda_param_jacobian_cpu_parity_with_variable_lengths(self):
        """CUDA param_jacobians must match CPU on masked batches with variable lengths."""
        B, max_L1, max_L2 = 3, 6, 7
        gap = -0.75
        temperature = 0.7

        torch.manual_seed(321)
        scores = torch.randn(B, max_L1, max_L2)
        scores[0, 0, :] -= 2.0
        scores[1, :, 0] -= 1.5
        lengths = torch.tensor([[6, 7], [4, 5], [5, 3]], dtype=torch.int32)

        scores_cuda = scores.cuda()
        lengths_cuda = lengths.cuda()

        for param_type, name in ((0, "gap"), (1, "temperature")):
            cpu = d2p.soft_nw_param_jacobian(scores, param_type, gap, temperature, lengths)
            cuda = d2p.soft_nw_param_jacobian(
                scores_cuda, param_type, gap, temperature, lengths_cuda
            )
            assert allclose(cpu, cuda, rtol=1e-2, atol=5e-3), \
                f"CPU/CUDA dP/d{name} mismatch: max diff = {max_diff(cpu, cuda)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            score_ref, _ = soft_nw_forward_naive(scores_b, gap, temperature)
            posteriors_ref = soft_nw_naive(scores_b, gap, temperature)

            # Score should match for this sequence
            assert allclose(score_ref, score[b:b+1]), \
                f"Score mismatch for batch {b}: {score_ref.item()} vs {score[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestLengthValidation:

    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [[-1, 5]], r"lengths\[0,0\] must be between 0 and 4"),
            ("cpu", [[4, 6]], r"lengths\[0,1\] must be between 0 and 5"),
            pytest.param("cuda", [[-1, 5]], r"lengths\[0,0\] must be between 0 and 4",
                         marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")),
            pytest.param("cuda", [[4, 6]], r"lengths\[0,1\] must be between 0 and 5",
                         marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")),
        ],
    )
    def test_invalid_lengths_raise(self, device_type, lengths, match):
        """Explicit lengths must stay within the padded score shape."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            d2p.soft_nw_float(scores, -1.0, 1.0, lengths_t)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
    def test_cuda_lengths_must_match_scores_device(self):
        """Explicit CUDA lengths must live on the same device as scores."""
        scores = torch.randn(1, 4, 5, device="cuda:0")
        lengths_t = torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:1")

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            d2p.soft_nw_float(scores, -1.0, 1.0, lengths_t)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        gap = -1.0
        temperature = 1.0

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, None)
        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        # For NW, the posterior is beta * w_diag (option-additive), NOT necessarily 1.0
        # With 1x1: options are diag+score vs up+gap vs left+gap, so w_diag < 1
        assert allclose(score, score_ref), f"Single element score wrong: {score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), f"Single element posterior wrong: {posteriors.item()} vs {posteriors_ref.item()}"

    def test_row_vector(self, device):
        """Test 1xN score matrix."""
        scores = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        gap = -1.0
        temperature = 1.0

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, None)
        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        assert allclose(score, score_ref), "Row vector score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Row vector posteriors mismatch"

    def test_col_vector(self, device):
        """Test Nx1 score matrix."""
        scores = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        gap = -1.0
        temperature = 1.0

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, None)
        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        assert allclose(score, score_ref), "Column vector score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Column vector posteriors mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard NW)."""
        B, L1, L2 = 2, 6, 8
        gap = -1.0
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        posteriors = d2p.soft_nw_float(scores, gap, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        gap = -1.0
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        posteriors = d2p.soft_nw_float(scores, gap, temperature, None)[1]
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_gap(self, device):
        """Test with zero gap penalty."""
        B, L1, L2 = 2, 6, 6
        gap = 0.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, None)
        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        assert allclose(score, score_ref), f"Zero gap score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3, atol=1e-4), \
            "Zero gap posteriors mismatch"

    def test_positive_gap(self, device):
        """Test with positive gap penalty (reward for gaps)."""
        B, L1, L2 = 2, 6, 6
        gap = 0.5  # Unusual but should work
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors = d2p.soft_nw_float(scores, gap, temperature, None)
        score_ref, _ = soft_nw_forward_naive(scores, gap, temperature)
        posteriors_ref = soft_nw_naive(scores, gap, temperature)

        assert allclose(score, score_ref), f"Positive gap score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3, atol=1e-4), \
            "Positive gap posteriors mismatch"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test soft_nw_with_grads returns correct values."""
        B, L1, L2 = 2, 6, 8
        gap = -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors, grad_gap, grad_T = d2p.soft_nw_with_grads(
            scores, gap, temperature, None
        )

        # Compare with regular function
        score_ref, posteriors_ref = d2p.soft_nw_float(scores, gap, temperature, None)

        assert allclose(score, score_ref), "with_grads score mismatch"
        assert allclose(posteriors, posteriors_ref), "with_grads posteriors mismatch"

        # grad_gap and grad_T should be [B] tensors
        assert grad_gap.shape == (B,), f"grad_gap shape wrong: {grad_gap.shape}"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


# --- Memory-safety regression tests (merged from test_nw_{cpp,cuda}_memsafety.py) ---


def _cpu_scores():
    torch.manual_seed(7)
    return torch.randn(2, 4, 5)


def _oversized_zero_numel_scores():
    int32_max = torch.iinfo(torch.int32).max
    try:
        return torch.empty((1, 0, int32_max), dtype=torch.float32)
    except RuntimeError as exc:
        pytest.skip(f"PyTorch cannot create the oversized zero-numel shape: {exc}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cuda_scores(device="cuda"):
    return torch.randn(2, 4, 5, device=device, dtype=torch.float32)


def _cuda_lengths(device="cuda"):
    return torch.tensor([[4, 5], [3, 4]], dtype=torch.int32, device=device)


# --- CPU memory-safety regressions (from test_nw_cpp_memsafety.py) ---


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_i12_rejects_alpha_workspace_product_overflow():
    scores = _oversized_zero_numel_scores()

    with pytest.raises(RuntimeError, match=r"NW alpha workspace size is too large"):
        d2p.soft_nw_float(scores, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_i26_hvp_rejects_mismatched_tangent_shape():
    scores = _cpu_scores()
    tangent = torch.randn(2, 4, 4)

    with pytest.raises(RuntimeError, match=r"scores and tangent must have same shape"):
        d2p.soft_nw_hvp(scores, tangent, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_i48_hvp_rejects_non_3d_scores():
    scores = torch.randn(1, 2, 3, 4)
    tangent = torch.randn_like(scores)

    with pytest.raises(RuntimeError, match=r"scores must be 3D"):
        d2p.soft_nw_hvp(scores, tangent, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_i49_param_jacobian_rejects_unknown_param_type():
    scores = _cpu_scores()

    with pytest.raises(RuntimeError, match=r"param_type must be 0 or 1"):
        d2p.soft_nw_param_jacobian(scores, 2, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_i23_backward_full_rejects_mismatched_grad_alignment_shape():
    scores = _cpu_scores()
    grad_alignment = torch.randn(2, 4, 4)

    with pytest.raises(RuntimeError, match=r"grad_alignment must have same shape as scores"):
        d2p.soft_nw_backward_full(scores, grad_alignment, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_i23_backward_full_rejects_cuda_grad_alignment_for_cpu_scores():
    scores = _cpu_scores()
    grad_alignment = torch.randn_like(scores, device="cuda")

    # A CUDA grad_alignment forces the op to dispatch to the CUDA backend (the
    # CUDA dispatch key wins for mixed CPU/CUDA args), which validates the CPU
    # scores first and rejects the device mismatch as "scores must be a CUDA
    # tensor". Either way the mismatched-device input is safely rejected.
    with pytest.raises(RuntimeError, match=r"scores must be a CUDA tensor"):
        d2p.soft_nw_backward_full(scores, grad_alignment, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_valid_cpu_entrypoints_still_work():
    scores = _cpu_scores()
    tangent = torch.randn_like(scores)
    lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32)
    gap = -0.75
    temperature = 1.25

    score, posteriors = d2p.soft_nw_float(scores, gap, temperature, lengths)
    score_grads, posteriors_grads, grad_gap, grad_temp = d2p.soft_nw_with_grads(
        scores, gap, temperature, lengths
    )
    hvp = d2p.soft_nw_hvp(scores, tangent, gap, temperature, lengths)
    dP_dgap = d2p.soft_nw_param_jacobian(scores, 0, gap, temperature, lengths)
    dP_dtemp = d2p.soft_nw_param_jacobian(scores, 1, gap, temperature, lengths)
    grad_scores, total_grad_gap, total_grad_temp = d2p.soft_nw_backward_full(
        scores, tangent, gap, temperature, lengths
    )

    assert score.shape == (2,)
    assert torch.allclose(score, score_grads)
    assert posteriors.shape == scores.shape
    assert posteriors_grads.shape == scores.shape
    assert grad_gap.shape == (2,)
    assert grad_temp.shape == (2,)
    assert hvp.shape == scores.shape
    assert dP_dgap.shape == scores.shape
    assert dP_dtemp.shape == scores.shape
    assert grad_scores.shape == scores.shape
    assert total_grad_gap.shape == (1,)
    assert total_grad_temp.shape == (1,)


# --- CUDA memory-safety regressions (from test_nw_cuda_memsafety.py) ---


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_nw_cuda_kernel_indices_are_size_t_widened():
    kernels = (_repo_root() / "src" / "nw" / "kernels.cu").read_text()

    forbidden = (
        "int idx       = i * stride + j",
        "int idx_diag  = (i - 1) * stride + (j - 1)",
        "int idx_up    = (i - 1) * stride + j",
        "int idx_left  = i * stride + (j - 1)",
        "int score_idx = (i - 1) * max_L2 + (j - 1)",
        "int final_idx = L1 * stride + L2",
    )
    for pattern in forbidden:
        assert pattern not in kernels

    assert "size_t idx       = static_cast<size_t>(i) * stride + j" in kernels
    assert "size_t score_idx = static_cast<size_t>(i - 1) * static_cast<size_t>(max_L2)" in kernels
    assert "size_t final_idx = static_cast<size_t>(L1) * stride + L2" in kernels


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_nw_cuda_wrapper_alpha_size_uses_size_t_widening():
    wrapper = (_repo_root() / "src" / "nw" / "torch_cuda.cpp").read_text()

    assert "int alpha_size = (max_L1 + 1) * (max_L2 + 1)" not in wrapper
    assert "size_t alpha_size = (static_cast<size_t>(max_L1) + 1)" in wrapper
    assert wrapper.count("nw_alpha_size_cuda(max_L1, max_L2)") >= 5


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_rejects_mismatched_grad_alignment_shape():
    scores = _cuda_scores()
    grad_alignment = torch.randn(2, 4, 4, device="cuda")

    with pytest.raises(RuntimeError, match="grad_alignment must have same shape as scores"):
        torch.ops.d2p.soft_nw_backward_full(scores, grad_alignment, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_backward_full_rejects_cpu_grad_alignment_for_cuda_scores():
    scores = _cuda_scores()
    grad_alignment = torch.randn(2, 4, 5)

    with pytest.raises(RuntimeError, match="grad_alignment must be a CUDA tensor"):
        torch.ops.d2p.soft_nw_backward_full(scores, grad_alignment, -1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_hvp_rejects_cross_device_tangent():
    scores = _cuda_scores("cuda:0")
    tangent = torch.randn_like(scores, device="cuda:1")
    lengths = _cuda_lengths("cuda:0")

    with pytest.raises(RuntimeError, match="tangent must be on same device as scores"):
        torch.ops.d2p.soft_nw_hvp(scores, tangent, -1.0, 1.0, lengths)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_backward_full_rejects_cross_device_grad_alignment():
    scores = _cuda_scores("cuda:0")
    grad_alignment = torch.randn_like(scores, device="cuda:1")
    lengths = _cuda_lengths("cuda:0")

    with pytest.raises(RuntimeError, match="grad_alignment must be on same device as scores"):
        torch.ops.d2p.soft_nw_backward_full(scores, grad_alignment, -1.0, 1.0, lengths)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_tensor_params_reject_cross_device_gap():
    scores = _cuda_scores("cuda:0")
    gap = torch.tensor([-1.0], device="cuda:1")
    temperature = torch.tensor([1.0], device="cuda:0")
    lengths = _cuda_lengths("cuda:0")

    with pytest.raises(RuntimeError, match="gap must be on same device as scores"):
        torch.ops.d2p.soft_nw(scores, gap, temperature, lengths)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_current_device_guard_uses_scores_device():
    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = _cuda_scores("cuda:1").requires_grad_(True)
        lengths = _cuda_lengths("cuda:1")

        score, posteriors = torch.ops.d2p.soft_nw_float(scores, -1.0, 1.0, lengths)
        loss = score.sum() + 0.01 * posteriors.sum()
        loss.backward()
        torch.cuda.synchronize("cuda:1")

        assert score.device == scores.device
        assert posteriors.device == scores.device
        assert scores.grad is not None
        assert scores.grad.device == scores.device
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_score_only_backward_skips_alignment_zero_grad_path():
    device = torch.device("cuda")

    def backward_peak_delta(include_alignment_grad: bool) -> int:
        torch.manual_seed(20260707)
        torch.cuda.empty_cache()

        scores = torch.randn(2, 192, 192, device=device, requires_grad=True)
        gap = torch.tensor([-1.0], device=device, requires_grad=True)
        temperature = torch.tensor([1.0], device=device, requires_grad=True)
        lengths = torch.tensor([[192, 192], [180, 176]], dtype=torch.int32, device=device)

        score, alignment = torch.ops.d2p.soft_nw(scores, gap, temperature, lengths)
        loss = score.sum()
        if include_alignment_grad:
            upstream = torch.randn_like(alignment)
            loss = loss + (alignment * upstream).sum()

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        loss.backward()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) - baseline

    score_only_delta = backward_peak_delta(include_alignment_grad=False)
    alignment_delta = backward_peak_delta(include_alignment_grad=True)

    assert alignment_delta > score_only_delta + 1_000_000


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
