"""
Correctness tests for Soft Needleman-Wunsch with Affine Gap (Global Alignment with Affine Gap).
"""

import contextlib
import re

import pytest
import torch

from reference import soft_nw_affine_forward_naive, soft_nw_affine_naive

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


def run_cpu_representative_outputs(scores, tangent, gap_open, gap_ext, temperature):
    score, posteriors, grad_gap_open, grad_gap_ext, grad_T = d2p.soft_nw_affine_with_grads(
        scores, gap_open, gap_ext, temperature, None
    )
    hvp = d2p.soft_nw_affine_hvp(scores, tangent, gap_open, gap_ext, temperature, None)
    dP_dgap_open = d2p.soft_nw_affine_param_jacobian(
        scores, 0, gap_open, gap_ext, temperature, None
    )
    dP_dgap_ext = d2p.soft_nw_affine_param_jacobian(
        scores, 1, gap_open, gap_ext, temperature, None
    )
    dP_dT = d2p.soft_nw_affine_param_jacobian(
        scores, 2, gap_open, gap_ext, temperature, None
    )
    return {
        "score": score,
        "posteriors": posteriors,
        "grad_gap_open": grad_gap_open,
        "grad_gap_ext": grad_gap_ext,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dgap_open": dP_dgap_open,
        "dP_dgap_ext": dP_dgap_ext,
        "dP_dT": dP_dT,
    }


def assert_threaded_nw_affine_correctness(outputs, reference_outputs, thread_count):
    assert allclose(reference_outputs["score"], outputs["score"]), \
        f"{thread_count}-thread score mismatch: max diff = {max_diff(reference_outputs['score'], outputs['score'])}"
    assert allclose(reference_outputs["posteriors"], outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(reference_outputs['posteriors'], outputs['posteriors'])}"
    assert allclose(reference_outputs["grad_gap_open"], outputs["grad_gap_open"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_gap_open mismatch: max diff = {max_diff(reference_outputs['grad_gap_open'], outputs['grad_gap_open'])}"
    assert allclose(reference_outputs["grad_gap_ext"], outputs["grad_gap_ext"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_gap_ext mismatch: max diff = {max_diff(reference_outputs['grad_gap_ext'], outputs['grad_gap_ext'])}"
    assert allclose(reference_outputs["grad_T"], outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(reference_outputs['grad_T'], outputs['grad_T'])}"
    assert allclose(reference_outputs["hvp"], outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(reference_outputs['hvp'], outputs['hvp'])}"
    assert allclose(reference_outputs["dP_dgap_open"], outputs["dP_dgap_open"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread dP/dgap_open mismatch: max diff = {max_diff(reference_outputs['dP_dgap_open'], outputs['dP_dgap_open'])}"
    assert allclose(reference_outputs["dP_dgap_ext"], outputs["dP_dgap_ext"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread dP/dgap_ext mismatch: max diff = {max_diff(reference_outputs['dP_dgap_ext'], outputs['dP_dgap_ext'])}"
    assert allclose(reference_outputs["dP_dT"], outputs["dP_dT"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(reference_outputs['dP_dT'], outputs['dP_dT'])}"


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


@pytest.fixture(params=[(-2.0, -0.5), (-1.0, -0.3), (-3.0, -1.0)])
def gap_penalties(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_score(self, batch_size, seq_lengths, gap_penalties, temperature, device):
        """Test that NW affine scores match."""
        L1, L2 = seq_lengths
        gap_open, gap_ext = gap_penalties

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        score_d2p = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch: max diff = {max_diff(score_ref, score_d2p)}"

    def test_score_positive_scores(self, batch_size, gap_penalties, device):
        """Test NW affine with positive match scores."""
        L1, L2 = 10, 12
        gap_open, gap_ext = gap_penalties
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device).abs()

        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        score_d2p = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch with positive scores: max diff = {max_diff(score_ref, score_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, gap_penalties, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        gap_open, gap_ext = gap_penalties

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)

        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)
        posteriors_d2p = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = soft_nw_affine_naive(scores_ref, gap_open, gap_ext, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_score_gradients(self, batch_size, device):
        """Test gradients through the score output."""
        L1, L2 = 6, 8
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        score = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[0]
        loss = score.sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores_ref, gap_open, gap_ext, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-3, atol=1e-4), \
            f"Score gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_gap_parameter_gradients_include_boundary_gaps(self):
        """CPU gap-parameter grads should include leading-gap boundary states."""
        B, L1, L2 = 3, 5, 7
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(7)
        scores = torch.randn(B, L1, L2)
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5

        with torch_num_threads(1):
            _, _, grad_gap_open, grad_gap_ext, _ = d2p.soft_nw_affine_with_grads(
                scores, gap_open, gap_ext, temperature, None
            )

        score_gap_open_plus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open + eps, gap_ext, temperature
        )
        score_gap_open_minus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open - eps, gap_ext, temperature
        )
        grad_gap_open_ref = (score_gap_open_plus - score_gap_open_minus) / (2 * eps)

        score_gap_ext_plus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext + eps, temperature
        )
        score_gap_ext_minus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext - eps, temperature
        )
        grad_gap_ext_ref = (score_gap_ext_plus - score_gap_ext_minus) / (2 * eps)

        assert allclose(grad_gap_open_ref, grad_gap_open, rtol=1e-2, atol=2e-3), \
            f"grad_gap_open mismatch: max diff = {max_diff(grad_gap_open_ref, grad_gap_open)}"
        assert allclose(grad_gap_ext_ref, grad_gap_ext, rtol=1e-2, atol=2e-3), \
            f"grad_gap_ext mismatch: max diff = {max_diff(grad_gap_ext_ref, grad_gap_ext)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_nw_affine_hvp(scores, V, gap_open, gap_ext, temperature, None)

        posteriors_plus = soft_nw_affine_naive(scores + eps * V, gap_open, gap_ext, temperature)
        posteriors_minus = soft_nw_affine_naive(scores - eps * V, gap_open, gap_ext, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_various_gaps(self, device):
        """Test HVP with various gap penalties."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0
        eps = 1e-4

        for gap_open, gap_ext in [(-1.0, -0.3), (-2.0, -0.5), (-3.0, -1.0)]:
            torch.manual_seed(42)
            scores = torch.randn(B, L1, L2, device=device)
            V = torch.randn(B, L1, L2, device=device)

            hvp_d2p = d2p.soft_nw_affine_hvp(scores, V, gap_open, gap_ext, temperature, None)

            posteriors_plus = soft_nw_affine_naive(scores + eps * V, gap_open, gap_ext, temperature)
            posteriors_minus = soft_nw_affine_naive(scores - eps * V, gap_open, gap_ext, temperature)
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=2e-3), \
                f"HVP mismatch for gap_open={gap_open}, gap_ext={gap_ext}: max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 4, 6, 7
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
            posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

            score_gap_open_plus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open + eps, gap_ext, temperature
            )
            score_gap_open_minus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open - eps, gap_ext, temperature
            )
            grad_gap_open_ref = (score_gap_open_plus - score_gap_open_minus) / (2 * eps)

            score_gap_ext_plus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open, gap_ext + eps, temperature
            )
            score_gap_ext_minus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open, gap_ext - eps, temperature
            )
            grad_gap_ext_ref = (score_gap_ext_plus - score_gap_ext_minus) / (2 * eps)

            score_temp_plus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open, gap_ext, temperature + eps
            )
            score_temp_minus, _, _, _ = soft_nw_affine_forward_naive(
                scores, gap_open, gap_ext, temperature - eps
            )
            grad_T_ref = (score_temp_plus - score_temp_minus) / (2 * eps)

            posteriors_plus = soft_nw_affine_naive(scores + eps * tangent, gap_open, gap_ext, temperature)
            posteriors_minus = soft_nw_affine_naive(scores - eps * tangent, gap_open, gap_ext, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            posteriors_gap_open_plus = soft_nw_affine_naive(
                scores, gap_open + eps, gap_ext, temperature
            )
            posteriors_gap_open_minus = soft_nw_affine_naive(
                scores, gap_open - eps, gap_ext, temperature
            )
            dP_dgap_open_ref = (posteriors_gap_open_plus - posteriors_gap_open_minus) / (2 * eps)

            posteriors_gap_ext_plus = soft_nw_affine_naive(
                scores, gap_open, gap_ext + eps, temperature
            )
            posteriors_gap_ext_minus = soft_nw_affine_naive(
                scores, gap_open, gap_ext - eps, temperature
            )
            dP_dgap_ext_ref = (posteriors_gap_ext_plus - posteriors_gap_ext_minus) / (2 * eps)

            posteriors_temp_plus = soft_nw_affine_naive(
                scores, gap_open, gap_ext, temperature + eps
            )
            posteriors_temp_minus = soft_nw_affine_naive(
                scores, gap_open, gap_ext, temperature - eps
            )
            dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

        reference_outputs = {
            "score": score_ref,
            "posteriors": posteriors_ref,
            "grad_gap_open": grad_gap_open_ref,
            "grad_gap_ext": grad_gap_ext_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dgap_open": dP_dgap_open_ref,
            "dP_dgap_ext": dP_dgap_ext_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, gap_open, gap_ext, temperature)
            assert_threaded_nw_affine_correctness(outputs, reference_outputs, thread_count)
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
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = d2p.soft_nw_affine_float(scores_cpu, gap_open, gap_ext, temperature, None)[1]
        posteriors_cuda = d2p.soft_nw_affine_float(scores_cuda, gap_open, gap_ext, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_score(self):
        """Test CPU vs CUDA scores match."""
        B, L1, L2 = 2, 10, 12
        gap_open, gap_ext = -1.5, -0.3
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        score_cpu = d2p.soft_nw_affine_float(scores_cpu, gap_open, gap_ext, temperature, None)[0]
        score_cuda = d2p.soft_nw_affine_float(scores_cuda, gap_open, gap_ext, temperature, None)[0]

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"

    def test_cuda_boundary_gap_gradients(self):
        """CUDA grad_open/grad_ext must include leading-gap boundary states (r26 regression)."""
        B, L1, L2 = 3, 5, 7
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(7)
        scores = torch.randn(B, L1, L2)
        # Push boundary-adjacent scores low to force beta mass onto boundaries
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5

        scores_cuda = scores.cuda()
        _, _, grad_open_cuda, grad_ext_cuda, grad_T_cuda = d2p.soft_nw_affine_with_grads(
            scores_cuda, gap_open, gap_ext, temperature, None
        )

        # Finite-difference reference (CPU naive)
        score_open_plus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open + eps, gap_ext, temperature
        )
        score_open_minus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open - eps, gap_ext, temperature
        )
        grad_open_ref = (score_open_plus - score_open_minus) / (2 * eps)

        score_ext_plus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext + eps, temperature
        )
        score_ext_minus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext - eps, temperature
        )
        grad_ext_ref = (score_ext_plus - score_ext_minus) / (2 * eps)

        score_T_plus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature + eps
        )
        score_T_minus, _, _, _ = soft_nw_affine_forward_naive(
            scores, gap_open, gap_ext, temperature - eps
        )
        grad_T_ref = (score_T_plus - score_T_minus) / (2 * eps)

        assert allclose(grad_open_ref, grad_open_cuda, rtol=1e-2, atol=2e-3), \
            f"CUDA grad_open boundary mismatch: max diff = {max_diff(grad_open_ref, grad_open_cuda)}"
        assert allclose(grad_ext_ref, grad_ext_cuda, rtol=1e-2, atol=2e-3), \
            f"CUDA grad_ext boundary mismatch: max diff = {max_diff(grad_ext_ref, grad_ext_cuda)}"
        assert allclose(grad_T_ref, grad_T_cuda, rtol=1e-2, atol=2e-3), \
            f"CUDA grad_T mismatch (boundary-heavy): max diff = {max_diff(grad_T_ref, grad_T_cuda)}"

    def test_cuda_with_grads_cpu_parity(self):
        """CUDA with_grads outputs must match CPU with_grads (r26 regression)."""
        B, L1, L2 = 4, 6, 8
        gap_open, gap_ext = -1.5, -0.3
        temperature = 1.0

        torch.manual_seed(99)
        scores = torch.randn(B, L1, L2)

        score_cpu, post_cpu, grad_open_cpu, grad_ext_cpu, grad_T_cpu = d2p.soft_nw_affine_with_grads(
            scores, gap_open, gap_ext, temperature, None
        )
        scores_cuda = scores.cuda()
        score_cuda, post_cuda, grad_open_cuda, grad_ext_cuda, grad_T_cuda = d2p.soft_nw_affine_with_grads(
            scores_cuda, gap_open, gap_ext, temperature, None
        )

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"
        assert allclose(post_cpu, post_cuda, rtol=1e-3, atol=1e-4), \
            f"CPU/CUDA posteriors mismatch: max diff = {max_diff(post_cpu, post_cuda)}"
        assert allclose(grad_open_cpu, grad_open_cuda, rtol=1e-2, atol=2e-3), \
            f"CPU/CUDA grad_open mismatch: max diff = {max_diff(grad_open_cpu, grad_open_cuda)}"
        assert allclose(grad_ext_cpu, grad_ext_cuda, rtol=1e-2, atol=2e-3), \
            f"CPU/CUDA grad_ext mismatch: max diff = {max_diff(grad_ext_cpu, grad_ext_cuda)}"
        assert allclose(grad_T_cpu, grad_T_cuda, rtol=1e-2, atol=2e-3), \
            f"CPU/CUDA grad_T mismatch: max diff = {max_diff(grad_T_cpu, grad_T_cuda)}"

    def test_cuda_param_jacobian_boundary_finite_diff(self):
        """CUDA param_jacobians must include boundary-U and terminal-W initialization (r44 regression)."""
        B, L1, L2 = 2, 5, 7
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, L1, L2)
        scores[:, 0, :] -= 2.5
        scores[:, :, 0] -= 2.5
        scores_cuda = scores.cuda()

        dP_dgap_open_cuda = d2p.soft_nw_affine_param_jacobian(
            scores_cuda, 0, gap_open, gap_ext, temperature, None
        )
        dP_dgap_ext_cuda = d2p.soft_nw_affine_param_jacobian(
            scores_cuda, 1, gap_open, gap_ext, temperature, None
        )
        dP_dT_cuda = d2p.soft_nw_affine_param_jacobian(
            scores_cuda, 2, gap_open, gap_ext, temperature, None
        )

        posteriors_gap_open_plus = soft_nw_affine_naive(
            scores, gap_open + eps, gap_ext, temperature
        )
        posteriors_gap_open_minus = soft_nw_affine_naive(
            scores, gap_open - eps, gap_ext, temperature
        )
        dP_dgap_open_ref = (posteriors_gap_open_plus - posteriors_gap_open_minus) / (2 * eps)

        posteriors_gap_ext_plus = soft_nw_affine_naive(
            scores, gap_open, gap_ext + eps, temperature
        )
        posteriors_gap_ext_minus = soft_nw_affine_naive(
            scores, gap_open, gap_ext - eps, temperature
        )
        dP_dgap_ext_ref = (posteriors_gap_ext_plus - posteriors_gap_ext_minus) / (2 * eps)

        posteriors_temp_plus = soft_nw_affine_naive(
            scores, gap_open, gap_ext, temperature + eps
        )
        posteriors_temp_minus = soft_nw_affine_naive(
            scores, gap_open, gap_ext, temperature - eps
        )
        dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

        assert allclose(dP_dgap_open_ref, dP_dgap_open_cuda, rtol=1e-2, atol=5e-3), \
            f"CUDA dP/dgap_open mismatch: max diff = {max_diff(dP_dgap_open_ref, dP_dgap_open_cuda)}"
        assert allclose(dP_dgap_ext_ref, dP_dgap_ext_cuda, rtol=1e-2, atol=5e-3), \
            f"CUDA dP/dgap_ext mismatch: max diff = {max_diff(dP_dgap_ext_ref, dP_dgap_ext_cuda)}"
        assert allclose(dP_dT_ref, dP_dT_cuda, rtol=1e-2, atol=5e-3), \
            f"CUDA dP/dT mismatch: max diff = {max_diff(dP_dT_ref, dP_dT_cuda)}"

    def test_cuda_param_jacobian_cpu_parity_with_variable_lengths(self):
        """CUDA param_jacobians must match CPU on masked batches with variable lengths."""
        B, max_L1, max_L2 = 3, 6, 7
        gap_open, gap_ext = -1.5, -0.3
        temperature = 0.7

        torch.manual_seed(321)
        scores = torch.randn(B, max_L1, max_L2)
        scores[0, 0, :] -= 2.0
        scores[1, :, 0] -= 1.5
        lengths = torch.tensor([[6, 7], [4, 5], [5, 3]], dtype=torch.int32)

        scores_cuda = scores.cuda()
        lengths_cuda = lengths.cuda()

        for param_type, name in ((0, "gap_open"), (1, "gap_ext"), (2, "temperature")):
            cpu = d2p.soft_nw_affine_param_jacobian(
                scores, param_type, gap_open, gap_ext, temperature, lengths
            )
            cuda = d2p.soft_nw_affine_param_jacobian(
                scores_cuda, param_type, gap_open, gap_ext, temperature, lengths_cuda
            )
            assert allclose(cpu, cuda, rtol=1e-2, atol=5e-3), \
                f"CPU/CUDA dP/d{name} mismatch: max diff = {max_diff(cpu, cuda)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        gap_open, gap_ext = -2.0, -0.5
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

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            score_ref, _, _, _ = soft_nw_affine_forward_naive(scores_b, gap_open, gap_ext, temperature)
            posteriors_ref = soft_nw_affine_naive(scores_b, gap_open, gap_ext, temperature)

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
            d2p.soft_nw_affine_float(scores, -2.0, -0.5, 1.0, lengths_t)

    def test_hvp_rejects_non_3d_scores_cpu(self):
        """CPU HVP scores must be rank-3 before sizes are read."""
        scores = torch.randn(4, 5)
        tangent = torch.randn(4, 5)

        with pytest.raises(RuntimeError, match=r"scores must be 3D"):
            d2p.soft_nw_affine_hvp(scores, tangent, -2.0, -0.5, 1.0, None)

    def test_hvp_rejects_mismatched_tangent_shape_cpu(self):
        """CPU HVP tangents must match the padded score tensor shape."""
        scores = torch.randn(1, 4, 5)
        tangent = torch.randn(1, 4, 4)

        with pytest.raises(RuntimeError, match=r"scores and tangent must have same shape"):
            d2p.soft_nw_affine_hvp(scores, tangent, -2.0, -0.5, 1.0, None)

    def test_param_jacobian_rejects_non_3d_scores_cpu(self):
        """CPU param_jacobian scores must be rank-3 before sizes are read."""
        scores = torch.randn(4, 5)

        with pytest.raises(RuntimeError, match=r"scores must be 3D"):
            d2p.soft_nw_affine_param_jacobian(scores, 0, -2.0, -0.5, 1.0, None)

    def test_param_jacobian_rejects_invalid_param_type_cpu(self):
        """CPU param_jacobian param_type must match the public CUDA contract."""
        scores = torch.randn(1, 4, 5)

        with pytest.raises(RuntimeError, match=r"param_type must be 0, 1, or 2"):
            d2p.soft_nw_affine_param_jacobian(scores, 3, -2.0, -0.5, 1.0, None)

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
    def test_cuda_lengths_must_match_scores_device(self):
        """Explicit CUDA lengths must live on the same device as scores."""
        scores = torch.randn(1, 4, 5, device="cuda:0")
        lengths_t = torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:1")

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            d2p.soft_nw_affine_float(scores, -2.0, -0.5, 1.0, lengths_t)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(score, score_ref), f"Single element score wrong: {score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), f"Single element posterior wrong: {posteriors.item()} vs {posteriors_ref.item()}"

    def test_row_vector(self, device):
        """Test 1xN score matrix."""
        scores = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(score, score_ref), "Row vector score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Row vector posteriors mismatch"

    def test_col_vector(self, device):
        """Test Nx1 score matrix."""
        scores = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(score, score_ref), "Column vector score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Column vector posteriors mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard NW)."""
        B, L1, L2 = 2, 6, 8
        gap_open, gap_ext = -2.0, -0.5
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        gap_open, gap_ext = -2.0, -0.5
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)[1]
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_equal_gap_penalties(self, device):
        """Test with gap_open == gap_ext (effectively linear gap)."""
        B, L1, L2 = 2, 6, 6
        gap_open, gap_ext = -1.0, -1.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(score, score_ref), f"Equal gap score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3, atol=1e-4), \
            "Equal gap posteriors mismatch"

    def test_zero_gap_ext(self, device):
        """Test with zero gap extension penalty."""
        B, L1, L2 = 2, 6, 6
        gap_open, gap_ext = -2.0, 0.0
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)
        score_ref, _, _, _ = soft_nw_affine_forward_naive(scores, gap_open, gap_ext, temperature)
        posteriors_ref = soft_nw_affine_naive(scores, gap_open, gap_ext, temperature)

        assert allclose(score, score_ref), f"Zero gap_ext score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3, atol=1e-4), \
            "Zero gap_ext posteriors mismatch"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test soft_nw_affine_with_grads returns correct values."""
        B, L1, L2 = 2, 6, 8
        gap_open, gap_ext = -2.0, -0.5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, L1, L2, device=device)

        score, posteriors, grad_gap_open, grad_gap_ext, grad_T = d2p.soft_nw_affine_with_grads(
            scores, gap_open, gap_ext, temperature, None
        )

        # Compare with regular function
        score_ref, posteriors_ref = d2p.soft_nw_affine_float(scores, gap_open, gap_ext, temperature, None)

        assert allclose(score, score_ref), "with_grads score mismatch"
        assert allclose(posteriors, posteriors_ref), "with_grads posteriors mismatch"

        # grad_gap_open, grad_gap_ext, and grad_T should be [B] tensors
        assert grad_gap_open.shape == (B,), f"grad_gap_open shape wrong: {grad_gap_open.shape}"
        assert grad_gap_ext.shape == (B,), f"grad_gap_ext shape wrong: {grad_gap_ext.shape}"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


# --- Memory-safety regression tests (merged from test_nw_affine_{cpp,cuda}_memsafety.py) ---

GAP_OPEN = -2.0
GAP_EXT = -0.5
TEMPERATURE = 1.0


def _nw_affine_cpu_scores():
    torch.manual_seed(20260707)
    return torch.randn(1, 4, 5)


def _nw_affine_cuda_scores(*shape, requires_grad=False):
    torch.manual_seed(20260707)
    return torch.randn(*shape, device="cuda", requires_grad=requires_grad)


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
    relevant = sorted(name for name in kernel_names if "nw_affine_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; nw_affine kernels seen: {relevant[:20]}"
    )


# --- CPU memory-safety regressions (from test_nw_affine_cpp_memsafety.py) ---


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_with_grads_rejects_rank4_scores_cpu():
    scores = torch.randn(1, 4, 5, 2)

    with pytest.raises(RuntimeError, match=r"scores must be 3D"):
        d2p.soft_nw_affine_with_grads(scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_backward_full_rejects_rank4_scores_cpu():
    scores = torch.randn(1, 4, 5, 2)
    grad_alignment = torch.randn(1, 4, 5, 2)

    with pytest.raises(RuntimeError, match=r"scores must be 3D"):
        d2p.soft_nw_affine_backward_full(
            scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_backward_full_rejects_mismatched_grad_alignment_cpu():
    scores = _nw_affine_cpu_scores()
    grad_alignment = torch.randn(1, 4, 4)

    with pytest.raises(RuntimeError, match=r"grad_alignment must have same shape as scores"):
        d2p.soft_nw_affine_backward_full(
            scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_namespaced_marginals_backward_rejects_mismatched_grad_marginals_cpu():
    scores = _nw_affine_cpu_scores()
    grad_marginals = torch.randn(1, 4, 4)

    with pytest.raises(RuntimeError, match=r"grad_alignment must have same shape as scores"):
        torch.ops.d2p.nw_affine_marginals_backward(
            scores, grad_marginals, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_backward_full_accepts_valid_noncontiguous_grad_alignment_cpu():
    scores = _nw_affine_cpu_scores()
    grad_alignment = torch.randn(1, 5, 4).transpose(1, 2)
    assert grad_alignment.shape == scores.shape
    assert not grad_alignment.is_contiguous()

    grad_scores, grad_open, grad_ext, grad_temp = d2p.soft_nw_affine_backward_full(
        scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )

    assert grad_scores.shape == scores.shape
    assert grad_open.shape == (1,)
    assert grad_ext.shape == (1,)
    assert grad_temp.shape == (1,)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_autograd_rejects_mismatched_grad_posteriors_cpu():
    scores = _nw_affine_cpu_scores().requires_grad_(True)
    _, posteriors = d2p.soft_nw_affine_float(scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None)
    bad_grad_posteriors = torch.randn(1, 4, 4)

    with pytest.raises(RuntimeError, match=r"Mismatch in shape|grad_posteriors"):
        posteriors.backward(bad_grad_posteriors)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_workspace_overflow_shape_rejected_before_allocation_cpu():
    scores = torch.empty((1, 0, 715_827_882), dtype=torch.float32)

    with pytest.raises(
        RuntimeError,
        match=re.escape("nw_affine CPU DP workspace is too large"),
    ):
        d2p.soft_nw_affine_with_grads(scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_valid_cpu_outputs_match_frontdoor_after_guards():
    scores = _nw_affine_cpu_scores()

    score, posteriors, grad_open, grad_ext, grad_temp = d2p.soft_nw_affine_with_grads(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )
    score_ref, posteriors_ref = d2p.soft_nw_affine_float(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )

    assert torch.allclose(score, score_ref)
    assert torch.allclose(posteriors, posteriors_ref)
    assert grad_open.shape == (1,)
    assert grad_ext.shape == (1,)
    assert grad_temp.shape == (1,)


# --- CUDA memory-safety regressions (from test_nw_affine_cuda_memsafety.py) ---

_NW_AFFINE_CUDA_REASON = "nw_affine CUDA memsafety regressions require a CUDA d2p build"


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_with_grads_rejects_non_3d_scores_cuda():
    scores = _nw_affine_cuda_scores(1, 2, 3, 4)

    with pytest.raises(RuntimeError, match="scores must be 3D"):
        d2p.soft_nw_affine_with_grads(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_with_grads_rejects_oversized_padded_dimension_cuda():
    try:
        scores = torch.empty((1, 0, 2**31 - 1), device="cuda")
    except RuntimeError as exc:
        pytest.skip(f"PyTorch rejected the zero-sized oversized tensor: {exc}")

    with pytest.raises(RuntimeError, match="alpha workspace size must fit in int32"):
        d2p.soft_nw_affine_with_grads(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_hvp_rejects_mismatched_tangent_shape_cuda():
    scores = _nw_affine_cuda_scores(1, 4, 5)
    tangent = _nw_affine_cuda_scores(1, 4, 4)

    with pytest.raises(RuntimeError, match="scores and tangent must have same shape"):
        d2p.soft_nw_affine_hvp(
            scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_hvp_rejects_tangent_on_wrong_cuda_device():
    scores = torch.randn(1, 4, 5, device="cuda:0")
    tangent = torch.randn(1, 4, 5, device="cuda:1")

    with pytest.raises(RuntimeError, match="tangent must be on same device as scores"):
        d2p.soft_nw_affine_hvp(
            scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_hvp_uses_scores_device_guard():
    scores = torch.randn(1, 4, 5, device="cuda:0")
    tangent = torch.randn(1, 4, 5, device="cuda:0")

    with torch.cuda.device(1):
        result = d2p.soft_nw_affine_hvp(
            scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )

    assert result.device == scores.device


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_backward_full_rejects_cpu_grad_alignment_for_cuda_scores():
    scores = _nw_affine_cuda_scores(1, 4, 5)
    grad_alignment = torch.randn(1, 4, 5)

    # The CUDA backend names this argument "grad" in its device check, so the
    # device-mismatch rejection reads "grad must be a CUDA tensor".
    with pytest.raises(RuntimeError, match=r"grad must be a CUDA tensor"):
        d2p.soft_nw_affine_backward_full(
            scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_backward_full_rejects_mismatched_grad_alignment_shape_cuda():
    scores = _nw_affine_cuda_scores(1, 4, 5)
    grad_alignment = _nw_affine_cuda_scores(1, 4, 4)

    with pytest.raises(RuntimeError, match="grad_alignment must have same shape as scores"):
        d2p.soft_nw_affine_backward_full(
            scores, grad_alignment, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_marginals_backward_rejects_mismatched_grad_marginals_shape_cuda():
    scores = _nw_affine_cuda_scores(1, 4, 5)
    grad_marginals = _nw_affine_cuda_scores(1, 4, 4)

    with pytest.raises(RuntimeError, match="must have same shape as scores"):
        torch.ops.d2p.nw_affine_marginals_backward(
            scores, grad_marginals, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_autograd_backward_rejects_bad_grad_posteriors_shape_cuda():
    scores = _nw_affine_cuda_scores(1, 4, 5, requires_grad=True)
    score, posteriors = d2p.soft_nw_affine_float(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )
    bad_grad_posteriors = _nw_affine_cuda_scores(1, 4, 4)

    # The mismatched grad_posteriors shape is correctly detected and rejected by
    # torch.autograd's _make_grads before the op backward runs. On this torch
    # version, formatting the "Mismatch in shape" message internally calls
    # list.index() on multi-element grad tensors, which surfaces as "Boolean
    # value of Tensor ... is ambiguous". Accept any of these rejection messages.
    with pytest.raises(
        RuntimeError,
        match=r"Mismatch in shape|must have same shape|Boolean value of Tensor",
    ):
        torch.autograd.grad(
            (score, posteriors),
            scores,
            grad_outputs=(torch.ones_like(score), bad_grad_posteriors),
        )


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_score_backward_skips_unused_posterior_hvp():
    def score_only_backward():
        scores = _nw_affine_cuda_scores(1, 8, 9, requires_grad=True)
        score, _ = d2p.soft_nw_affine_float(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )
        score.sum().backward()

    def explicit_zero_posterior_backward():
        scores = _nw_affine_cuda_scores(1, 8, 9, requires_grad=True)
        score, posteriors = d2p.soft_nw_affine_float(
            scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
        )
        (score.sum() + 0.0 * posteriors.sum()).backward()

    score_only_kernels = cuda_kernel_names(score_only_backward)
    assert_cuda_kernel_seen(score_only_kernels, "nw_affine_forward_diag_kernel")
    assert not any("nw_affine_hvp_" in name for name in score_only_kernels)
    assert not any("nw_affine_param_grad_" in name for name in score_only_kernels)

    explicit_zero_kernels = cuda_kernel_names(explicit_zero_posterior_backward)
    assert_cuda_kernel_seen(explicit_zero_kernels, "nw_affine_hvp_forward_diag_kernel")
    assert_cuda_kernel_seen(explicit_zero_kernels, "nw_affine_param_grad_forward_diag_kernel")


@pytest.mark.skipif(not (D2P_AVAILABLE and CUDA_AVAILABLE), reason=_NW_AFFINE_CUDA_REASON)
def test_valid_cuda_inputs_still_match_primary_surfaces():
    scores = _nw_affine_cuda_scores(2, 5, 6)
    tangent = _nw_affine_cuda_scores(2, 5, 6)

    score, posteriors = d2p.soft_nw_affine_float(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )
    with_grads = d2p.soft_nw_affine_with_grads(
        scores, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )
    hvp = d2p.soft_nw_affine_hvp(
        scores, tangent, GAP_OPEN, GAP_EXT, TEMPERATURE, None
    )

    assert torch.allclose(with_grads[0], score)
    assert torch.allclose(with_grads[1], posteriors)
    assert hvp.shape == scores.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
