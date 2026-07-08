"""
Correctness tests for Soft LCS (Longest Common Subsequence).
"""

import contextlib
import importlib
import os
import re
import site
import sys
from pathlib import Path

import pytest
import torch

from reference import soft_lcs_forward_naive, soft_lcs_naive


def _skip_stale_d2p_editable_loader():
    editable_paths = []
    for site_dir in site.getsitepackages():
        loader_path = Path(site_dir) / "_d2p_editable_loader.py"
        if not loader_path.exists():
            continue
        match = re.search(
            r"install\(\s*'d2p',\s*\{'d2p'\},\s*'([^']+)'",
            loader_path.read_text(),
            re.S,
        )
        if match:
            editable_paths.append(match.group(1))
    if not editable_paths:
        return

    existing = [path for path in os.environ.get("MESONPY_EDITABLE_SKIP", "").split(os.pathsep) if path]
    for path in editable_paths:
        if path not in existing:
            existing.append(path)
    os.environ["MESONPY_EDITABLE_SKIP"] = os.pathsep.join(existing)


_skip_stale_d2p_editable_loader()

try:
    import d2p
    D2P_AVAILABLE = True
except ImportError:
    sys.modules.pop("d2p", None)
    source_root = str(Path(__file__).resolve().parents[1])
    sys.path = [path for path in sys.path if path != source_root]
    for site_dir in reversed(site.getsitepackages()):
        if site_dir in sys.path:
            sys.path.remove(site_dir)
        sys.path.insert(0, site_dir)
    try:
        d2p = importlib.import_module("d2p")
        D2P_AVAILABLE = True
    except ImportError:
        D2P_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()


def allclose(a, b, rtol=1e-4, atol=1e-5):
    return torch.allclose(a.cpu(), b.cpu(), rtol=rtol, atol=atol)


def max_diff(a, b):
    return (a.cpu() - b.cpu()).abs().max().item()


def require_raises(fn, match):
    with pytest.raises(RuntimeError, match=match):
        fn()


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, temperature):
    lcs_score, posteriors, grad_T = d2p.soft_lcs_with_grads(scores, temperature, None)
    hvp = d2p.soft_lcs_hvp(scores, tangent, temperature, None)
    dP_dT = d2p.soft_lcs_param_jacobian(scores, temperature, None)
    return {
        "lcs_score": lcs_score,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_lcs_correctness(outputs, reference_outputs, thread_count):
    lcs_score_ref = reference_outputs["lcs_score"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(lcs_score_ref, outputs["lcs_score"]), \
        f"{thread_count}-thread score mismatch: max diff = {max_diff(lcs_score_ref, outputs['lcs_score'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
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


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_lcs_score(self, batch_size, seq_lengths, temperature, device):
        """Test that LCS scores match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        # Create match score matrix (higher for matches)
        scores = torch.rand(batch_size, L1, L2, device=device)

        score_ref, _ = soft_lcs_forward_naive(scores, temperature)
        score_d2p = d2p.soft_lcs_float(scores, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"LCS score mismatch: max diff = {max_diff(score_ref, score_d2p)}"

    def test_lcs_score_matches_only(self, batch_size, temperature, device):
        """Test with explicit match/mismatch pattern."""
        L1, L2 = 10, 12

        torch.manual_seed(42)
        # Binary scores: 1 for match, 0 for mismatch
        scores = torch.zeros(batch_size, L1, L2, device=device)
        # Add some matches on diagonal-ish pattern
        for i in range(min(L1, L2)):
            scores[:, i, i] = 1.0

        score_ref, _ = soft_lcs_forward_naive(scores, temperature)
        score_d2p = d2p.soft_lcs_float(scores, temperature, None)[0]

        assert allclose(score_ref, score_d2p), \
            f"LCS score mismatch (matches only): max diff = {max_diff(score_ref, score_d2p)}"

    def test_lcs_known_pattern(self, device):
        """Test on known pattern: identical sequences should have high LCS."""
        B = 2
        L = 5
        temperature = 1.0

        # First batch: matches on diagonal (identical sequences)
        # Second batch: all zeros (different sequences)
        scores = torch.zeros(B, L, L, device=device)
        for i in range(L):
            scores[0, i, i] = 1.0

        score, _ = d2p.soft_lcs_float(scores, temperature, None)

        # First batch (matches on diagonal) should have higher score
        assert score[0] > score[1], "Identical sequences should have higher LCS score"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)

        posteriors_ref = soft_lcs_naive(scores, temperature)
        posteriors_d2p = d2p.soft_lcs_float(scores, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        posteriors = d2p.soft_lcs_float(scores, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = soft_lcs_naive(scores_ref, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_returns_temp_grad(self, device):
        """Test that soft_lcs_with_grads returns temperature gradient."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        lcs_score, posteriors, grad_T = d2p.soft_lcs_with_grads(
            scores, temperature, None
        )

        # Check shapes
        assert lcs_score.shape == (B,)
        assert posteriors.shape == (B, L1, L2)
        assert grad_T.shape == (B,)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_lcs_hvp(scores, V, temperature, None)

        posteriors_plus = soft_lcs_naive(scores + eps * V, temperature)
        posteriors_minus = soft_lcs_naive(scores - eps * V, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_various_temps(self, device):
        """Test HVP at various temperatures."""
        B, L1, L2 = 2, 6, 7
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.rand(B, L1, L2, device=device)
            V = torch.randn(B, L1, L2, device=device)

            hvp_d2p = d2p.soft_lcs_hvp(scores, V, temperature, None)

            posteriors_plus = soft_lcs_naive(scores + eps * V, temperature)
            posteriors_minus = soft_lcs_naive(scores - eps * V, temperature)
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            # Slightly more relaxed tolerance for finite differences (O(eps^2) error)
            assert allclose(hvp_fd, hvp_d2p, rtol=2e-2, atol=5e-3), \
                f"HVP mismatch at T={temperature}: max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.rand(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            lcs_score_ref, _ = soft_lcs_forward_naive(scores, temperature)
            posteriors_ref = soft_lcs_naive(scores, temperature)

            lcs_score_plus, _ = soft_lcs_forward_naive(scores, temperature + eps)
            lcs_score_minus, _ = soft_lcs_forward_naive(scores, temperature - eps)
            grad_T_ref = (lcs_score_plus - lcs_score_minus) / (2 * eps)

            posteriors_plus = soft_lcs_naive(scores + eps * tangent, temperature)
            posteriors_minus = soft_lcs_naive(scores - eps * tangent, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            posteriors_temp_plus = soft_lcs_naive(scores, temperature + eps)
            posteriors_temp_minus = soft_lcs_naive(scores, temperature - eps)
            dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

        reference_outputs = {
            "lcs_score": lcs_score_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, temperature)
            assert_threaded_lcs_correctness(outputs, reference_outputs, thread_count)
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
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.rand(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = d2p.soft_lcs_float(scores_cpu, temperature, None)[1]
        posteriors_cuda = d2p.soft_lcs_float(scores_cuda, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_various_temps(self):
        """Test CPU vs CUDA at various temperatures."""
        B, L1, L2 = 2, 10, 12

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores_cpu = torch.rand(B, L1, L2)
            scores_cuda = scores_cpu.cuda()

            posteriors_cpu = d2p.soft_lcs_float(scores_cpu, temperature, None)[1]
            posteriors_cuda = d2p.soft_lcs_float(scores_cuda, temperature, None)[1]

            assert allclose(posteriors_cpu, posteriors_cuda), \
                f"CPU/CUDA mismatch at T={temperature}: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_score_autograd_matches_cpu(self):
        """CUDA autograd should include the score path, not just the posterior path."""
        B, L1, L2 = 2, 5, 6
        temperature = 0.9

        torch.manual_seed(123)
        scores_cpu = torch.rand(B, L1, L2, dtype=torch.float32, requires_grad=True)
        scores_cuda = scores_cpu.detach().clone().cuda().requires_grad_(True)

        score_cpu = d2p.soft_lcs_float(scores_cpu, temperature, None)[0]
        score_cuda = d2p.soft_lcs_float(scores_cuda, temperature, None)[0]

        score_cpu.sum().backward()
        score_cuda.sum().backward()

        assert allclose(scores_cpu.grad, scores_cuda.grad, rtol=1e-3, atol=1e-4), \
            f"Score-path gradient mismatch: max diff = {max_diff(scores_cpu.grad, scores_cuda.grad)}"

    def test_backward_full_consistency(self):
        """Explicit backward_full should match the CPU dP/dT contraction on CUDA."""
        B, L1, L2 = 2, 6, 7
        temperature = 0.9

        torch.manual_seed(321)
        scores_cpu = torch.rand(B, L1, L2, dtype=torch.float32)
        grad_cpu = torch.randn(B, L1, L2, dtype=torch.float32)
        lengths_cpu = torch.tensor([[6, 7], [4, 5]], dtype=torch.int32)

        scores_cuda = scores_cpu.cuda()
        grad_cuda = grad_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        backward_cpu = d2p.soft_lcs_backward_full(scores_cpu, grad_cpu, temperature, lengths_cpu)
        backward_cuda = d2p.soft_lcs_backward_full(scores_cuda, grad_cuda, temperature, lengths_cuda)

        assert allclose(backward_cpu[0], backward_cuda[0], rtol=2e-2, atol=5e-3), \
            f"backward_full grad_scores mismatch: max diff = {max_diff(backward_cpu[0], backward_cuda[0])}"
        assert allclose(backward_cpu[1], backward_cuda[1], rtol=2e-2, atol=5e-3), \
            f"backward_full grad_T mismatch: max diff = {max_diff(backward_cpu[1], backward_cuda[1])}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            score_ref, _ = soft_lcs_forward_naive(scores_b, temperature)
            posteriors_ref = soft_lcs_naive(scores_b, temperature)

            # Score should match for this sequence
            assert allclose(score_ref, lcs_score[b:b+1]), \
                f"LCS score mismatch for batch {b}: {score_ref.item()} vs {lcs_score[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, None)
        score_ref, _ = soft_lcs_forward_naive(scores, temperature)
        posteriors_ref = soft_lcs_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), f"Single element score wrong: {lcs_score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), "Single element posterior wrong"

    def test_row_vector(self, device):
        """Test 1xN score matrix."""
        scores = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, None)
        score_ref, _ = soft_lcs_forward_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), "Row vector score mismatch"

    def test_col_vector(self, device):
        """Test Nx1 score matrix."""
        scores = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        temperature = 1.0

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, None)
        score_ref, _ = soft_lcs_forward_naive(scores, temperature)

        assert allclose(lcs_score, score_ref), "Column vector score mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard LCS)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = d2p.soft_lcs_float(scores, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = d2p.soft_lcs_float(scores, temperature, None)[1]
        posteriors_ref = soft_lcs_naive(scores, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_temperature_clamp(self, device):
        """Test that very low temperature doesn't cause NaN."""
        B, L1, L2 = 2, 5, 6
        temperature = 1e-6

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, None)

        assert not torch.isnan(lcs_score).any(), "LCS score contains NaN"
        assert not torch.isnan(posteriors).any(), "Posteriors contain NaN"
        assert not torch.isinf(lcs_score).any(), "LCS score contains Inf"

    def test_all_matches(self, device):
        """Test with all matches (identical sequences on diagonal)."""
        B, L = 2, 5
        temperature = 1.0

        # High match scores on diagonal
        scores = torch.zeros(B, L, L, device=device)
        for i in range(L):
            scores[:, i, i] = 1.0

        lcs_score, posteriors = d2p.soft_lcs_float(scores, temperature, None)

        # With matches on diagonal, LCS should be close to L for low temperature
        # At T=1.0, it will be somewhat higher due to softmax aggregation
        assert lcs_score.mean() > 0, "LCS score should be positive"

    def test_invalid_lengths_rejected(self, device):
        """Lengths must stay within the padded score tensor bounds."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_lengths = torch.tensor([[L1 + 1, L2], [L1, L2]], dtype=torch.int32, device=device)

        require_raises(
            lambda: d2p.soft_lcs_float(scores, temperature, bad_lengths),
            rf"lengths\[0,0\] must be between 0 and {L1}",
        )

    def test_hvp_rejects_bad_tangent_shape(self, device):
        """HVP tangents must match the score tensor shape."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_tangent = torch.rand(B, L1 - 1, L2, device=device)

        require_raises(
            lambda: d2p.soft_lcs_hvp(scores, bad_tangent, temperature, None),
            "scores and tangent must have same shape",
        )

    def test_backward_full_rejects_bad_grad_shape(self, device):
        """Explicit backward inputs must match the posterior tensor shape."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0

        scores = torch.rand(B, L1, L2, device=device)
        bad_grad = torch.rand(B, L1 - 1, L2, device=device)

        require_raises(
            lambda: d2p.soft_lcs_backward_full(scores, bad_grad, temperature, None),
            "grad_posteriors must have same shape as scores",
        )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestParamJacobian:

    def test_param_jacobian_temperature(self, device):
        """Test parameter Jacobian for temperature."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        # dP/dT via param_jacobian
        dP_dT = d2p.soft_lcs_param_jacobian(scores, temperature, None)

        # Finite diff
        posteriors_plus = soft_lcs_naive(scores, temperature + eps)
        posteriors_minus = soft_lcs_naive(scores, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    def test_param_jacobian_various_temps(self, device):
        """Test parameter Jacobian at various temperatures."""
        B, L1, L2 = 2, 5, 6
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.rand(B, L1, L2, device=device)

            dP_dT = d2p.soft_lcs_param_jacobian(scores, temperature, None)

            posteriors_plus = soft_lcs_naive(scores, temperature + eps)
            posteriors_minus = soft_lcs_naive(scores, temperature - eps)
            dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
                f"Param Jacobian (T={temperature}) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"


# --- Memory-safety regression tests (merged from test_lcs_{cpp,cuda}_memsafety.py) ---

OVERSIZED_LCS_SHAPE = (1, 65535, 65535)
OVERSIZED_ERROR = r"LCS workspace too large"


def _oversized_inputs():
    # Real, contiguous, correctly-typed oversized tensors. The workspace-size
    # guard rejects purely on shape metadata before any allocation or data read,
    # so the ~16 GiB backing storage is never faulted in (it stays virtual).
    # A single scores buffer is reused for the second (tangent / grad) argument
    # because the scores guard fires first; this avoids a second large mapping.
    scores = torch.empty(OVERSIZED_LCS_SHAPE, dtype=torch.float32)
    lengths = torch.empty((1, 2), dtype=torch.int32)
    temperature = torch.empty((1,), dtype=torch.float32)
    return scores, scores, lengths, temperature


def _call_soft_lcs(scores, same_shape, lengths, temperature):
    del same_shape
    return torch.ops.d2p.soft_lcs(scores, temperature, lengths)


def _call_soft_lcs_float(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return torch.ops.d2p.soft_lcs_float(scores, 1.0, None)


def _call_soft_lcs_with_grads(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return torch.ops.d2p.soft_lcs_with_grads(scores, 1.0, None)


def _call_soft_lcs_hvp(scores, same_shape, lengths, temperature):
    del lengths, temperature
    return torch.ops.d2p.soft_lcs_hvp(scores, same_shape, 1.0, None)


def _call_soft_lcs_param_jacobian(scores, same_shape, lengths, temperature):
    del same_shape, lengths, temperature
    return torch.ops.d2p.soft_lcs_param_jacobian(scores, 1.0, None)


def _call_soft_lcs_backward_full(scores, same_shape, lengths, temperature):
    del lengths, temperature
    return torch.ops.d2p.soft_lcs_backward_full(scores, same_shape, 1.0, None)


OVERSIZED_WORKSPACE_CALLS = [
    ("soft_lcs", _call_soft_lcs),
    ("soft_lcs_float", _call_soft_lcs_float),
    ("soft_lcs_with_grads", _call_soft_lcs_with_grads),
    ("soft_lcs_hvp", _call_soft_lcs_hvp),
    ("soft_lcs_param_jacobian", _call_soft_lcs_param_jacobian),
    ("soft_lcs_backward_full", _call_soft_lcs_backward_full),
]


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.parametrize(
    ("name", "call"),
    OVERSIZED_WORKSPACE_CALLS,
    ids=[name for name, _ in OVERSIZED_WORKSPACE_CALLS],
)
def test_oversized_lcs_cpu_workspace_rejected_before_allocation(name, call):
    del name
    scores, same_shape, lengths, temperature = _oversized_inputs()
    with pytest.raises(RuntimeError, match=OVERSIZED_ERROR):
        call(scores, same_shape, lengths, temperature)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_valid_lcs_cpu_outputs_unchanged():
    scores = torch.tensor(
        [
            [
                [0.7, 0.1, 0.2],
                [0.2, 0.9, 0.3],
                [0.4, 0.3, 0.8],
            ]
        ],
        dtype=torch.float32,
    )
    tangent = torch.randn_like(scores)
    temperature = 1.0

    score, posteriors = torch.ops.d2p.soft_lcs_float(scores, temperature, None)
    score_dbg, posteriors_dbg, grad_T = torch.ops.d2p.soft_lcs_with_grads(scores, temperature, None)
    hvp = torch.ops.d2p.soft_lcs_hvp(scores, tangent, temperature, None)
    dP_dT = torch.ops.d2p.soft_lcs_param_jacobian(scores, temperature, None)
    grad_scores, grad_temperature = torch.ops.d2p.soft_lcs_backward_full(scores, tangent, temperature, None)

    torch.testing.assert_close(score, score_dbg)
    torch.testing.assert_close(posteriors, posteriors_dbg)
    assert grad_T.shape == (1,)
    assert hvp.shape == scores.shape
    assert dP_dT.shape == scores.shape
    assert grad_scores.shape == scores.shape
    assert grad_temperature.shape == (1,)


def _oversized_logical_scores():
    """A huge logical LCS table with one-float storage.

    The zero strides avoid allocating the 16+ GiB score matrix, while the wrapper
    still sees the dangerous L=65535 dimensions from I-18.
    """
    base = torch.empty((1,), device="cuda", dtype=torch.float32)
    return base.as_strided((1, 65535, 65535), (0, 0, 0))


def _assert_rejects_oversized_dp_table(call):
    scores = _oversized_logical_scores()
    with pytest.raises(RuntimeError, match="LCS CUDA DP table is too large"):
        call(scores)


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA d2p backend",
)
@pytest.mark.parametrize(
    "call",
    [
        lambda scores: d2p.soft_lcs(scores, torch.tensor(1.0, device=scores.device), None),
        lambda scores: d2p.soft_lcs_float(scores, 1.0, None),
        lambda scores: d2p.soft_lcs_with_grads(scores, 1.0, None),
        lambda scores: d2p.soft_lcs_hvp(scores, scores, 1.0, None),
        lambda scores: d2p.soft_lcs_param_jacobian(scores, 1.0, None),
        lambda scores: d2p.soft_lcs_backward_full(scores, scores, 1.0, None),
    ],
)
def test_lcs_cuda_rejects_dp_table_int32_overflow_before_allocation(call):
    _assert_rejects_oversized_dp_table(call)


def _backward_peak_bytes(loss_kind, size=512):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    scores = torch.randn(1, size, size, device="cuda", requires_grad=True)
    lcs_score, posteriors = d2p.soft_lcs_float(scores, 1.0, None)
    if loss_kind == "score":
        loss = lcs_score.sum()
    else:
        loss = posteriors.square().sum()
    loss.backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_lcs_cuda_score_only_backward_skips_materialized_hvp_path():
    size = 512
    score_only_peak = _backward_peak_bytes("score", size=size)
    posterior_peak = _backward_peak_bytes("posterior", size=size)

    alpha_bytes = (size + 1) * (size + 1) * 4
    assert posterior_peak - score_only_peak > 2 * alpha_bytes


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="LCS CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_lcs_cuda_valid_small_input_still_matches_cpu():
    torch.manual_seed(20260707)
    scores_cpu = torch.rand(2, 8, 9)
    scores_cuda = scores_cpu.cuda()
    lengths_cpu = torch.tensor([[8, 9], [6, 7]], dtype=torch.int32)
    lengths_cuda = lengths_cpu.cuda()

    score_cpu, post_cpu = d2p.soft_lcs_float(scores_cpu, 1.0, lengths_cpu)
    score_cuda, post_cuda = d2p.soft_lcs_float(scores_cuda, 1.0, lengths_cuda)

    assert torch.allclose(score_cpu, score_cuda.cpu(), rtol=1e-4, atol=1e-5)
    assert torch.allclose(post_cpu, post_cuda.cpu(), rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
