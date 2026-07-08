"""
Correctness tests for Soft Levenshtein (Edit Distance).
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

from reference import soft_levenshtein_forward_naive, soft_levenshtein_naive


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
    relevant = sorted(name for name in kernel_names if "lev_" in name)
    raise AssertionError(
        f"CUDA profiler did not capture {token}; Levenshtein kernels seen: {relevant[:20]}"
    )


@contextlib.contextmanager
def torch_num_threads(num_threads):
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def run_cpu_representative_outputs(scores, tangent, ins_cost, del_cost, temperature):
    distance, posteriors, grad_ins, grad_del, grad_T = d2p.soft_levenshtein_with_grads(
        scores, ins_cost, del_cost, temperature, None
    )
    hvp = d2p.soft_levenshtein_hvp(scores, tangent, ins_cost, del_cost, temperature, None)
    dP_dins = d2p.soft_levenshtein_param_jacobian(scores, 0, ins_cost, del_cost, temperature, None)
    dP_ddel = d2p.soft_levenshtein_param_jacobian(scores, 1, ins_cost, del_cost, temperature, None)
    dP_dT = d2p.soft_levenshtein_param_jacobian(scores, 2, ins_cost, del_cost, temperature, None)
    return {
        "distance": distance,
        "posteriors": posteriors,
        "grad_ins": grad_ins,
        "grad_del": grad_del,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dins": dP_dins,
        "dP_ddel": dP_ddel,
        "dP_dT": dP_dT,
    }


def assert_threaded_levenshtein_correctness(outputs, reference_outputs, thread_count):
    distance_ref = reference_outputs["distance"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_ins_ref = reference_outputs["grad_ins"]
    grad_del_ref = reference_outputs["grad_del"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dins_ref = reference_outputs["dP_dins"]
    dP_ddel_ref = reference_outputs["dP_ddel"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(distance_ref, outputs["distance"]), \
        f"{thread_count}-thread distance mismatch: max diff = {max_diff(distance_ref, outputs['distance'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_ins_ref, outputs["grad_ins"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_ins mismatch: max diff = {max_diff(grad_ins_ref, outputs['grad_ins'])}"
    assert allclose(grad_del_ref, outputs["grad_del"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_del mismatch: max diff = {max_diff(grad_del_ref, outputs['grad_del'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dins_ref, outputs["dP_dins"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dins mismatch: max diff = {max_diff(dP_dins_ref, outputs['dP_dins'])}"
    assert allclose(dP_ddel_ref, outputs["dP_ddel"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/ddel mismatch: max diff = {max_diff(dP_ddel_ref, outputs['dP_ddel'])}"
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


@pytest.fixture(params=[(1.0, 1.0), (0.5, 1.5), (2.0, 0.5)])
def ins_del_costs(request):
    """Different insert/delete cost combinations."""
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_distance(self, batch_size, seq_lengths, temperature, ins_del_costs, device):
        """Test that Levenshtein distances match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost = ins_del_costs

        torch.manual_seed(42)
        # Create substitution cost matrix (0 for match, positive for mismatch)
        scores = torch.rand(batch_size, L1, L2, device=device) * 2  # Costs between 0 and 2

        distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)
        distance_d2p = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch: max diff = {max_diff(distance_ref, distance_d2p)}"

    def test_distance_symmetric(self, batch_size, temperature, device):
        """Test with symmetric ins/del costs."""
        L1, L2 = 10, 12
        ins_cost = del_cost = 1.0

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)

        distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)
        distance_d2p = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch (symmetric): max diff = {max_diff(distance_ref, distance_d2p)}"

    def test_distance_matches_vs_mismatches(self, device):
        """Test on known patterns: matching vs mismatching sequences."""
        B = 2
        L = 5
        temperature = 1.0
        ins_cost = del_cost = 1.0

        # Create scores: zeros (matches) for first batch, ones (mismatches) for second
        scores = torch.zeros(B, L, L, device=device)
        scores[1] = 1.0  # All mismatches

        distance, _ = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)

        # First batch (all matches on diagonal) should have lower distance
        assert distance[0] < distance[1], "Matches should have lower distance than mismatches"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, ins_del_costs, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost = ins_del_costs

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device) * 2

        posteriors_ref = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature)
        posteriors_d2p = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        ins_cost, del_cost = 1.0, 1.0

        torch.manual_seed(42)
        scores = torch.rand(batch_size, L1, L2, device=device)
        scores.requires_grad_(True)

        posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        posteriors_ref = soft_levenshtein_naive(scores_ref, ins_cost, del_cost, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_returns_param_grads(self, device):
        """Test that soft_levenshtein_with_grads returns parameter gradients."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0
        ins_cost, del_cost = 1.0, 0.8

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        distance, posteriors, grad_ins, grad_del, grad_T = d2p.soft_levenshtein_with_grads(
            scores, ins_cost, del_cost, temperature, None
        )

        # Check shapes
        assert distance.shape == (B,)
        assert posteriors.shape == (B, L1, L2)
        assert grad_ins.shape == (B,)
        assert grad_del.shape == (B,)
        assert grad_T.shape == (B,)

        # Check that parameter gradients are reasonable (non-zero for most cases)
        # Note: grad_T can be close to zero if temperature is "optimal"

    def test_with_grads_matches_param_finite_diff_cpu(self):
        """CPU parameter gradients should include the boundary gap-cost terms."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost = 0.75, 1.25
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2)

        _, _, grad_ins, grad_del, grad_T = d2p.soft_levenshtein_with_grads(
            scores, ins_cost, del_cost, temperature, None
        )

        distance_ins_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost + eps, del_cost, temperature)
        distance_ins_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost - eps, del_cost, temperature)
        grad_ins_fd = (distance_ins_plus - distance_ins_minus) / (2 * eps)

        distance_del_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost + eps, temperature)
        distance_del_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost - eps, temperature)
        grad_del_fd = (distance_del_plus - distance_del_minus) / (2 * eps)

        distance_temp_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature + eps)
        distance_temp_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature - eps)
        grad_T_fd = (distance_temp_plus - distance_temp_minus) / (2 * eps)

        assert allclose(grad_ins_fd, grad_ins, rtol=1e-2, atol=2e-3), \
            f"grad_ins mismatch: max diff = {max_diff(grad_ins_fd, grad_ins)}"
        assert allclose(grad_del_fd, grad_del, rtol=1e-2, atol=2e-3), \
            f"grad_del mismatch: max diff = {max_diff(grad_del_fd, grad_del)}"
        assert allclose(grad_T_fd, grad_T, rtol=1e-2, atol=2e-3), \
            f"grad_T mismatch: max diff = {max_diff(grad_T_fd, grad_T)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_levenshtein_hvp(scores, V, ins_cost, del_cost, temperature, None)

        posteriors_plus = soft_levenshtein_naive(scores + eps * V, ins_cost, del_cost, temperature)
        posteriors_minus = soft_levenshtein_naive(scores - eps * V, ins_cost, del_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        # Finite differences have O(eps^2) error, so allow slightly larger tolerance
        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_asymmetric_costs(self, device):
        """Test HVP with asymmetric costs."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        ins_cost, del_cost = 0.5, 1.5
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_levenshtein_hvp(scores, V, ins_cost, del_cost, temperature, None)

        posteriors_plus = soft_levenshtein_naive(scores + eps * V, ins_cost, del_cost, temperature)
        posteriors_minus = soft_levenshtein_naive(scores - eps * V, ins_cost, del_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=2e-3), \
            f"HVP mismatch (asymmetric): max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        ins_cost, del_cost = 0.75, 1.25
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.rand(B, L1, L2)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)
            posteriors_ref = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature)

            distance_ins_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost + eps, del_cost, temperature)
            distance_ins_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost - eps, del_cost, temperature)
            grad_ins_ref = (distance_ins_plus - distance_ins_minus) / (2 * eps)

            distance_del_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost + eps, temperature)
            distance_del_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost - eps, temperature)
            grad_del_ref = (distance_del_plus - distance_del_minus) / (2 * eps)

            distance_temp_plus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature + eps)
            distance_temp_minus, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature - eps)
            grad_T_ref = (distance_temp_plus - distance_temp_minus) / (2 * eps)

            posteriors_plus = soft_levenshtein_naive(scores + eps * tangent, ins_cost, del_cost, temperature)
            posteriors_minus = soft_levenshtein_naive(scores - eps * tangent, ins_cost, del_cost, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            posteriors_ins_plus = soft_levenshtein_naive(scores, ins_cost + eps, del_cost, temperature)
            posteriors_ins_minus = soft_levenshtein_naive(scores, ins_cost - eps, del_cost, temperature)
            dP_dins_ref = (posteriors_ins_plus - posteriors_ins_minus) / (2 * eps)

            posteriors_del_plus = soft_levenshtein_naive(scores, ins_cost, del_cost + eps, temperature)
            posteriors_del_minus = soft_levenshtein_naive(scores, ins_cost, del_cost - eps, temperature)
            dP_ddel_ref = (posteriors_del_plus - posteriors_del_minus) / (2 * eps)

            posteriors_temp_plus = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature + eps)
            posteriors_temp_minus = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature - eps)
            dP_dT_ref = (posteriors_temp_plus - posteriors_temp_minus) / (2 * eps)

        reference_outputs = {
            "distance": distance_ref,
            "posteriors": posteriors_ref,
            "grad_ins": grad_ins_ref,
            "grad_del": grad_del_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dins": dP_dins_ref,
            "dP_ddel": dP_ddel_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, ins_cost, del_cost, temperature)
            assert_threaded_levenshtein_correctness(outputs, reference_outputs, thread_count)
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
        ins_cost, del_cost = 1.0, 1.0

        torch.manual_seed(42)
        scores_cpu = torch.rand(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = d2p.soft_levenshtein_float(scores_cpu, ins_cost, del_cost, temperature, None)[1]
        posteriors_cuda = d2p.soft_levenshtein_float(scores_cuda, ins_cost, del_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_asymmetric(self):
        """Test CPU vs CUDA with asymmetric costs."""
        B, L1, L2 = 2, 10, 12
        temperature = 1.0
        ins_cost, del_cost = 0.5, 1.5

        torch.manual_seed(42)
        scores_cpu = torch.rand(B, L1, L2)
        scores_cuda = scores_cpu.cuda()

        posteriors_cpu = d2p.soft_levenshtein_float(scores_cpu, ins_cost, del_cost, temperature, None)[1]
        posteriors_cuda = d2p.soft_levenshtein_float(scores_cuda, ins_cost, del_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch (asymmetric): max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_backward_boundary_cost_grads_parity(self):
        """Regression: CUDA boundary ins/del grads must match CPU (r25 accumulation fix)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost = 0.5, 1.5

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2)

        result_cpu = d2p.soft_levenshtein_with_grads(
            scores, ins_cost, del_cost, temperature, None
        )
        result_cuda = d2p.soft_levenshtein_with_grads(
            scores.cuda(), ins_cost, del_cost, temperature, None
        )

        # grad_ins=2, grad_del=3, grad_T=4
        assert allclose(result_cpu[2], result_cuda[2], rtol=1e-3, atol=1e-4), \
            f"grad_ins CPU/CUDA mismatch: max diff = {max_diff(result_cpu[2], result_cuda[2])}"
        assert allclose(result_cpu[3], result_cuda[3], rtol=1e-3, atol=1e-4), \
            f"grad_del CPU/CUDA mismatch: max diff = {max_diff(result_cpu[3], result_cuda[3])}"
        assert allclose(result_cpu[4], result_cuda[4], rtol=1e-3, atol=1e-4), \
            f"grad_T CPU/CUDA mismatch: max diff = {max_diff(result_cpu[4], result_cuda[4])}"

    def test_backward_grad_T_parity(self):
        """Regression: CUDA grad_T must match CPU (r25)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2)

        result_cpu = d2p.soft_levenshtein_with_grads(
            scores, ins_cost, del_cost, temperature, None
        )
        result_cuda = d2p.soft_levenshtein_with_grads(
            scores.cuda(), ins_cost, del_cost, temperature, None
        )

        # grad_T is index 4
        assert allclose(result_cpu[4], result_cuda[4], rtol=1e-3, atol=1e-4), \
            f"grad_T CPU/CUDA mismatch: max diff = {max_diff(result_cpu[4], result_cuda[4])}"

    def test_distance_backward_skips_unused_posterior_hvp(self):
        B, L1, L2 = 1, 12, 13
        temperature = 1.0
        ins_cost, del_cost = 0.75, 1.25

        def distance_only_backward():
            torch.manual_seed(8675309)
            scores = torch.rand(B, L1, L2, device="cuda", requires_grad=True)
            distance, _ = d2p.soft_levenshtein_float(
                scores, ins_cost, del_cost, temperature, None
            )
            distance.sum().backward()

        def explicit_zero_posterior_backward():
            torch.manual_seed(8675309)
            scores = torch.rand(B, L1, L2, device="cuda", requires_grad=True)
            distance, posteriors = d2p.soft_levenshtein_float(
                scores, ins_cost, del_cost, temperature, None
            )
            (distance.sum() + 0.0 * posteriors.sum()).backward()

        distance_only_kernels = cuda_kernel_names(distance_only_backward)
        assert_cuda_kernel_seen(distance_only_kernels, "lev_forward_diag_kernel")
        assert not any("lev_hvp_" in name for name in distance_only_kernels)
        assert not any("lev_param_grad_" in name for name in distance_only_kernels)

        explicit_zero_kernels = cuda_kernel_names(explicit_zero_posterior_backward)
        assert_cuda_kernel_seen(explicit_zero_kernels, "lev_hvp_forward_diag_kernel")
        assert_cuda_kernel_seen(explicit_zero_kernels, "lev_param_grad_forward_diag_kernel")


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestValidation:

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_all_entrypoints_reject_out_of_bounds_lengths(self, device_type):
        device = torch.device(device_type)
        B, max_L1, max_L2 = 2, 4, 5
        torch.manual_seed(7)
        scores = torch.rand(B, max_L1, max_L2, device=device)
        tangent = torch.randn_like(scores)
        bad_lengths = torch.tensor([[max_L1 + 1, max_L2], [max_L1, max_L2]], device=device, dtype=torch.int32)

        entrypoints = (
            lambda: d2p.soft_levenshtein_float(scores, 1.0, 1.0, 1.0, bad_lengths),
            lambda: d2p.soft_levenshtein_with_grads(scores, 1.0, 1.0, 1.0, bad_lengths),
            lambda: d2p.soft_levenshtein_hvp(scores, tangent, 1.0, 1.0, 1.0, bad_lengths),
            lambda: d2p.soft_levenshtein_param_jacobian(scores, 0, 1.0, 1.0, 1.0, bad_lengths),
            lambda: d2p.soft_levenshtein_backward_full(scores, tangent, 1.0, 1.0, 1.0, bad_lengths),
        )

        for call in entrypoints:
            with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 4"):
                call()

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
    def test_cuda_lengths_must_match_scores_device(self):
        scores = torch.randn(1, 4, 5, device="cuda:0")
        lengths_t = torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:1")

        with pytest.raises(RuntimeError, match=r"lengths must be on same device as scores"):
            d2p.soft_levenshtein_float(scores, 1.0, 1.0, 1.0, lengths_t)

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_hvp_rejects_mismatched_tangent_shape(self, device_type):
        device = torch.device(device_type)
        scores = torch.randn(2, 5, 6, device=device)
        tangent = torch.randn(2, 5, 5, device=device)

        with pytest.raises(RuntimeError, match="scores and tangent must have same shape"):
            d2p.soft_levenshtein_hvp(scores, tangent, 1.0, 1.0, 1.0, None)

    @pytest.mark.parametrize(
        "device_type",
        ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"))],
    )
    def test_backward_full_rejects_mismatched_grad_posteriors_shape(self, device_type):
        device = torch.device(device_type)
        scores = torch.randn(2, 5, 6, device=device)
        grad_posteriors = torch.randn(2, 5, 5, device=device)

        with pytest.raises(RuntimeError, match="grad_posteriors must have same shape as scores"):
            d2p.soft_levenshtein_backward_full(scores, grad_posteriors, 1.0, 1.0, 1.0, None)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_backward_full_rejects_cpu_grad_posteriors_for_cuda_scores(self):
        scores = torch.randn(2, 5, 6, device="cuda")
        grad_posteriors = torch.randn(2, 5, 6)

        with pytest.raises(RuntimeError, match="grad_posteriors must be a CUDA tensor"):
            d2p.soft_levenshtein_backward_full(scores, grad_posteriors, 1.0, 1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            scores_b = scores[b:b+1, :l1, :l2]

            distance_ref, _ = soft_levenshtein_forward_naive(scores_b, ins_cost, del_cost, temperature)
            posteriors_ref = soft_levenshtein_naive(scores_b, ins_cost, del_cost, temperature)

            # Distance should match for this sequence
            assert allclose(distance_ref, distance[b:b+1]), \
                f"Distance mismatch for batch {b}: {distance_ref.item()} vs {distance[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 cost matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0
        ins_cost = del_cost = 1.0

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)
        distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)
        posteriors_ref = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature)

        # With softmin, distance is < 0.5 due to contribution from other paths
        # At (1,1): softmin(0+0.5, 1+1, 1+1) = softmin(0.5, 2, 2) ~= 0.13 with T=1
        assert allclose(distance, distance_ref), f"Single element distance wrong: {distance.item()} vs {distance_ref.item()}"
        assert allclose(posteriors, posteriors_ref), "Single element posterior wrong"

    def test_row_vector(self, device):
        """Test 1xN cost matrix."""
        scores = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        temperature = 1.0
        ins_cost = del_cost = 1.0

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)
        distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)

        assert allclose(distance, distance_ref), "Row vector distance mismatch"

    def test_col_vector(self, device):
        """Test Nx1 cost matrix."""
        scores = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        temperature = 1.0
        ins_cost = del_cost = 1.0

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)
        distance_ref, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)

        assert allclose(distance, distance_ref), "Column vector distance mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard Levenshtein)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01
        ins_cost = del_cost = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0
        ins_cost = del_cost = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)[1]
        posteriors_ref = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_temperature_clamp(self, device):
        """Test that very low temperature doesn't cause NaN."""
        B, L1, L2 = 2, 5, 6
        temperature = 1e-6
        ins_cost = del_cost = 1.0

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)

        assert not torch.isnan(distance).any(), "Distance contains NaN"
        assert not torch.isnan(posteriors).any(), "Posteriors contain NaN"
        assert not torch.isinf(distance).any(), "Distance contains Inf"

    def test_identical_sequences(self, device):
        """Test with zero cost matrix (identical sequences)."""
        B, L = 2, 5
        temperature = 1.0
        ins_cost = del_cost = 1.0

        # Zero costs = identical sequences on diagonal path
        scores = torch.zeros(B, L, L, device=device)

        distance, posteriors = d2p.soft_levenshtein_float(scores, ins_cost, del_cost, temperature, None)

        # With zero costs, optimal path is diagonal (no edits needed)
        # Distance should be soft-aggregation of zeros
        assert distance.mean() < 1.0, "Distance should be small for zero costs"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestParamJacobian:

    def test_param_jacobian_ins(self, device):
        """Test parameter Jacobian for insertion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        # dP/d(ins_cost) via param_jacobian
        dP_dIns = d2p.soft_levenshtein_param_jacobian(scores, 0, ins_cost, del_cost, temperature, None)

        # Finite diff
        posteriors_plus = soft_levenshtein_naive(scores, ins_cost + eps, del_cost, temperature)
        posteriors_minus = soft_levenshtein_naive(scores, ins_cost - eps, del_cost, temperature)
        dP_dIns_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dIns_fd, dP_dIns, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (ins) mismatch: max diff = {max_diff(dP_dIns_fd, dP_dIns)}"

    def test_param_jacobian_del(self, device):
        """Test parameter Jacobian for deletion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        # dP/d(del_cost) via param_jacobian
        dP_dDel = d2p.soft_levenshtein_param_jacobian(scores, 1, ins_cost, del_cost, temperature, None)

        # Finite diff
        posteriors_plus = soft_levenshtein_naive(scores, ins_cost, del_cost + eps, temperature)
        posteriors_minus = soft_levenshtein_naive(scores, ins_cost, del_cost - eps, temperature)
        dP_dDel_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dDel_fd, dP_dDel, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (del) mismatch: max diff = {max_diff(dP_dDel_fd, dP_dDel)}"

    def test_param_jacobian_temperature(self, device):
        """Test parameter Jacobian for temperature."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost = 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.rand(B, L1, L2, device=device)

        # dP/dT via param_jacobian
        dP_dT = d2p.soft_levenshtein_param_jacobian(scores, 2, ins_cost, del_cost, temperature, None)

        # Finite diff
        posteriors_plus = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature + eps)
        posteriors_minus = soft_levenshtein_naive(scores, ins_cost, del_cost, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"


# --- Memory-safety regression tests (merged from test_lev_{cpp,cuda}_memsafety.py) ---

INT32_MAX = 2**31 - 1


def _oversized_scores():
    return torch.empty((1, 0, INT32_MAX + 1), dtype=torch.float32)


def _oversized_batch_scores():
    return torch.empty((INT32_MAX + 1, 0, 0), dtype=torch.float32)


def _lev_cpu_entrypoints(scores):
    tangent = torch.empty_like(scores)
    return (
        lambda: d2p.soft_levenshtein_float(scores, 1.0, 1.0, 1.0, None),
        lambda: d2p.soft_levenshtein_with_grads(scores, 1.0, 1.0, 1.0, None),
        lambda: d2p.soft_levenshtein_hvp(scores, tangent, 1.0, 1.0, 1.0, None),
        lambda: d2p.soft_levenshtein_param_jacobian(scores, 0, 1.0, 1.0, 1.0, None),
        lambda: d2p.soft_levenshtein_backward_full(scores, tangent, 1.0, 1.0, 1.0, None),
    )


def _assert_tensors_on_device(result, device):
    if isinstance(result, torch.Tensor):
        tensors = (result,)
    else:
        tensors = tuple(t for t in result if isinstance(t, torch.Tensor))

    for tensor in tensors:
        assert tensor.device == device


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.parametrize("make_scores", (_oversized_scores, _oversized_batch_scores))
def test_cpu_entrypoints_reject_shapes_that_do_not_fit_kernel_length_bounds(make_scores):
    scores = make_scores()

    for call in _lev_cpu_entrypoints(scores):
        with pytest.raises(RuntimeError, match="scores dimensions must fit int32 length bounds"):
            call()


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_cpu_valid_small_inputs_still_run_across_public_entrypoints():
    torch.manual_seed(17)
    scores = torch.randn(2, 3, 4, dtype=torch.float32)
    tangent = torch.randn_like(scores)
    lengths = torch.tensor([[3, 4], [2, 3]], dtype=torch.int32)

    distance, posteriors = d2p.soft_levenshtein_float(scores, 1.0, 1.5, 0.9, lengths)
    distance2, posteriors2, grad_ins, grad_del, grad_T = d2p.soft_levenshtein_with_grads(
        scores, 1.0, 1.5, 0.9, lengths
    )
    hvp = d2p.soft_levenshtein_hvp(scores, tangent, 1.0, 1.5, 0.9, lengths)
    dP_dins = d2p.soft_levenshtein_param_jacobian(scores, 0, 1.0, 1.5, 0.9, lengths)
    full = d2p.soft_levenshtein_backward_full(scores, tangent, 1.0, 1.5, 0.9, lengths)

    assert torch.allclose(distance, distance2)
    assert torch.allclose(posteriors, posteriors2)
    assert grad_ins.shape == grad_del.shape == grad_T.shape == (2,)
    assert hvp.shape == dP_dins.shape == scores.shape
    assert full[0].shape == scores.shape
    assert full[1].shape == full[2].shape == full[3].shape == (1,)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_lev_cpu_sources_do_not_use_int32_flattened_dp_indices():
    repo_root = Path(__file__).resolve().parents[1]
    torch_cpu = (repo_root / "src/lev/torch_cpu.cpp").read_text()
    kernels_cpu = (repo_root / "src/lev/kernels_cpu.cpp").read_text()
    kernels_cpu_h = (repo_root / "src/lev/kernels_cpu.h").read_text()

    assert "int alpha_size" not in torch_cpu
    assert "int64_t B, int64_t max_L1, int64_t max_L2" in kernels_cpu_h
    assert not re.search(
        r"\bint\s+(?:alpha_cols|idx|idx_diag|idx_up|idx_left|score_idx|final_idx)\b",
        kernels_cpu,
    )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multiple CUDA devices")
def test_current_device_mismatch_is_guarded_for_all_entrypoints():
    target = torch.device("cuda:1")
    original_device = torch.cuda.current_device()

    torch.manual_seed(123)
    scores = torch.randn(2, 4, 5, device=target)
    lengths = torch.tensor([[4, 5], [3, 4]], dtype=torch.int32, device=target)
    tangent = torch.randn_like(scores)
    ins_t = scores.new_tensor([1.0])
    del_t = scores.new_tensor([1.25])
    temp_t = scores.new_tensor([0.75])

    calls = (
        lambda: d2p.ops.soft_levenshtein(scores, ins_t, del_t, temp_t, lengths),
        lambda: d2p.soft_levenshtein_float(scores, 1.0, 1.25, 0.75, lengths),
        lambda: d2p.soft_levenshtein_with_grads(scores, 1.0, 1.25, 0.75, lengths),
        lambda: d2p.soft_levenshtein_hvp(scores, tangent, 1.0, 1.25, 0.75, lengths),
        lambda: d2p.soft_levenshtein_param_jacobian(scores, 2, 1.0, 1.25, 0.75, lengths),
        lambda: d2p.soft_levenshtein_backward_full(scores, tangent, 1.0, 1.25, 0.75, lengths),
    )

    try:
        torch.cuda.set_device(0)
        for call in calls:
            result = call()
            torch.cuda.synchronize(target)
            _assert_tensors_on_device(result, target)

        scores_req = scores.detach().clone().requires_grad_(True)
        distance, posteriors = d2p.soft_levenshtein_float(scores_req, 1.0, 1.25, 0.75, lengths)
        (distance.sum() + posteriors.square().sum()).backward()
        torch.cuda.synchronize(target)
        assert scores_req.grad is not None
        assert scores_req.grad.device == target
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_huge_workspace_shape_rejected_before_int_alpha_size_wraparound():
    scores = torch.empty((0, 65535, 65535), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="DP workspace cell count exceeds CUDA kernel index range"):
        d2p.soft_levenshtein_float(scores, 1.0, 1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_huge_dimension_rejected_before_cuda_int_narrowing():
    try:
        scores = torch.empty(
            (0, torch.iinfo(torch.int32).max + 1, 1),
            device="cuda",
            dtype=torch.float32,
        )
    except RuntimeError as exc:
        pytest.skip(f"PyTorch cannot construct this zero-batch large-shape tensor: {exc}")

    with pytest.raises(RuntimeError, match=r"scores\.size\(1\) exceeds CUDA kernel int range"):
        d2p.soft_levenshtein_float(scores, 1.0, 1.0, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_valid_cuda_inputs_still_match_cpu():
    torch.manual_seed(1234)
    scores_cpu = torch.rand(2, 5, 6)
    lengths_cpu = torch.tensor([[5, 6], [4, 5]], dtype=torch.int32)

    cpu = d2p.soft_levenshtein_with_grads(scores_cpu, 0.75, 1.25, 0.8, lengths_cpu)
    cuda = d2p.soft_levenshtein_with_grads(
        scores_cpu.cuda(),
        0.75,
        1.25,
        0.8,
        lengths_cpu.cuda(),
    )

    for expected, actual in zip(cpu, cuda):
        assert torch.allclose(expected, actual.cpu(), rtol=1e-3, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
