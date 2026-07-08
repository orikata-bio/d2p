"""
Correctness tests for Soft Damerau (True Damerau-Levenshtein with Unrestricted Transpositions).
"""

import contextlib
from pathlib import Path

import pytest
import torch

from reference import soft_damerau_forward_naive, soft_damerau_naive

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


def run_cpu_representative_outputs(sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature):
    distance, posteriors, grad_T, grad_ins, grad_del, grad_trans = d2p.soft_damerau_with_grads(
        sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None
    )
    hvp = d2p.soft_damerau_hvp(sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature, None)
    dP_dins = d2p.soft_damerau_param_jacobian(
        sub_costs, trans_src, 0, ins_cost, del_cost, trans_cost, temperature, None
    )
    dP_ddel = d2p.soft_damerau_param_jacobian(
        sub_costs, trans_src, 1, ins_cost, del_cost, trans_cost, temperature, None
    )
    dP_dtrans = d2p.soft_damerau_param_jacobian(
        sub_costs, trans_src, 2, ins_cost, del_cost, trans_cost, temperature, None
    )
    dP_dT = d2p.soft_damerau_param_jacobian(
        sub_costs, trans_src, 3, ins_cost, del_cost, trans_cost, temperature, None
    )
    return {
        "distance": distance,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "grad_ins": grad_ins,
        "grad_del": grad_del,
        "grad_trans": grad_trans,
        "hvp": hvp,
        "dP_dins": dP_dins,
        "dP_ddel": dP_ddel,
        "dP_dtrans": dP_dtrans,
        "dP_dT": dP_dT,
    }


def assert_threaded_damerau_correctness(outputs, reference_outputs, thread_count):
    assert allclose(reference_outputs["distance"], outputs["distance"]), \
        f"{thread_count}-thread distance mismatch: max diff = {max_diff(reference_outputs['distance'], outputs['distance'])}"
    assert allclose(reference_outputs["posteriors"], outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(reference_outputs['posteriors'], outputs['posteriors'])}"
    assert allclose(reference_outputs["grad_T"], outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(reference_outputs['grad_T'], outputs['grad_T'])}"
    assert allclose(reference_outputs["grad_ins"], outputs["grad_ins"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_ins mismatch: max diff = {max_diff(reference_outputs['grad_ins'], outputs['grad_ins'])}"
    assert allclose(reference_outputs["grad_del"], outputs["grad_del"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_del mismatch: max diff = {max_diff(reference_outputs['grad_del'], outputs['grad_del'])}"
    assert allclose(reference_outputs["grad_trans"], outputs["grad_trans"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_trans mismatch: max diff = {max_diff(reference_outputs['grad_trans'], outputs['grad_trans'])}"
    assert allclose(reference_outputs["hvp"], outputs["hvp"], rtol=2e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(reference_outputs['hvp'], outputs['hvp'])}"
    assert allclose(reference_outputs["dP_dins"], outputs["dP_dins"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dins mismatch: max diff = {max_diff(reference_outputs['dP_dins'], outputs['dP_dins'])}"
    assert allclose(reference_outputs["dP_ddel"], outputs["dP_ddel"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/ddel mismatch: max diff = {max_diff(reference_outputs['dP_ddel'], outputs['dP_ddel'])}"
    assert allclose(reference_outputs["dP_dtrans"], outputs["dP_dtrans"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dtrans mismatch: max diff = {max_diff(reference_outputs['dP_dtrans'], outputs['dP_dtrans'])}"
    assert allclose(reference_outputs["dP_dT"], outputs["dP_dT"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread dP/dT mismatch: max diff = {max_diff(reference_outputs['dP_dT'], outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


def assert_padded_region_zero(name, tensor, lengths):
    tensor_cpu = tensor.cpu()
    lengths_cpu = lengths.cpu()
    for batch, (l1, l2) in enumerate(lengths_cpu.tolist()):
        if l1 < tensor_cpu.size(1):
            trailing_rows = tensor_cpu[batch, l1:, :]
            assert torch.count_nonzero(trailing_rows).item() == 0, \
                f"{name} batch {batch} wrote past L1={l1}"
        if l2 < tensor_cpu.size(2):
            trailing_cols = tensor_cpu[batch, :, l2:]
            assert torch.count_nonzero(trailing_cols).item() == 0, \
                f"{name} batch {batch} wrote past L2={l2}"


def create_trans_src(B, L1, L2, density=0.3, device='cpu'):
    """Create a random transposition source index tensor.

    Unlike OSA's trans_mask, trans_src[b, i, j, :] = (k, l) specifies the
    source indices for transposition. k=-1 means invalid transposition.

    For true Damerau, transposition at (i+1, j+1) can come from any (k, l)
    where k < i+1 and l < j+1. Here we randomly select valid sources.
    """
    trans_src = torch.full((B, L1, L2, 2), -1, dtype=torch.int32, device=device)

    for b in range(B):
        for i in range(L1):
            for j in range(L2):
                # Transposition at DP position (i+1, j+1) requires source (k, l)
                # with k >= 0, l >= 0, k < i+1, l < j+1
                # For simplicity, we allow source from (i-1, j-1) like OSA
                # but with variable distances
                if i >= 1 and j >= 1 and torch.rand(1).item() < density:
                    # Random valid source: k in [0, i), l in [0, j)
                    # For testing, use adjacent (k=i-1, l=j-1) most often
                    # but occasionally use more distant sources
                    if torch.rand(1).item() < 0.7:
                        # Adjacent (like OSA)
                        trans_src[b, i, j, 0] = i - 1
                        trans_src[b, i, j, 1] = j - 1
                    else:
                        # Random valid source
                        k = torch.randint(0, i, (1,)).item()
                        l = torch.randint(0, j, (1,)).item()
                        trans_src[b, i, j, 0] = k
                        trans_src[b, i, j, 1] = l

    return trans_src


def create_adjacent_trans_src(B, L1, L2, density=0.3, device='cpu'):
    """Create trans_src with only adjacent transpositions (like OSA).

    This helps verify that Damerau reduces to OSA for adjacent transpositions.
    """
    trans_src = torch.full((B, L1, L2, 2), -1, dtype=torch.int32, device=device)

    for b in range(B):
        for i in range(1, L1):  # Need i >= 1 for adjacent
            for j in range(1, L2):  # Need j >= 1 for adjacent
                if torch.rand(1).item() < density:
                    trans_src[b, i, j, 0] = i - 1
                    trans_src[b, i, j, 1] = j - 1

    return trans_src


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(8, 10), (16, 16), (5, 20)])
def seq_lengths(request):
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture(params=[(1.0, 1.0, 1.5), (0.5, 1.5, 1.0), (2.0, 0.5, 0.8)])
def cost_params(request):
    """Different (ins_cost, del_cost, trans_cost) combinations."""
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_distance(self, batch_size, seq_lengths, temperature, cost_params, device):
        """Test that Damerau distances match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost, trans_cost = cost_params

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device) * 2
        trans_src = create_trans_src(batch_size, L1, L2, device=device)

        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        distance_d2p = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch: max diff = {max_diff(distance_ref, distance_d2p)}"

    def test_distance_no_transpositions(self, batch_size, temperature, device):
        """Test Damerau reduces to Levenshtein when trans_src has all -1."""
        L1, L2 = 10, 12
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device)
        trans_src = torch.full((batch_size, L1, L2, 2), -1, dtype=torch.int32, device=device)

        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        distance_d2p = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch (no trans): max diff = {max_diff(distance_ref, distance_d2p)}"

    def test_distance_adjacent_transpositions(self, batch_size, temperature, device):
        """Test Damerau with adjacent-only transpositions (should behave like OSA)."""
        L1, L2 = 8, 10
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device)
        trans_src = create_adjacent_trans_src(batch_size, L1, L2, density=0.4, device=device)

        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        distance_d2p = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch (adjacent): max diff = {max_diff(distance_ref, distance_d2p)}"

    def test_transposition_with_gaps(self, device):
        """Test transposition with intermediate characters (gaps)."""
        B = 2
        L = 6
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5

        torch.manual_seed(42)
        sub_costs = torch.ones(B, L, L, device=device)

        # Create trans_src with varying source distances
        trans_src = torch.full((B, L, L, 2), -1, dtype=torch.int32, device=device)
        # Position (4, 4) can transpose from (1, 1) - gap of 2 each
        trans_src[:, 3, 3, 0] = 1
        trans_src[:, 3, 3, 1] = 1
        # Position (5, 5) can transpose from (0, 0) - gap of 4 each
        trans_src[:, 4, 4, 0] = 0
        trans_src[:, 4, 4, 1] = 0

        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        distance_d2p = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[0]

        assert allclose(distance_ref, distance_d2p), \
            f"Distance mismatch (gaps): max diff = {max_diff(distance_ref, distance_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, cost_params, device):
        """Test that alignment posteriors match."""
        L1, L2 = seq_lengths
        ins_cost, del_cost, trans_cost = cost_params

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device) * 2
        trans_src = create_trans_src(batch_size, L1, L2, device=device)

        posteriors_ref = soft_damerau_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        posteriors_d2p = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, temperature, device):
        """Test gradients through the soft alignment."""
        L1, L2 = 6, 8
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(batch_size, L1, L2, device=device)
        trans_src = create_trans_src(batch_size, L1, L2, device=device)
        sub_costs.requires_grad_(True)

        posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[1]
        loss = (posteriors ** 2).sum()
        loss.backward()
        grad_d2p = sub_costs.grad.clone()

        sub_costs_ref = sub_costs.detach().clone().requires_grad_(True)
        posteriors_ref = soft_damerau_naive(sub_costs_ref, trans_src, ins_cost, del_cost, trans_cost, temperature)
        loss_ref = (posteriors_ref ** 2).sum()
        loss_ref.backward()
        grad_ref = sub_costs_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_returns_param_grads(self, device):
        """Test that soft_damerau_with_grads returns parameter gradients."""
        B, L1, L2 = 2, 6, 8
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 0.8, 1.2

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        distance, posteriors, grad_T, grad_ins, grad_del, grad_trans = d2p.soft_damerau_with_grads(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None
        )

        # Check shapes
        assert distance.shape == (B,)
        assert posteriors.shape == (B, L1, L2)
        assert grad_T.shape == (B,)
        assert grad_ins.shape == (B,)
        assert grad_del.shape == (B,)
        assert grad_trans.shape == (B,)

    def test_parameter_grads_match_finite_diff_cpu(self):
        """CPU parameter gradients should match finite differences."""
        B, L1, L2 = 3, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.8, 1.1, 0.6
        eps = 1e-4

        torch.manual_seed(123)
        sub_costs = torch.rand(B, L1, L2)
        trans_src = create_trans_src(B, L1, L2, density=0.5)

        with torch_num_threads(1):
            _, _, grad_T, grad_ins, grad_del, grad_trans = d2p.soft_damerau_with_grads(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None
            )

        distance_ins_plus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost + eps, del_cost, trans_cost, temperature
        )
        distance_ins_minus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost - eps, del_cost, trans_cost, temperature
        )
        grad_ins_ref = (distance_ins_plus - distance_ins_minus) / (2 * eps)

        distance_del_plus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost + eps, trans_cost, temperature
        )
        distance_del_minus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost - eps, trans_cost, temperature
        )
        grad_del_ref = (distance_del_plus - distance_del_minus) / (2 * eps)

        distance_trans_plus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost + eps, temperature
        )
        distance_trans_minus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost - eps, temperature
        )
        grad_trans_ref = (distance_trans_plus - distance_trans_minus) / (2 * eps)

        distance_plus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature + eps
        )
        distance_minus, _ = soft_damerau_forward_naive(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature - eps
        )
        grad_T_ref = (distance_plus - distance_minus) / (2 * eps)

        assert allclose(grad_ins_ref, grad_ins, rtol=1e-2, atol=2e-3), \
            f"Insertion gradient mismatch: max diff = {max_diff(grad_ins_ref, grad_ins)}"
        assert allclose(grad_del_ref, grad_del, rtol=1e-2, atol=2e-3), \
            f"Deletion gradient mismatch: max diff = {max_diff(grad_del_ref, grad_del)}"
        assert allclose(grad_trans_ref, grad_trans, rtol=1e-2, atol=2e-3), \
            f"Transposition gradient mismatch: max diff = {max_diff(grad_trans_ref, grad_trans)}"
        assert allclose(grad_T_ref, grad_T, rtol=1e-2, atol=2e-3), \
            f"Temperature gradient mismatch: max diff = {max_diff(grad_T_ref, grad_T)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_damerau_hvp(sub_costs, trans_src, V, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs + eps * V, trans_src, ins_cost, del_cost, trans_cost, temperature)
        posteriors_minus = soft_damerau_naive(sub_costs - eps * V, trans_src, ins_cost, del_cost, trans_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=2e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_no_transpositions(self, device):
        """Test HVP with no transpositions."""
        B, L1, L2 = 2, 6, 7
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.0
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = torch.full((B, L1, L2, 2), -1, dtype=torch.int32, device=device)
        V = torch.randn(B, L1, L2, device=device)

        hvp_d2p = d2p.soft_damerau_hvp(sub_costs, trans_src, V, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs + eps * V, trans_src, ins_cost, del_cost, trans_cost, temperature)
        posteriors_minus = soft_damerau_naive(sub_costs - eps * V, trans_src, ins_cost, del_cost, trans_cost, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=2e-2, atol=5e-3), \
            f"HVP mismatch (no trans): max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, L1, L2 = 8, 6, 7
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.8, 1.1, 0.6
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        sub_costs = torch.rand(B, L1, L2)
        trans_src = create_trans_src(B, L1, L2, density=0.5)
        tangent = torch.randn(B, L1, L2)

        with torch_num_threads(1):
            distance_ref, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature
            )
            posteriors_ref = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature
            )

            distance_temp_plus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature + eps
            )
            distance_temp_minus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature - eps
            )
            grad_T_ref = (distance_temp_plus - distance_temp_minus) / (2 * eps)

            distance_ins_plus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost + eps, del_cost, trans_cost, temperature
            )
            distance_ins_minus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost - eps, del_cost, trans_cost, temperature
            )
            grad_ins_ref = (distance_ins_plus - distance_ins_minus) / (2 * eps)

            distance_del_plus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost + eps, trans_cost, temperature
            )
            distance_del_minus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost - eps, trans_cost, temperature
            )
            grad_del_ref = (distance_del_plus - distance_del_minus) / (2 * eps)

            distance_trans_plus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost + eps, temperature
            )
            distance_trans_minus, _ = soft_damerau_forward_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost - eps, temperature
            )
            grad_trans_ref = (distance_trans_plus - distance_trans_minus) / (2 * eps)

            posteriors_plus = soft_damerau_naive(
                sub_costs + eps * tangent, trans_src, ins_cost, del_cost, trans_cost, temperature
            )
            posteriors_minus = soft_damerau_naive(
                sub_costs - eps * tangent, trans_src, ins_cost, del_cost, trans_cost, temperature
            )
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            dP_dins_plus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost + eps, del_cost, trans_cost, temperature
            )
            dP_dins_minus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost - eps, del_cost, trans_cost, temperature
            )
            dP_dins_ref = (dP_dins_plus - dP_dins_minus) / (2 * eps)

            dP_ddel_plus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost + eps, trans_cost, temperature
            )
            dP_ddel_minus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost - eps, trans_cost, temperature
            )
            dP_ddel_ref = (dP_ddel_plus - dP_ddel_minus) / (2 * eps)

            dP_dtrans_plus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost + eps, temperature
            )
            dP_dtrans_minus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost - eps, temperature
            )
            dP_dtrans_ref = (dP_dtrans_plus - dP_dtrans_minus) / (2 * eps)

            dP_dT_plus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature + eps
            )
            dP_dT_minus = soft_damerau_naive(
                sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature - eps
            )
            dP_dT_ref = (dP_dT_plus - dP_dT_minus) / (2 * eps)

        reference_outputs = {
            "distance": distance_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "grad_ins": grad_ins_ref,
            "grad_del": grad_del_ref,
            "grad_trans": grad_trans_ref,
            "hvp": hvp_ref,
            "dP_dins": dP_dins_ref,
            "dP_ddel": dP_ddel_ref,
            "dP_dtrans": dP_dtrans_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(
                    sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature
                )
            assert_threaded_damerau_correctness(outputs, reference_outputs, thread_count)
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
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs_cpu = torch.rand(B, L1, L2)
        trans_src_cpu = create_trans_src(B, L1, L2)
        sub_costs_cuda = sub_costs_cpu.cuda()
        trans_src_cuda = trans_src_cpu.cuda()

        posteriors_cpu = d2p.soft_damerau_float(sub_costs_cpu, trans_src_cpu, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_cuda = d2p.soft_damerau_float(sub_costs_cuda, trans_src_cuda, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_with_gaps(self):
        """Test CPU vs CUDA with transpositions that have gaps."""
        B, L1, L2 = 2, 10, 12
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.5, 1.5, 0.8

        torch.manual_seed(42)
        sub_costs_cpu = torch.rand(B, L1, L2)
        trans_src_cpu = create_trans_src(B, L1, L2, density=0.5)
        sub_costs_cuda = sub_costs_cpu.cuda()
        trans_src_cuda = trans_src_cpu.cuda()

        posteriors_cpu = d2p.soft_damerau_float(sub_costs_cpu, trans_src_cpu, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_cuda = d2p.soft_damerau_float(sub_costs_cuda, trans_src_cuda, ins_cost, del_cost, trans_cost, temperature, None)[1]

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch (gaps): max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_backward_grad_T_parity(self):
        """Regression: CUDA grad_T must match CPU (r25 sign fix)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2)
        trans_src = create_trans_src(B, L1, L2)

        result_cpu = d2p.soft_damerau_with_grads(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None
        )
        result_cuda = d2p.soft_damerau_with_grads(
            sub_costs.cuda(), trans_src.cuda(), ins_cost, del_cost, trans_cost, temperature, None
        )

        # grad_T is index 2
        assert allclose(result_cpu[2], result_cuda[2], rtol=1e-3, atol=1e-4), \
            f"grad_T CPU/CUDA mismatch: max diff = {max_diff(result_cpu[2], result_cuda[2])}"

    def test_backward_boundary_cost_grads_parity(self):
        """Regression: CUDA boundary ins/del grads must match CPU (r25 accumulation fix)."""
        B, L1, L2 = 4, 8, 10
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 0.5, 1.5, 0.8

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2)
        trans_src = create_trans_src(B, L1, L2, density=0.5)

        result_cpu = d2p.soft_damerau_with_grads(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None
        )
        result_cuda = d2p.soft_damerau_with_grads(
            sub_costs.cuda(), trans_src.cuda(), ins_cost, del_cost, trans_cost, temperature, None
        )

        # grad_ins=3, grad_del=4, grad_trans=5
        assert allclose(result_cpu[3], result_cuda[3], rtol=1e-3, atol=1e-4), \
            f"grad_ins CPU/CUDA mismatch: max diff = {max_diff(result_cpu[3], result_cuda[3])}"
        assert allclose(result_cpu[4], result_cuda[4], rtol=1e-3, atol=1e-4), \
            f"grad_del CPU/CUDA mismatch: max diff = {max_diff(result_cpu[4], result_cuda[4])}"
        assert allclose(result_cpu[5], result_cuda[5], rtol=1e-3, atol=1e-4), \
            f"grad_trans CPU/CUDA mismatch: max diff = {max_diff(result_cpu[5], result_cuda[5])}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_L1, max_L2 = 10, 12
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 1.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, max_L1, max_L2, device=device)
        trans_src = create_trans_src(B, max_L1, max_L2, device=device)

        # Variable lengths
        lengths = torch.tensor([
            [8, 10],
            [10, 12],
            [6, 8],
            [9, 11]
        ], device=device, dtype=torch.int32)

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            l1, l2 = lengths[b].tolist()
            sub_costs_b = sub_costs[b:b+1, :l1, :l2]
            trans_src_b = trans_src[b:b+1, :l1, :l2, :]

            distance_ref, _ = soft_damerau_forward_naive(sub_costs_b, trans_src_b, ins_cost, del_cost, trans_cost, temperature)
            posteriors_ref = soft_damerau_naive(sub_costs_b, trans_src_b, ins_cost, del_cost, trans_cost, temperature)

            # Distance should match for this sequence
            assert allclose(distance_ref, distance[b:b+1]), \
                f"Distance mismatch for batch {b}: {distance_ref.item()} vs {distance[b].item()}"

            # Posteriors for valid region should match
            assert allclose(posteriors_ref, posteriors[b:b+1, :l1, :l2], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"

    def test_padded_regions_stay_zero(self, device):
        """Variable-length batches must leave padded regions untouched."""
        B = 3
        max_L1, max_L2 = 6, 7
        temperature = 0.9
        ins_cost, del_cost, trans_cost = 0.5, 1.4, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2, device=device)
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.55, device=device)
        tangent = torch.randn(B, max_L1, max_L2, device=device)
        lengths = torch.tensor([[6, 7], [4, 6], [5, 3]], dtype=torch.int32, device=device)

        _, forward_posteriors = d2p.soft_damerau_float(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
        )
        assert_padded_region_zero("forward_posteriors", forward_posteriors, lengths)

        with_grads = d2p.soft_damerau_with_grads(
            sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, lengths
        )
        assert_padded_region_zero("with_grads_posteriors", with_grads[1], lengths)

        hvp = d2p.soft_damerau_hvp(
            sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
        )
        assert_padded_region_zero("hvp", hvp, lengths)

        for param_type, name in (
            (0, "param_ins"),
            (1, "param_del"),
            (2, "param_trans"),
            (3, "param_temperature"),
        ):
            param = d2p.soft_damerau_param_jacobian(
                sub_costs, trans_src, param_type, ins_cost, del_cost, trans_cost, temperature, lengths
            )
            assert_padded_region_zero(name, param, lengths)

        backward_full = d2p.soft_damerau_backward_full(
            sub_costs, trans_src, tangent, ins_cost, del_cost, trans_cost, temperature, lengths
        )
        assert_padded_region_zero("backward_full_grad_scores", backward_full[0], lengths)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestValidation:

    def test_invalid_lengths_rejected_cpu(self):
        B, max_L1, max_L2 = 3, 6, 7
        temperature = 0.9
        ins_cost, del_cost, trans_cost = 0.5, 1.4, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2)
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.55)
        bad_lengths = torch.tensor([[max_L1 + 1, max_L2], [max_L1, max_L2], [max_L1, max_L2]], dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 6"):
            d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, bad_lengths)

    def test_invalid_trans_src_shape_rejected_cpu(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2)
        bad_trans_src = torch.full((B, max_L1, max_L2, 3), -1, dtype=torch.int32)

        with pytest.raises(RuntimeError, match="trans_src must have shape"):
            d2p.soft_damerau_float(sub_costs, bad_trans_src, ins_cost, del_cost, trans_cost, temperature, None)

    def test_hvp_requires_matching_shape_cpu(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2)
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.4)
        bad_tangent = torch.randn(B, max_L1 - 1, max_L2)

        with pytest.raises(RuntimeError, match="sub_costs and tangent must have same shape"):
            d2p.soft_damerau_hvp(sub_costs, trans_src, bad_tangent, ins_cost, del_cost, trans_cost, temperature, None)

    def test_backward_full_requires_matching_shape_cpu(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2)
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.4)
        bad_grad = torch.randn(B, max_L1 - 1, max_L2)

        with pytest.raises(RuntimeError, match="grad_posteriors must have same shape as sub_costs"):
            d2p.soft_damerau_backward_full(
                sub_costs, trans_src, bad_grad, ins_cost, del_cost, trans_cost, temperature, None
            )

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_invalid_lengths_rejected_cuda(self):
        B, max_L1, max_L2 = 3, 6, 7
        temperature = 0.9
        ins_cost, del_cost, trans_cost = 0.5, 1.4, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2).cuda()
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.55).cuda()
        bad_lengths = torch.tensor([[max_L1 + 1, max_L2], [max_L1, max_L2], [max_L1, max_L2]], dtype=torch.int32).cuda()

        with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 0 and 6"):
            d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, bad_lengths)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_invalid_trans_src_shape_rejected_cuda(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2).cuda()
        bad_trans_src = torch.full((B, max_L1, max_L2, 3), -1, dtype=torch.int32).cuda()

        with pytest.raises(RuntimeError, match="trans_src must have shape"):
            d2p.soft_damerau_float(sub_costs, bad_trans_src, ins_cost, del_cost, trans_cost, temperature, None)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_hvp_requires_matching_shape_cuda(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2).cuda()
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.4).cuda()
        bad_tangent = torch.randn(B, max_L1 - 1, max_L2).cuda()

        with pytest.raises(RuntimeError, match="sub_costs and tangent must have same shape"):
            d2p.soft_damerau_hvp(sub_costs, trans_src, bad_tangent, ins_cost, del_cost, trans_cost, temperature, None)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_backward_full_requires_matching_shape_cuda(self):
        B, max_L1, max_L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.8

        torch.manual_seed(123)
        sub_costs = torch.rand(B, max_L1, max_L2).cuda()
        trans_src = create_trans_src(B, max_L1, max_L2, density=0.4).cuda()
        bad_grad = torch.randn(B, max_L1 - 1, max_L2).cuda()

        with pytest.raises(RuntimeError, match="grad_posteriors must have same shape as sub_costs"):
            d2p.soft_damerau_backward_full(
                sub_costs, trans_src, bad_grad, ins_cost, del_cost, trans_cost, temperature, None
            )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 cost matrix."""
        sub_costs = torch.tensor([[[0.5]]], device=device)
        trans_src = torch.full((1, 1, 1, 2), -1, dtype=torch.int32, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)
        posteriors_ref = soft_damerau_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), f"Single element distance wrong: {distance.item()} vs {distance_ref.item()}"
        assert allclose(posteriors, posteriors_ref), "Single element posterior wrong"

    def test_2x2_with_adjacent_transposition(self, device):
        """Test 2x2 cost matrix with adjacent transposition."""
        sub_costs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device)
        trans_src = torch.full((1, 2, 2, 2), -1, dtype=torch.int32, device=device)
        trans_src[0, 1, 1, 0] = 0  # k = 0
        trans_src[0, 1, 1, 1] = 0  # l = 0
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), f"2x2 trans distance mismatch: {max_diff(distance, distance_ref)}"

    def test_row_vector(self, device):
        """Test 1xN cost matrix."""
        sub_costs = torch.tensor([[[0.1, 0.2, 0.3]]], device=device)
        trans_src = torch.full((1, 1, 3, 2), -1, dtype=torch.int32, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), "Row vector distance mismatch"

    def test_col_vector(self, device):
        """Test Nx1 cost matrix."""
        sub_costs = torch.tensor([[[0.1], [0.2], [0.3]]], device=device)
        trans_src = torch.full((1, 3, 1, 2), -1, dtype=torch.int32, device=device)
        temperature = 1.0
        ins_cost = del_cost = trans_cost = 1.0

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), "Column vector distance mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard Damerau)."""
        B, L1, L2 = 2, 6, 8
        temperature = 0.01
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[1]

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, L1, L2 = 2, 6, 8
        temperature = 10.0
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)[1]
        posteriors_ref = soft_damerau_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_zero_temperature_clamp(self, device):
        """Test that very low temperature doesn't cause NaN."""
        B, L1, L2 = 2, 5, 6
        temperature = 1e-6
        ins_cost = del_cost = trans_cost = 1.0

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)

        assert not torch.isnan(distance).any(), "Distance contains NaN"
        assert not torch.isnan(posteriors).any(), "Posteriors contain NaN"
        assert not torch.isinf(distance).any(), "Distance contains Inf"

    def test_distant_transposition(self, device):
        """Test transposition from distant source (gap of 3)."""
        B, L = 2, 8
        temperature = 1.0
        ins_cost = del_cost = 1.0
        trans_cost = 0.5

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L, L, device=device)
        trans_src = torch.full((B, L, L, 2), -1, dtype=torch.int32, device=device)
        # Position (6, 6) can transpose from (2, 2) - gap of 3 each
        trans_src[:, 5, 5, 0] = 2
        trans_src[:, 5, 5, 1] = 2

        distance, posteriors = d2p.soft_damerau_float(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature, None)
        distance_ref, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

        assert allclose(distance, distance_ref), \
            f"Distant trans distance mismatch: {max_diff(distance, distance_ref)}"


devices = ['cpu']
if CUDA_AVAILABLE:
    devices.append('cuda')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not installed")
@pytest.mark.parametrize("device", devices)
class TestParamJacobian:

    def test_param_jacobian_ins(self, device):
        """Test parameter Jacobian for insertion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        dP_dIns = d2p.soft_damerau_param_jacobian(sub_costs, trans_src, 0, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost + eps, del_cost, trans_cost, temperature)
        posteriors_minus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost - eps, del_cost, trans_cost, temperature)
        dP_dIns_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dIns_fd, dP_dIns, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (ins) mismatch: max diff = {max_diff(dP_dIns_fd, dP_dIns)}"

    def test_param_jacobian_del(self, device):
        """Test parameter Jacobian for deletion cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        dP_dDel = d2p.soft_damerau_param_jacobian(sub_costs, trans_src, 1, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost + eps, trans_cost, temperature)
        posteriors_minus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost - eps, trans_cost, temperature)
        dP_dDel_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dDel_fd, dP_dDel, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (del) mismatch: max diff = {max_diff(dP_dDel_fd, dP_dDel)}"

    def test_param_jacobian_trans(self, device):
        """Test parameter Jacobian for transposition cost."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        dP_dTrans = d2p.soft_damerau_param_jacobian(sub_costs, trans_src, 2, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost, trans_cost + eps, temperature)
        posteriors_minus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost, trans_cost - eps, temperature)
        dP_dTrans_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dTrans_fd, dP_dTrans, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (trans) mismatch: max diff = {max_diff(dP_dTrans_fd, dP_dTrans)}"

    def test_param_jacobian_temperature(self, device):
        """Test parameter Jacobian for temperature."""
        B, L1, L2 = 2, 5, 6
        temperature = 1.0
        ins_cost, del_cost, trans_cost = 1.0, 1.0, 0.7
        eps = 1e-4

        torch.manual_seed(42)
        sub_costs = torch.rand(B, L1, L2, device=device)
        trans_src = create_trans_src(B, L1, L2, device=device)

        dP_dT = d2p.soft_damerau_param_jacobian(sub_costs, trans_src, 3, ins_cost, del_cost, trans_cost, temperature, None)

        posteriors_plus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost, trans_cost, temperature + eps)
        posteriors_minus = soft_damerau_naive(sub_costs.cpu(), trans_src.cpu(), ins_cost, del_cost, trans_cost, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=1e-2, atol=2e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"


# --- Memory-safety regression tests (merged from test_damerau_{cpp,cuda}_memsafety.py) ---

INS_COST = 0.7
DEL_COST = 1.2
TRANS_COST = 0.9
TEMPERATURE = 0.8


def huge_empty_inputs(l1, l2):
    sub_costs = torch.empty((0, l1, l2), dtype=torch.float32)
    trans_src = torch.empty((0, l1, l2, 2), dtype=torch.int32)
    return sub_costs, trans_src


def assert_all_cpu_entrypoints_reject(sub_costs, trans_src, match):
    scalar = torch.tensor([1.0], dtype=torch.float32)
    lengths = torch.empty((0, 2), dtype=torch.int32)
    tangent = torch.empty_like(sub_costs)

    calls = [
        lambda: torch.ops.d2p.soft_damerau(
            sub_costs, trans_src, scalar, scalar, scalar, scalar, lengths
        ),
        lambda: d2p.soft_damerau_float(
            sub_costs, trans_src, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        ),
        lambda: d2p.soft_damerau_with_grads(
            sub_costs, trans_src, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        ),
        lambda: d2p.soft_damerau_hvp(
            sub_costs, trans_src, tangent, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        ),
        lambda: d2p.soft_damerau_param_jacobian(
            sub_costs, trans_src, 0, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        ),
        lambda: d2p.soft_damerau_backward_full(
            sub_costs, trans_src, tangent, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
        ),
    ]

    for call in calls:
        with pytest.raises(RuntimeError, match=match):
            call()


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_cpu_rejects_trans_src_pair_index_overflow_shape():
    sub_costs, trans_src = huge_empty_inputs(32769, 32769)

    assert_all_cpu_entrypoints_reject(
        sub_costs,
        trans_src,
        r"Damerau CPU trans_src grid is too large",
    )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_cpu_rejects_dp_workspace_index_overflow_shape():
    sub_costs, trans_src = huge_empty_inputs(1, 1_073_741_824)

    assert_all_cpu_entrypoints_reject(
        sub_costs,
        trans_src,
        r"Damerau CPU DP workspace is too large",
    )


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_cpu_valid_small_input_matches_reference():
    sub_costs = torch.tensor(
        [[[0.2, 0.7, 1.0], [0.4, 0.1, 0.8], [1.2, 0.3, 0.5]]],
        dtype=torch.float32,
    )
    trans_src = torch.full((1, 3, 3, 2), -1, dtype=torch.int32)
    trans_src[0, 1, 1] = torch.tensor([0, 0], dtype=torch.int32)
    trans_src[0, 2, 2] = torch.tensor([1, 1], dtype=torch.int32)

    score, posteriors = d2p.soft_damerau_float(
        sub_costs, trans_src, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE, None
    )
    expected_score, _ = soft_damerau_forward_naive(
        sub_costs, trans_src, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE
    )
    expected_posteriors = soft_damerau_naive(
        sub_costs, trans_src, INS_COST, DEL_COST, TRANS_COST, TEMPERATURE
    )

    assert torch.allclose(score, expected_score, rtol=1e-4, atol=1e-5)
    assert torch.allclose(posteriors, expected_posteriors, rtol=1e-4, atol=1e-5)


def _zero_stride_cuda(shape, *, dtype):
    base = torch.empty(1, dtype=dtype, device="cuda")
    return base.as_strided(shape, (0,) * len(shape))


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="Damerau CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_trans_src_pair_index_overflow_rejected_before_kernel_launch():
    sub_costs = _zero_stride_cuda((1, 32769, 32769), dtype=torch.float32)
    trans_src = _zero_stride_cuda((1, 32769, 32769, 2), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="too large for 32-bit trans_src indexing"):
        d2p.soft_damerau_float(sub_costs, trans_src, 1.0, 1.0, 1.0, 1.0, None)


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="Damerau CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_alpha_index_overflow_rejected_before_workspace_allocation():
    sub_costs = _zero_stride_cuda((1, 0, 2_147_483_647), dtype=torch.float32)
    trans_src = _zero_stride_cuda((1, 0, 2_147_483_647, 2), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="too large for 32-bit kernel indexing"):
        d2p.soft_damerau_float(sub_costs, trans_src, 1.0, 1.0, 1.0, 1.0, None)


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="Damerau CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_cuda_autograd_keeps_posterior_grad_unmaterialized():
    source = Path(__file__).resolve().parents[1] / "src" / "damerau" / "torch_cuda.cpp"
    text = source.read_text()

    assert "ctx->set_materialize_grads(false)" in text
    assert "if (grad_posteriors.defined() && grad_posteriors.numel() > 0)" in text


@pytest.mark.skipif(
    not (D2P_AVAILABLE and CUDA_AVAILABLE),
    reason="Damerau CUDA memory-safety regressions need the CUDA d2p backend",
)
def test_score_only_backward_still_matches_posteriors():
    sub_costs = torch.tensor(
        [[[0.2, 1.1, 0.7], [1.0, 0.3, 0.8], [0.9, 0.6, 0.4]]],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    trans_src = torch.full((1, 3, 3, 2), -1, dtype=torch.int32, device="cuda")
    trans_src[0, 2, 2] = torch.tensor([0, 0], dtype=torch.int32, device="cuda")

    score, posteriors = d2p.soft_damerau_float(
        sub_costs, trans_src, 0.7, 1.2, 0.9, 1.1, None
    )
    score.sum().backward()

    assert torch.allclose(sub_costs.grad, posteriors, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
