"""
Correctness tests for Soft MAS (Monotonic Alignment Search).
"""

import contextlib

import pytest
import torch

from reference import soft_mas_forward_naive, soft_mas_naive

try:
    import d2p
    D2P_AVAILABLE = True
except ImportError:
    D2P_AVAILABLE = False

CUDA_AVAILABLE = torch.cuda.is_available()

PARAM_JACOBIAN_FIRST_COLUMN_REGRESSION_SCORES = torch.tensor(
    [[
        [0.6390, 0.7386, 0.3336, 0.6488, -0.1221, -1.4663],
        [1.8326, -0.1780, -1.0180, 0.5100, -2.5325, 0.0219],
        [1.5904, 0.0740, -0.6459, -0.9266, 0.7246, 1.2455],
        [0.3673, 0.2022, -1.3746, 0.0133, -0.3854, -0.2535],
        [-1.4655, 0.5268, 1.4230, 0.7589, 0.1948, -0.6701],
        [-0.0178, 0.3026, 0.0068, -0.3653, 2.6368, 0.0119],
        [0.0449, 0.1808, 0.9646, 0.6207, -0.1754, -0.2981],
        [-1.1332, 0.2050, -0.2275, -1.3782, -1.5521, 1.0560],
        [-1.5421, -1.1325, 0.3822, 0.7824, 0.3121, -1.1421],
        [0.6226, -2.3291, 1.1045, 1.4058, 0.4952, -1.0435],
        [1.3387, -0.9426, -0.7033, 1.3839, -0.6329, -0.0738],
        [0.4396, 0.0949, 1.0510, -1.5036, -0.4133, -1.7405],
    ]],
    dtype=torch.float32,
)


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


def run_cpu_representative_outputs(scores, tangent, temperature):
    partition, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)
    grad_T = d2p.soft_mas_backward_full(scores, temperature, None)[2]
    hvp = d2p.soft_mas_hvp(scores, tangent, temperature, None)
    dP_dT = d2p.soft_mas_param_jacobian(scores, temperature, None)
    return {
        "partition": partition,
        "posteriors": posteriors,
        "grad_T": grad_T,
        "hvp": hvp,
        "dP_dT": dP_dT,
    }


def assert_threaded_mas_correctness(outputs, reference_outputs, thread_count):
    partition_ref = reference_outputs["partition"]
    posteriors_ref = reference_outputs["posteriors"]
    grad_T_ref = reference_outputs["grad_T"]
    hvp_ref = reference_outputs["hvp"]
    dP_dT_ref = reference_outputs["dP_dT"]

    assert allclose(partition_ref, outputs["partition"]), \
        f"{thread_count}-thread partition mismatch: max diff = {max_diff(partition_ref, outputs['partition'])}"
    assert allclose(posteriors_ref, outputs["posteriors"], rtol=1e-3, atol=1e-4), \
        f"{thread_count}-thread posterior mismatch: max diff = {max_diff(posteriors_ref, outputs['posteriors'])}"
    assert allclose(grad_T_ref, outputs["grad_T"], rtol=1e-2, atol=2e-3), \
        f"{thread_count}-thread grad_T mismatch: max diff = {max_diff(grad_T_ref, outputs['grad_T'])}"
    assert allclose(hvp_ref, outputs["hvp"], rtol=1e-2, atol=5e-3), \
        f"{thread_count}-thread HVP mismatch: max diff = {max_diff(hvp_ref, outputs['hvp'])}"
    assert allclose(dP_dT_ref, outputs["dP_dT"], rtol=2e-2, atol=5e-3), \
        f"{thread_count}-thread dP_dT mismatch: max diff = {max_diff(dP_dT_ref, outputs['dP_dT'])}"


def assert_exact_thread_match(reference_outputs, outputs, thread_count):
    for name, reference in reference_outputs.items():
        actual = outputs[name]
        assert torch.equal(reference, actual), \
            f"{name} changed between 1 and {thread_count} threads: max diff = {max_diff(reference, actual)}"


@pytest.fixture(params=[1, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[(10, 8), (16, 16), (20, 5)])
def seq_lengths(request):
    """(T, S) where T=frames, S=text length. Must have T >= S."""
    return request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def temperature(request):
    return request.param


@pytest.fixture
def device():
    return torch.device('cuda' if CUDA_AVAILABLE else 'cpu')


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestForward:

    def test_score(self, batch_size, seq_lengths, temperature, device):
        """Test that MAS scores match."""
        T, S = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)

        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        score_d2p, _ = d2p.soft_mas_with_grads(scores, temperature, None)

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch: max diff = {max_diff(score_ref, score_d2p)}"

    def test_score_positive_scores(self, batch_size, device):
        """Test MAS with positive scores."""
        T, S = 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device).abs()

        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        score_d2p, _ = d2p.soft_mas_with_grads(scores, temperature, None)

        assert allclose(score_ref, score_d2p), \
            f"Score mismatch with positive scores: max diff = {max_diff(score_ref, score_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestBackward:

    def test_posteriors(self, batch_size, seq_lengths, temperature, device):
        """Test that alignment posteriors match."""
        T, S = seq_lengths

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)

        posteriors_ref = soft_mas_naive(scores, temperature)
        _, posteriors_d2p = d2p.soft_mas_with_grads(scores, temperature, None)

        assert allclose(posteriors_ref, posteriors_d2p, rtol=1e-3, atol=1e-4), \
            f"Posteriors mismatch: max diff = {max_diff(posteriors_ref, posteriors_d2p)}"

    def test_gradients(self, batch_size, device):
        """Test gradients through the soft alignment."""
        T, S = 8, 5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)
        scores.requires_grad_(True)

        # Use soft_mas_float which has autograd support
        score, _ = d2p.soft_mas_float(scores, temperature, None)
        loss = score.sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _ = soft_mas_forward_naive(scores_ref, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-2, atol=1e-3), \
            f"Gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"

    def test_score_gradients(self, batch_size, device):
        """Test gradients through the score output."""
        T, S = 8, 5
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(batch_size, T, S, device=device)
        scores.requires_grad_(True)

        score, _ = d2p.soft_mas_float(scores, temperature, None)
        loss = score.sum()
        loss.backward()
        grad_d2p = scores.grad.clone()

        scores_ref = scores.detach().clone().requires_grad_(True)
        score_ref, _ = soft_mas_forward_naive(scores_ref, temperature)
        loss_ref = score_ref.sum()
        loss_ref.backward()
        grad_ref = scores_ref.grad

        assert allclose(grad_ref, grad_d2p, rtol=1e-3, atol=1e-4), \
            f"Score gradient mismatch: max diff = {max_diff(grad_ref, grad_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestHVP:

    def test_hvp_finite_diff(self, device):
        """Test HVP against finite differences."""
        B, T, S = 2, 8, 5
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)
        V = torch.randn(B, T, S, device=device)

        hvp_d2p = d2p.soft_mas_hvp(scores, V, temperature, None)

        posteriors_plus = soft_mas_naive(scores + eps * V, temperature)
        posteriors_minus = soft_mas_naive(scores - eps * V, temperature)
        hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
            f"HVP mismatch: max diff = {max_diff(hvp_fd, hvp_d2p)}"

    def test_hvp_various_temps(self, device):
        """Test HVP with various temperatures."""
        B, T, S = 2, 10, 6
        eps = 1e-4

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores = torch.randn(B, T, S, device=device)
            V = torch.randn(B, T, S, device=device)

            hvp_d2p = d2p.soft_mas_hvp(scores, V, temperature, None)

            posteriors_plus = soft_mas_naive(scores + eps * V, temperature)
            posteriors_minus = soft_mas_naive(scores - eps * V, temperature)
            hvp_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

            assert allclose(hvp_fd, hvp_d2p, rtol=1e-2, atol=5e-3), \
                f"HVP mismatch for temp={temperature}: max diff = {max_diff(hvp_fd, hvp_d2p)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestCPUThreading:

    def test_thread_count_stability(self):
        """Representative CPU outputs should stay correct and bit-exact across thread counts."""
        B, T, S = 8, 9, 6
        temperature = 1.0
        eps = 1e-4
        thread_counts = (1, 2, 4)

        torch.manual_seed(123)
        scores = torch.randn(B, T, S)
        tangent = torch.randn(B, T, S)

        with torch_num_threads(1):
            partition_ref, _ = soft_mas_forward_naive(scores, temperature)
            posteriors_ref = soft_mas_naive(scores, temperature)

            partition_temp_plus, _ = soft_mas_forward_naive(scores, temperature + eps)
            partition_temp_minus, _ = soft_mas_forward_naive(scores, temperature - eps)
            grad_T_ref = (partition_temp_plus - partition_temp_minus) / (2 * eps)

            posteriors_plus = soft_mas_naive(scores + eps * tangent, temperature)
            posteriors_minus = soft_mas_naive(scores - eps * tangent, temperature)
            hvp_ref = (posteriors_plus - posteriors_minus) / (2 * eps)

            dP_dT_plus = soft_mas_naive(scores, temperature + eps)
            dP_dT_minus = soft_mas_naive(scores, temperature - eps)
            dP_dT_ref = (dP_dT_plus - dP_dT_minus) / (2 * eps)

        reference_outputs = {
            "partition": partition_ref,
            "posteriors": posteriors_ref,
            "grad_T": grad_T_ref,
            "hvp": hvp_ref,
            "dP_dT": dP_dT_ref,
        }

        outputs_by_thread = {}
        for thread_count in thread_counts:
            with torch_num_threads(thread_count):
                outputs = run_cpu_representative_outputs(scores, tangent, temperature)
            assert_threaded_mas_correctness(outputs, reference_outputs, thread_count)
            outputs_by_thread[thread_count] = outputs

        baseline = outputs_by_thread[1]
        assert_exact_thread_match(baseline, outputs_by_thread[2], 2)
        assert_exact_thread_match(baseline, outputs_by_thread[4], 4)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestCPUCUDA:

    def test_consistency(self):
        """Test CPU vs CUDA produce identical results."""
        B, T, S = 2, 12, 8
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        _, posteriors_cpu = d2p.soft_mas_with_grads(scores_cpu, temperature, None)
        _, posteriors_cuda = d2p.soft_mas_with_grads(scores_cuda, temperature, None)

        assert allclose(posteriors_cpu, posteriors_cuda), \
            f"CPU/CUDA mismatch: max diff = {max_diff(posteriors_cpu, posteriors_cuda)}"

    def test_consistency_score(self):
        """Test CPU vs CUDA scores match."""
        B, T, S = 2, 15, 10
        temperature = 1.0

        torch.manual_seed(42)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        score_cpu, _ = d2p.soft_mas_with_grads(scores_cpu, temperature, None)
        score_cuda, _ = d2p.soft_mas_with_grads(scores_cuda, temperature, None)

        assert allclose(score_cpu, score_cuda), \
            f"CPU/CUDA score mismatch: max diff = {max_diff(score_cpu, score_cuda)}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestVariableLength:

    def test_variable_lengths(self, device):
        """Test with variable sequence lengths in batch."""
        B = 4
        max_T, max_S = 15, 10
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, max_T, max_S, device=device)

        # Variable lengths [T, S] pairs
        lengths = torch.tensor([
            [12, 8],
            [15, 10],
            [10, 6],
            [14, 9]
        ], device=device, dtype=torch.int32)

        score, posteriors = d2p.soft_mas_with_grads(scores, temperature, lengths)

        # Check each batch element individually
        for b in range(B):
            t_len, s_len = lengths[b].tolist()
            scores_b = scores[b:b+1, :t_len, :s_len]

            score_ref, _ = soft_mas_forward_naive(scores_b, temperature)
            posteriors_ref = soft_mas_naive(scores_b, temperature)

            assert allclose(score_ref, score[b:b+1]), \
                f"Score mismatch for batch {b}: {score_ref.item()} vs {score[b].item()}"

            assert allclose(posteriors_ref, posteriors[b:b+1, :t_len, :s_len], rtol=1e-3, atol=1e-4), \
                f"Posteriors mismatch for batch {b}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestLengthValidation:

    @staticmethod
    def call_op(op_name, scores, tangent, temperature, lengths):
        if op_name == "soft_mas":
            return torch.ops.d2p.soft_mas(scores, temperature, lengths)
        if op_name == "soft_mas_float":
            return d2p.soft_mas_float(scores, temperature, lengths)
        if op_name == "soft_mas_with_grads":
            return d2p.soft_mas_with_grads(scores, temperature, lengths)
        if op_name == "soft_mas_hvp":
            return d2p.soft_mas_hvp(scores, tangent, temperature, lengths)
        if op_name == "soft_mas_param_jacobian":
            return d2p.soft_mas_param_jacobian(scores, temperature, lengths)
        if op_name == "soft_mas_backward_full":
            return d2p.soft_mas_backward_full(scores, temperature, lengths)
        raise AssertionError(f"unknown MAS op {op_name}")

    @pytest.mark.parametrize(
        "op_name",
        [
            "soft_mas",
            "soft_mas_float",
            "soft_mas_with_grads",
            "soft_mas_hvp",
            "soft_mas_param_jacobian",
            "soft_mas_backward_full",
        ],
    )
    @pytest.mark.parametrize(
        ("device_type", "lengths", "match"),
        [
            ("cpu", [[0, 5]], r"lengths\[0,0\] must be between 1 and 4"),
            ("cpu", [[4, 0]], r"lengths\[0,1\] must be between 1 and 5"),
            ("cpu", [[5, 5]], r"lengths\[0,0\] must be between 1 and 4"),
            ("cpu", [[4, 6]], r"lengths\[0,1\] must be between 1 and 5"),
            pytest.param(
                "cuda",
                [[0, 5]],
                r"lengths\[0,0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 0]],
                r"lengths\[0,1\] must be between 1 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[5, 5]],
                r"lengths\[0,0\] must be between 1 and 4",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
            pytest.param(
                "cuda",
                [[4, 6]],
                r"lengths\[0,1\] must be between 1 and 5",
                marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available"),
            ),
        ],
    )
    def test_invalid_lengths_raise_at_op_boundary(self, op_name, device_type, lengths, match):
        """All MAS entry points must reject zero and over-length rows."""
        device = torch.device(device_type)
        scores = torch.randn(1, 4, 5, device=device)
        tangent = torch.randn_like(scores)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)

        with pytest.raises(RuntimeError, match=match):
            self.call_op(op_name, scores, tangent, 1.0, lengths_t)

    def test_lengths_shape_dtype_and_contiguity_raise(self):
        """Explicit CPU lengths must have the expected boundary contract."""
        scores = torch.randn(1, 4, 5)

        with pytest.raises(RuntimeError, match=r"lengths must be \[B, 2\]"):
            d2p.soft_mas_float(scores, 1.0, torch.tensor([4, 5], dtype=torch.int32))

        with pytest.raises(RuntimeError, match=r"lengths must be int32"):
            d2p.soft_mas_float(scores, 1.0, torch.tensor([[4, 5]], dtype=torch.int64))

        non_contiguous = torch.tensor([[4, 0], [5, 0]], dtype=torch.int32)[:, 0].unsqueeze(0)
        assert not non_contiguous.is_contiguous()
        with pytest.raises(RuntimeError, match=r"lengths must be contiguous"):
            d2p.soft_mas_float(scores, 1.0, non_contiguous)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_cuda_lengths_must_match_scores_device(self):
        """CUDA MAS must reject CPU lengths before launching kernels."""
        scores = torch.randn(1, 4, 5, device="cuda")
        lengths = torch.tensor([[4, 5]], dtype=torch.int32)

        with pytest.raises(RuntimeError, match=r"lengths must be a CUDA tensor"):
            d2p.soft_mas_float(scores, 1.0, lengths)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_public_soft_mas_audit_repro_raises(self):
        """The confirmed public repro should raise cleanly instead of faulting."""
        scores = torch.randn(1, 8, 12, device="cuda")
        lengths = torch.tensor([[0, 12]], dtype=torch.int32, device="cuda")

        with pytest.raises(RuntimeError, match=r"lengths\[0,0\] must be between 1 and 8"):
            d2p.soft_mas(scores, 1.0, lengths=lengths)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestEdgeCases:

    def test_single_element(self, device):
        """Test 1x1 score matrix."""
        scores = torch.tensor([[[0.5]]], device=device)
        temperature = 1.0

        score, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)
        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        posteriors_ref = soft_mas_naive(scores, temperature)

        # For 1x1, posterior should be 1.0 (only one path)
        assert allclose(score, score_ref), f"Single element score wrong: {score.item()} vs {score_ref.item()}"
        assert allclose(posteriors, posteriors_ref), f"Single element posterior wrong: {posteriors.item()} vs {posteriors_ref.item()}"

    def test_single_text_token(self, device):
        """Test Tx1 score matrix (single text token)."""
        scores = torch.tensor([[[0.1], [0.2], [0.3], [0.4]]], device=device)
        temperature = 1.0

        score, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)
        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        posteriors_ref = soft_mas_naive(scores, temperature)

        # All frames must align to the single text token
        assert allclose(score, score_ref), "Single text token score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Single text token posteriors mismatch"

    def test_equal_length(self, device):
        """Test TxT score matrix (one-to-one alignment)."""
        T = 5
        scores = torch.randn(1, T, T, device=device)
        temperature = 1.0

        score, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)
        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        posteriors_ref = soft_mas_naive(scores, temperature)

        assert allclose(score, score_ref), "Equal length score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "Equal length posteriors mismatch"

    def test_low_temperature(self, device):
        """Test with low temperature (approaches hard alignment)."""
        B, T, S = 2, 10, 6
        temperature = 0.01

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        _, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)

        # With low temperature, posteriors should be close to 0 or 1
        assert posteriors.min() >= -0.1, "Low temp posteriors should be >= 0"
        assert posteriors.max() <= 1.1, "Low temp posteriors should be <= 1"

    def test_high_temperature(self, device):
        """Test with high temperature (more uniform distribution)."""
        B, T, S = 2, 10, 6
        temperature = 10.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        _, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)
        posteriors_ref = soft_mas_naive(scores, temperature)

        assert allclose(posteriors_ref, posteriors, rtol=1e-3, atol=1e-4), \
            "High temperature posteriors mismatch"

    def test_monotonicity_constraint(self, device):
        """Test that alignment respects monotonicity."""
        B, T, S = 1, 8, 4
        temperature = 0.1  # Low temp for clear alignment

        # Create scores that favor diagonal alignment
        scores = torch.zeros(B, T, S, device=device)
        for t in range(T):
            s = min(t // 2, S - 1)  # Expected alignment: each text gets ~2 frames
            scores[0, t, s] = 5.0

        _, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)

        # Posteriors should concentrate on the diagonal region
        # (monotonicity is enforced by the DP structure)
        assert posteriors.sum(dim=-1).min() > 0.9, "Each frame should align somewhere"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestParamJacobian:

    def test_param_jacobian_temperature_cpu(self):
        """CPU dP/dT should match a finite-difference reference."""
        B, T, S = 2, 9, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, T, S)

        with torch_num_threads(1):
            dP_dT = d2p.soft_mas_param_jacobian(scores, temperature, None)
            posteriors_plus = soft_mas_naive(scores, temperature + eps)
            posteriors_minus = soft_mas_naive(scores, temperature - eps)
            dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=2e-2, atol=5e-3), \
            f"Param Jacobian (T) mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_cpu_parity(self):
        """CUDA dP/dT should match CPU dP/dT (r27 regression)."""
        B, T, S = 2, 9, 6
        temperature = 1.0

        torch.manual_seed(123)
        scores_cpu = torch.randn(B, T, S)
        scores_cuda = scores_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = d2p.soft_mas_param_jacobian(scores_cpu, temperature, None)
        dP_dT_cuda = d2p.soft_mas_param_jacobian(scores_cuda, temperature, None)

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_finite_diff(self):
        """CUDA dP/dT should match a finite-difference reference (r27 regression)."""
        B, T, S = 2, 9, 6
        temperature = 1.0
        eps = 1e-4

        torch.manual_seed(123)
        scores = torch.randn(B, T, S).cuda()

        dP_dT = d2p.soft_mas_param_jacobian(scores, temperature, None)
        posteriors_plus = soft_mas_naive(scores, temperature + eps)
        posteriors_minus = soft_mas_naive(scores, temperature - eps)
        dP_dT_fd = (posteriors_plus - posteriors_minus) / (2 * eps)

        assert allclose(dP_dT_fd, dP_dT, rtol=2e-2, atol=5e-3), \
            f"CUDA param Jacobian vs finite diff mismatch: max diff = {max_diff(dP_dT_fd, dP_dT)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_various_temps(self):
        """CUDA dP/dT should match CPU across multiple temperatures (r27 regression)."""
        B, T, S = 2, 10, 6

        for temperature in [0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            scores_cpu = torch.randn(B, T, S)
            scores_cuda = scores_cpu.cuda()

            with torch_num_threads(1):
                dP_dT_cpu = d2p.soft_mas_param_jacobian(scores_cpu, temperature, None)
            dP_dT_cuda = d2p.soft_mas_param_jacobian(scores_cuda, temperature, None)

            assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
                f"CUDA/CPU param Jacobian mismatch for temp={temperature}: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_first_column_regression(self):
        """CUDA dP/dT should write back the propagated first-column value at (0,0) (r45 regression)."""
        temperature = 0.1
        scores_cpu = PARAM_JACOBIAN_FIRST_COLUMN_REGRESSION_SCORES.clone()
        scores_cuda = scores_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = d2p.soft_mas_param_jacobian(scores_cpu, temperature, None)
        dP_dT_cuda = d2p.soft_mas_param_jacobian(scores_cuda, temperature, None).cpu()

        assert abs(dP_dT_cpu[0, 0, 0].item()) > 1e-5, \
            "regression fixture expected a nonzero CPU first-column dP/dT at (0,0)"
        assert abs(dP_dT_cuda[0, 0, 0].item()) > 1e-5, \
            f"CUDA left first-column dP/dT at (0,0) effectively zero: {dP_dT_cuda[0, 0, 0].item()}"
        assert torch.sign(dP_dT_cuda[0, 0, 0]) == torch.sign(dP_dT_cpu[0, 0, 0]), \
            f"CUDA first-column dP/dT changed sign at (0,0): cpu={dP_dT_cpu[0, 0, 0].item()} cuda={dP_dT_cuda[0, 0, 0].item()}"
        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU param Jacobian mismatch in first-column regression: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_param_jacobian_cuda_variable_lengths_masked_parity(self):
        """CUDA dP/dT should respect per-batch lengths and keep padded regions zero."""
        B, max_T, max_S = 3, 12, 6
        temperature = 0.9

        torch.manual_seed(20260409)
        scores_cpu = torch.randn(B, max_T, max_S)
        lengths_cpu = torch.tensor([[12, 6], [7, 4], [1, 1]], dtype=torch.int32)
        scores_cuda = scores_cpu.cuda()
        lengths_cuda = lengths_cpu.cuda()

        with torch_num_threads(1):
            dP_dT_cpu = d2p.soft_mas_param_jacobian(scores_cpu, temperature, lengths_cpu)
        dP_dT_cuda = d2p.soft_mas_param_jacobian(scores_cuda, temperature, lengths_cuda).cpu()

        assert allclose(dP_dT_cpu, dP_dT_cuda, rtol=1e-3, atol=1e-4), \
            f"CUDA/CPU variable-length param Jacobian mismatch: max diff = {max_diff(dP_dT_cpu, dP_dT_cuda)}"

        for batch, (t_len, s_len) in enumerate(lengths_cpu.tolist()):
            if t_len < max_T:
                assert torch.count_nonzero(dP_dT_cuda[batch, t_len:, :]).item() == 0, \
                    f"CUDA wrote past active T length for batch {batch}"
            if s_len < max_S:
                assert torch.count_nonzero(dP_dT_cuda[batch, :, s_len:]).item() == 0, \
                    f"CUDA wrote past active S length for batch {batch}"


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
class TestWithGrads:

    def test_with_grads_output(self, device):
        """Test soft_mas_with_grads returns correct values."""
        B, T, S = 2, 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        score, posteriors = d2p.soft_mas_with_grads(scores, temperature, None)

        # Compare with reference
        score_ref, _ = soft_mas_forward_naive(scores, temperature)
        posteriors_ref = soft_mas_naive(scores, temperature)

        assert allclose(score, score_ref), "with_grads score mismatch"
        assert allclose(posteriors, posteriors_ref, rtol=1e-3), "with_grads posteriors mismatch"

    def test_backward_full_output(self, device):
        """Test soft_mas_backward_full returns correct values."""
        B, T, S = 2, 10, 6
        temperature = 1.0

        torch.manual_seed(42)
        scores = torch.randn(B, T, S, device=device)

        score, posteriors, grad_T = d2p.soft_mas_backward_full(scores, temperature, None)

        # Compare with soft_mas_with_grads
        score_ref, posteriors_ref = d2p.soft_mas_with_grads(scores, temperature, None)

        assert allclose(score, score_ref), "backward_full score mismatch"
        assert allclose(posteriors, posteriors_ref), "backward_full posteriors mismatch"
        assert grad_T.shape == (B,), f"grad_T shape wrong: {grad_T.shape}"


# --- Memory-safety regression tests (merged from test_mas_{cpp,cuda}_memsafety.py) ---

import re
from pathlib import Path


def _make_noncontiguous(tensor):
    storage = torch.empty(
        tensor.size(0),
        tensor.size(1),
        tensor.size(2) * 2,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    view = storage[..., ::2]
    view.copy_(tensor)
    assert not view.is_contiguous()
    return view


def _assert_tree_allclose(actual, expected):
    if isinstance(actual, (tuple, list)):
        assert isinstance(expected, (tuple, list))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_allclose(actual_item, expected_item)
        return

    assert actual.shape == expected.shape
    if actual.dim() == 3:
        assert actual.is_contiguous()
    assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)


CPU_MAS_CALLS = {
    "soft_mas": lambda scores, tangent: torch.ops.d2p.soft_mas(scores, 1.0, None),
    "soft_mas_float": lambda scores, tangent: d2p.soft_mas_float(scores, 1.0, None),
    "soft_mas_with_grads": lambda scores, tangent: d2p.soft_mas_with_grads(scores, 1.0, None),
    "soft_mas_hvp": lambda scores, tangent: d2p.soft_mas_hvp(scores, tangent, 1.0, None),
    "soft_mas_param_jacobian": lambda scores, tangent: d2p.soft_mas_param_jacobian(
        scores, 1.0, None
    ),
    "soft_mas_backward_full": lambda scores, tangent: d2p.soft_mas_backward_full(
        scores, 1.0, None
    ),
}


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.parametrize("op_name", CPU_MAS_CALLS)
def test_noncontiguous_scores_return_fully_initialized_contiguous_outputs(op_name):
    torch.manual_seed(20260707)
    scores_contig = torch.randn(2, 6, 4)
    tangent_contig = torch.randn_like(scores_contig)
    scores_view = _make_noncontiguous(scores_contig)
    tangent_view = _make_noncontiguous(tangent_contig)

    call = CPU_MAS_CALLS[op_name]
    expected = call(scores_contig, tangent_contig)
    actual = call(scores_view, tangent_view)

    _assert_tree_allclose(actual, expected)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.parametrize("op_name", CPU_MAS_CALLS)
def test_rank4_scores_rejected_at_cpu_boundary(op_name):
    scores = torch.randn(1, 4, 5, 2)
    tangent = torch.randn_like(scores)

    with pytest.raises(RuntimeError, match=r"scores must be 3D"):
        CPU_MAS_CALLS[op_name](scores, tangent)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_hvp_rejects_bad_tangent_shape_before_cpu_kernel():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 3)

    with pytest.raises(RuntimeError, match=r"V must have same shape as scores"):
        d2p.soft_mas_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_hvp_rejects_bad_tangent_dtype_before_cpu_kernel():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 4, dtype=torch.float64)

    with pytest.raises(RuntimeError, match=r"V must be float32"):
        d2p.soft_mas_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hvp_rejects_cross_device_tangent_cleanly():
    scores = torch.randn(2, 6, 4)
    tangent = torch.randn(2, 6, 4, device="cuda")

    with pytest.raises(RuntimeError, match=r"(scores must be a CUDA tensor|V must be a CPU tensor)"):
        d2p.soft_mas_hvp(scores, tangent, 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
def test_cpu_kernel_cell_offsets_are_wide():
    kernel_path = Path(__file__).resolve().parents[1] / "src" / "mas" / "kernels_cpu.cpp"
    kernel = kernel_path.read_text()

    assert "static inline size_t cell_index" in kernel
    assert "int idx = t * max_S + s" not in kernel
    assert "int idx_stay = (t - 1) * max_S + s" not in kernel
    assert "int idx_diag = (t - 1) * max_S + (s - 1)" not in kernel


def _source_text(relative_path):
    return (Path(__file__).resolve().parents[1] / relative_path).read_text()


def _as_tuple(result):
    # Some ops (e.g. soft_mas_float) return a list of tensors, others a tuple,
    # others a single tensor. Normalise all of them to a tuple so callers can
    # iterate and compare element-wise.
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def _assert_outputs_match(name, actual, expected, scores_shape):
    for index, (actual_t, expected_t) in enumerate(zip(_as_tuple(actual), _as_tuple(expected))):
        torch.testing.assert_close(actual_t, expected_t, rtol=1e-5, atol=1e-5)
        if actual_t.dim() == 3:
            assert tuple(actual_t.shape) == scores_shape, f"{name}[{index}] has wrong shape"
            assert actual_t.is_contiguous(), f"{name}[{index}] must be contiguous"


def test_cuda_kernels_use_wide_cell_offsets_for_mas_indices():
    text = _source_text("src/mas/kernels.cu")

    assert "__device__ __forceinline__ size_t cell_offset" in text
    assert "size_t total_valid = (size_t)T * (size_t)S;" in text

    banned_patterns = [
        r"\bint\s+(idx|idx_stay|idx_diag|final_idx|flat_idx)\s*=",
        r"\bfor\s*\(\s*int\s+idx\s*=\s*threadIdx\.x;\s*idx\s*<\s*T\s*\*\s*S",
        r"\bt\s*\*\s*max_S",
        r"\(T\s*-\s*1\)\s*\*\s*max_S",
    ]
    for pattern in banned_patterns:
        assert re.search(pattern, text) is None, f"found narrow MAS CUDA index pattern: {pattern}"


def test_cuda_wrappers_guard_mas_launch_device_and_validate_tangent():
    text = _source_text("src/mas/torch_cuda.cpp")

    assert text.count("D2P_CUDA_GUARD(scores)") >= 6
    assert "validate_mas_tangent_cuda(scores, V);" in text
    assert "tangent.device() == scores.device()" in text


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_hvp_rejects_bad_tangent_shape_dtype_and_device():
    scores = torch.randn(2, 5, 4, device="cuda")

    with pytest.raises(RuntimeError, match="tangent must have same shape as scores"):
        d2p.soft_mas_hvp(scores, torch.randn(2, 5, 3, device="cuda"), 1.0, None)

    with pytest.raises(RuntimeError, match="tangent must be float32"):
        d2p.soft_mas_hvp(scores, torch.randn(2, 5, 4, device="cuda", dtype=torch.float64), 1.0, None)

    with pytest.raises(RuntimeError, match="tangent must be a CUDA tensor"):
        d2p.soft_mas_hvp(scores, torch.randn(2, 5, 4), 1.0, None)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("soft_mas_float", lambda scores: d2p.soft_mas_float(scores, 1.0, None)),
        ("soft_mas_with_grads", lambda scores: d2p.soft_mas_with_grads(scores, 1.0, None)),
        ("soft_mas_hvp", lambda scores: d2p.soft_mas_hvp(scores, torch.randn_like(scores), 1.0, None)),
        ("soft_mas_param_jacobian", lambda scores: d2p.soft_mas_param_jacobian(scores, 1.0, None)),
        ("soft_mas_backward_full", lambda scores: d2p.soft_mas_backward_full(scores, 1.0, None)),
    ],
)
def test_cuda_wrappers_reject_rank4_scores(name, call):
    scores = torch.randn(1, 5, 4, 2, device="cuda")

    with pytest.raises(RuntimeError, match=r"scores must be 3D \[B, T, S\]"):
        call(scores)


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_noncontiguous_scores_match_contiguous_outputs():
    torch.manual_seed(123)
    scores = torch.randn(2, 5, 8, device="cuda")[:, :, ::2]
    tangent = torch.randn(2, 5, 8, device="cuda")[:, :, ::2]
    assert not scores.is_contiguous()
    assert not tangent.is_contiguous()

    scores_c = scores.contiguous()
    tangent_c = tangent.contiguous()
    cases = [
        ("soft_mas_float", lambda s, v: d2p.soft_mas_float(s, 1.0, None)),
        ("soft_mas_with_grads", lambda s, v: d2p.soft_mas_with_grads(s, 1.0, None)),
        ("soft_mas_hvp", lambda s, v: d2p.soft_mas_hvp(s, v, 1.0, None)),
        ("soft_mas_param_jacobian", lambda s, v: d2p.soft_mas_param_jacobian(s, 1.0, None)),
        ("soft_mas_backward_full", lambda s, v: d2p.soft_mas_backward_full(s, 1.0, None)),
    ]

    for name, call in cases:
        actual = call(scores, tangent)
        expected = call(scores_c, tangent_c)
        _assert_outputs_match(name, actual, expected, tuple(scores.shape))


@pytest.mark.skipif(not D2P_AVAILABLE, reason="d2p not built")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_current_device_can_differ_from_scores_device():
    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        scores = torch.randn(2, 5, 4, device="cuda:1")
        tangent = torch.randn_like(scores)

        partition, posteriors = d2p.soft_mas_with_grads(scores, 1.0, None)
        hvp = d2p.soft_mas_hvp(scores, tangent, 1.0, None)

        assert partition.device == scores.device
        assert posteriors.device == scores.device
        assert hvp.device == scores.device
        torch.cuda.synchronize(scores.device)
    finally:
        torch.cuda.set_device(previous_device)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
