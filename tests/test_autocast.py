"""
Tests for autocast (AMP) support.

Verifies that d2p operators correctly promote inputs to FP32 when running
under torch.autocast, ensuring numerical stability for DP algorithms.
"""

import pytest
import torch
import d2p


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastSW:
    """Test autocast for Smith-Waterman operators."""

    def test_soft_sw_autocast_promotes_to_fp32(self):
        """Verify soft_sw promotes FP16 inputs to FP32 under autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            result = d2p.soft_sw(scores, gap=-1.0)

        # Outputs should be FP32 for numerical stability
        assert result.value.dtype == torch.float32
        assert result.marginals.dtype == torch.float32

    def test_soft_sw_autocast_bf16(self):
        """Verify soft_sw works with BF16 autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.bfloat16)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = d2p.soft_sw(scores, gap=-1.0)

        assert result.value.dtype == torch.float32
        assert result.marginals.dtype == torch.float32

    def test_soft_sw_affine_autocast(self):
        """Verify soft_sw_affine promotes inputs to FP32 under autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            result = d2p.soft_sw_affine(scores, gap_open=-2.0, gap_ext=-0.5)

        assert result.value.dtype == torch.float32
        assert result.marginals.dtype == torch.float32

    def test_autocast_gradient_flow(self):
        """Verify gradients flow correctly with autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16, requires_grad=True)

        with torch.autocast("cuda", dtype=torch.float16):
            result = d2p.soft_sw(scores, gap=-1.0)
            loss = result.value.sum()

        loss.backward()

        # Gradient should match scores dtype (FP16)
        assert scores.grad is not None
        assert scores.grad.dtype == torch.float16

    def test_autocast_numerical_stability(self):
        """Compare autocast results with explicit FP32 for stability."""
        torch.manual_seed(42)
        scores_fp16 = torch.randn(4, 16, 20, device="cuda", dtype=torch.float16)
        scores_fp32 = scores_fp16.float()

        # Run with autocast (should promote to FP32)
        with torch.autocast("cuda", dtype=torch.float16):
            result_autocast = d2p.soft_sw(scores_fp16, gap=-1.0)

        # Run without autocast using FP32
        result_fp32 = d2p.soft_sw(scores_fp32, gap=-1.0)

        # Results should be very close (both computed in FP32)
        torch.testing.assert_close(
            result_autocast.value,
            result_fp32.value,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            result_autocast.marginals,
            result_fp32.marginals,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastNoEffect:
    """Test that autocast doesn't break FP32 operations."""

    def test_fp32_unchanged_under_autocast(self):
        """FP32 inputs should pass through unchanged."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float32)

        with torch.autocast("cuda", dtype=torch.float16):
            result = d2p.soft_sw(scores, gap=-1.0)

        assert result.value.dtype == torch.float32
        assert result.marginals.dtype == torch.float32

    def test_disabled_autocast(self):
        """Operations should work normally without autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float32)

        result = d2p.soft_sw(scores, gap=-1.0)

        assert result.value.dtype == torch.float32
        assert result.marginals.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestAutocastNewAPI:
    """Test autocast for new namespaced API (low-level operators)."""

    def test_sw_forward_autocast(self):
        """Verify low-level soft_sw_float promotes inputs to FP32 under autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            # Low-level API returns [score, alignment] list
            value, marginals = d2p.sw.soft_sw_forward(scores, -1.0, 1.0, None)

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32

    def test_sw_affine_forward_autocast(self):
        """Verify low-level soft_sw_affine_float promotes inputs to FP32 under autocast."""
        scores = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16)

        with torch.autocast("cuda", dtype=torch.float16):
            # Low-level API returns [score, alignment] list
            value, marginals = d2p.sw.soft_sw_affine_forward(
                scores, -2.0, -0.5, 1.0, None
            )

        assert value.dtype == torch.float32
        assert marginals.dtype == torch.float32
