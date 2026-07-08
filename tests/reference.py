"""
Naive PyTorch reference implementations for differentiable DP algorithms.

These are slow but correct implementations using standard PyTorch ops,
used to validate the optimized CUDA/C++ implementations.

The algorithms match the d2p implementations:
- soft_sw_regular: Single-state local alignment with linear gap penalty
- soft_sw_affine: Three-state local alignment with affine gap penalty
- soft_cky: CKY parsing with merge scores
- soft_dtw: Dynamic time warping with optional Sakoe-Chiba band
- soft_nw: Needleman-Wunsch global alignment with linear gap penalty
- soft_nw_affine: Needleman-Wunsch global alignment with affine gap penalty
"""

import torch
from typing import Optional


def logsumexp_temp(x: torch.Tensor, T: torch.Tensor | float, dim: int = -1) -> torch.Tensor:
    """Temperature-scaled logsumexp: T * log(sum(exp(x/T)))"""
    return T * torch.logsumexp(x / T, dim=dim)


def softmin_temp(x: torch.Tensor, T: float, dim: int = -1) -> torch.Tensor:
    """Temperature-scaled softmin: -T * log(sum(exp(-x/T)))"""
    return -T * torch.logsumexp(-x / T, dim=dim)


# =============================================================================
# Soft Smith-Waterman (Regular / Linear Gap) - LOCAL ALIGNMENT
# =============================================================================

def soft_sw_regular_forward_naive(
    scores: torch.Tensor,
    gap: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft Smith-Waterman with linear gap penalty (LOCAL alignment).

    This matches the d2p implementation which uses:
    - Single DP state
    - Four transitions: align, gap_up, gap_left, sky (start fresh)
    - Partition = logsumexp over ALL cells (local alignment)

    Args:
        scores: [B, L1, L2] similarity scores
        gap: gap penalty (typically negative)
        temperature: softmax temperature

    Returns:
        partition: [B] partition function (soft alignment score)
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    NINF = float('-inf')

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)
    alpha[:, 0, 0] = 0.0  # Base case

    # Fill DP table - process cells in order
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            score = scores[:, i-1, j-1]

            # Four transitions (matching d2p implementation):
            # 1. Align: come from diagonal and match
            v_align = alpha[:, i-1, j-1] + score if (i > 1 and j > 1) else torch.full((B,), NINF, device=device, dtype=dtype)
            if i == 1 or j == 1:
                v_align = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                v_align = alpha[:, i-1, j-1] + score

            # 2. Gap up: come from above (gap in seq2)
            v_up = alpha[:, i-1, j] + gap if i > 1 else torch.full((B,), NINF, device=device, dtype=dtype)
            if i == 1:
                v_up = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                v_up = alpha[:, i-1, j] + gap

            # 3. Gap left: come from left (gap in seq1)
            v_left = alpha[:, i, j-1] + gap if j > 1 else torch.full((B,), NINF, device=device, dtype=dtype)
            if j == 1:
                v_left = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                v_left = alpha[:, i, j-1] + gap

            # 4. Sky: start fresh at this position (local alignment)
            v_sky = score

            candidates = torch.stack([v_align, v_up, v_left, v_sky], dim=-1)
            alpha[:, i, j] = logsumexp_temp(candidates, temperature)

    # Partition function: logsumexp over ALL cells (local alignment)
    # Include all cells from (0,0) to (L1, L2)
    all_values = alpha.reshape(B, -1)
    partition = logsumexp_temp(all_values, temperature, dim=-1)

    return partition, alpha


def soft_sw_regular_naive(
    scores: torch.Tensor,
    gap: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Smith-Waterman returning soft alignment (posteriors).
    Uses autograd to compute gradients.

    Note: If input scores requires_grad, gradients will flow through.
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    partition, _ = soft_sw_regular_forward_naive(scores, gap, temperature)

    # Gradient of partition w.r.t. scores gives alignment posteriors
    posteriors = torch.autograd.grad(
        partition.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft Smith-Waterman (Affine Gap) - LOCAL ALIGNMENT
# =============================================================================

def soft_sw_affine_forward_naive(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Naive soft Smith-Waterman with affine gap penalty (LOCAL alignment).
    Three-state DP: M (match), I (insert/gap in seq2), D (delete/gap in seq1).

    Args:
        scores: [B, L1, L2] similarity scores
        gap_open: gap opening penalty (typically negative)
        gap_ext: gap extension penalty (typically negative)
        temperature: softmax temperature

    Returns:
        partition: [B] partition function
        M, I, D: [B, L1+1, L2+1] DP tables for each state
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    NINF = float('-inf')

    # Initialize DP tables
    M = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)
    I = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)
    D = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)

    M[:, 0, 0] = 0.0

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            s = scores[:, i-1, j-1]

            # M[i,j]: match/mismatch - come from any state at (i-1, j-1) OR start fresh
            m_from_M = M[:, i-1, j-1] if (i > 1 and j > 1) else torch.full((B,), NINF, device=device, dtype=dtype)
            m_from_I = I[:, i-1, j-1] if (i > 1 and j > 1) else torch.full((B,), NINF, device=device, dtype=dtype)
            m_from_D = D[:, i-1, j-1] if (i > 1 and j > 1) else torch.full((B,), NINF, device=device, dtype=dtype)

            if i == 1 or j == 1:
                m_from_M = torch.full((B,), NINF, device=device, dtype=dtype)
                m_from_I = torch.full((B,), NINF, device=device, dtype=dtype)
                m_from_D = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                m_from_M = M[:, i-1, j-1]
                m_from_I = I[:, i-1, j-1]
                m_from_D = D[:, i-1, j-1]

            # Sky transition: start fresh (local alignment)
            m_sky = torch.zeros(B, device=device, dtype=dtype)

            m_candidates = torch.stack([m_from_M + s, m_from_I + s, m_from_D + s, m_sky + s], dim=-1)
            M[:, i, j] = logsumexp_temp(m_candidates, temperature)

            # I[i,j]: gap in seq2 (insertion) - extend or open
            i_open = M[:, i-1, j] + gap_open if i > 1 else torch.full((B,), NINF, device=device, dtype=dtype)
            i_ext = I[:, i-1, j] + gap_ext if i > 1 else torch.full((B,), NINF, device=device, dtype=dtype)

            if i == 1:
                i_open = torch.full((B,), NINF, device=device, dtype=dtype)
                i_ext = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                i_open = M[:, i-1, j] + gap_open
                i_ext = I[:, i-1, j] + gap_ext

            i_candidates = torch.stack([i_open, i_ext], dim=-1)
            I[:, i, j] = logsumexp_temp(i_candidates, temperature)

            # D[i,j]: gap in seq1 (deletion) - extend or open
            d_open = M[:, i, j-1] + gap_open if j > 1 else torch.full((B,), NINF, device=device, dtype=dtype)
            d_ext = D[:, i, j-1] + gap_ext if j > 1 else torch.full((B,), NINF, device=device, dtype=dtype)

            if j == 1:
                d_open = torch.full((B,), NINF, device=device, dtype=dtype)
                d_ext = torch.full((B,), NINF, device=device, dtype=dtype)
            else:
                d_open = M[:, i, j-1] + gap_open
                d_ext = D[:, i, j-1] + gap_ext

            d_candidates = torch.stack([d_open, d_ext], dim=-1)
            D[:, i, j] = logsumexp_temp(d_candidates, temperature)

    # Partition: logsumexp over all cells across all states (local alignment)
    all_M = M.reshape(B, -1)
    all_I = I.reshape(B, -1)
    all_D = D.reshape(B, -1)
    all_values = torch.cat([all_M, all_I, all_D], dim=-1)
    partition = logsumexp_temp(all_values, temperature, dim=-1)

    return partition, M, I, D


def soft_sw_affine_naive(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Smith-Waterman (affine) returning soft alignment posteriors.

    Note: If input scores requires_grad, gradients will flow through.
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    partition, _, _, _ = soft_sw_affine_forward_naive(
        scores, gap_open, gap_ext, temperature
    )

    posteriors = torch.autograd.grad(
        partition.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft CKY Parsing
# =============================================================================

def soft_cky_forward_naive(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft CKY parsing.

    Args:
        merge_scores: [B, N, N, N] where merge_scores[b, i, k, j] is the score
                     for merging spans [i,k] and [k+1,j]
        leaf_scores: [B, N] scores for leaf spans (single elements)
        temperature: softmax temperature

    Returns:
        partition: [B] partition function (log of sum over all trees)
        chart: [B, N, N] CKY chart
    """
    B, N, _, _ = merge_scores.shape
    device = merge_scores.device
    dtype = merge_scores.dtype

    NINF = float('-inf')

    if isinstance(temperature, torch.Tensor):
        if temperature.numel() == 1:
            temp_tensor = temperature.reshape(1).to(device=device, dtype=dtype)
        else:
            temp_tensor = temperature.to(device=device, dtype=dtype)
            if temp_tensor.shape != (B, N, N):
                raise ValueError(
                    f"temperature must be scalar or [B, N, N], got {tuple(temp_tensor.shape)}"
                )
    else:
        temp_tensor = torch.tensor([float(temperature)], device=device, dtype=dtype)

    # chart[b, i, j] = log partition for span [i, j]
    chart = torch.full((B, N, N), NINF, device=device, dtype=dtype)

    # Base case: single elements (span of length 1)
    for i in range(N):
        chart[:, i, i] = leaf_scores[:, i]

    # Fill chart by span width
    for width in range(2, N + 1):
        for i in range(N - width + 1):
            j = i + width - 1

            # Collect all split points
            split_scores = []
            for k in range(i, j):
                # Split at k: left span [i, k], right span [k+1, j]
                # merge_scores[i, k, j] is the score for this merge
                split_score = chart[:, i, k] + chart[:, k+1, j] + merge_scores[:, i, k, j]
                split_scores.append(split_score)

            if split_scores:
                candidates = torch.stack(split_scores, dim=-1)
                if temp_tensor.numel() == 1:
                    chart[:, i, j] = logsumexp_temp(candidates, temp_tensor[0])
                else:
                    T_ij = temp_tensor[:, i, j].unsqueeze(-1)
                    chart[:, i, j] = T_ij.squeeze(-1) * torch.logsumexp(candidates / T_ij, dim=-1)

    partition = chart[:, 0, N-1]
    return partition, chart


def soft_cky_naive(
    merge_scores: torch.Tensor,
    leaf_scores: torch.Tensor,
    temperature: torch.Tensor | float
) -> torch.Tensor:
    """
    Soft CKY returning tree posteriors (split marginals).

    Returns posteriors [B, N, N, N] where posteriors[b, i, k, j] is the
    marginal probability of split point k for span [i, j].

    Note: If inputs require_grad, gradients will flow through.
    """
    needs_grad_merge = merge_scores.requires_grad
    needs_grad_leaf = leaf_scores.requires_grad

    if not needs_grad_merge:
        merge_scores = merge_scores.detach().requires_grad_(True)
    if not needs_grad_leaf:
        leaf_scores = leaf_scores.detach().requires_grad_(True)

    partition, _ = soft_cky_forward_naive(merge_scores, leaf_scores, temperature)

    # Gradient w.r.t. merge_scores gives tree posteriors
    posteriors = torch.autograd.grad(
        partition.sum(), merge_scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft DTW (Dynamic Time Warping) - GLOBAL ALIGNMENT
# =============================================================================

def soft_dtw_forward_naive(
    costs: torch.Tensor,
    temperature: float,
    bandwidth: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft DTW with softmin (GLOBAL alignment).

    DTW minimizes total cost, so uses softmin instead of softmax.

    Args:
        costs: [B, L1, L2] cost matrix (lower is better)
        temperature: softmin temperature
        bandwidth: Optional Sakoe-Chiba band width. If None, full DP table.
                  If set, only cells where |i*L2/L1 - j| <= bandwidth are computed.

    Returns:
        score: [B] DTW distance (soft minimum cost path)
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = costs.shape
    device = costs.device
    dtype = costs.dtype

    PINF = 1e30  # Use large finite value for numerical stability

    # Initialize alpha table
    alpha = torch.full((B, L1 + 1, L2 + 1), PINF, device=device, dtype=dtype)
    alpha[:, 0, 0] = 0.0  # Base case

    # Helper to check if cell is within Sakoe-Chiba band
    def in_band(i: int, j: int) -> bool:
        if bandwidth is None:
            return True
        # Map to diagonal: expected_j = i * L2 / L1
        if L1 == 0:
            return True
        expected_j = i * L2 / L1
        return abs(j - expected_j) <= bandwidth

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            if not in_band(i, j):
                continue  # Leave as +INF

            cost = costs[:, i-1, j-1]

            # Three transitions (GLOBAL alignment - no sky restart):
            # 1. Diagonal: come from (i-1, j-1)
            a_diag = alpha[:, i-1, j-1]
            # 2. Up: come from (i-1, j)
            a_up = alpha[:, i-1, j]
            # 3. Left: come from (i, j-1)
            a_left = alpha[:, i, j-1]

            candidates = torch.stack([a_diag, a_up, a_left], dim=-1)

            # softmin + cost (cost added to cell, not in options)
            alpha[:, i, j] = cost + softmin_temp(candidates, temperature)

    # Global alignment: score is just the final cell
    score = alpha[:, L1, L2]

    return score, alpha


def soft_dtw_naive(
    costs: torch.Tensor,
    temperature: float,
    bandwidth: Optional[int] = None
) -> torch.Tensor:
    """
    Soft DTW returning soft alignment (posteriors = expected cell occupancy).
    Uses autograd to compute gradients.

    For DTW, since cost c_{i,j} is added directly to the cell (not inside
    the options), the gradient dS/dc_{i,j} = beta_{i,j} (the outside values).

    Args:
        costs: [B, L1, L2] cost matrix
        temperature: softmin temperature
        bandwidth: Optional Sakoe-Chiba band width

    Returns:
        posteriors: [B, L1, L2] expected occupancy of each cell
    """
    needs_grad = costs.requires_grad
    if not needs_grad:
        costs = costs.detach().requires_grad_(True)

    score, _ = soft_dtw_forward_naive(costs, temperature, bandwidth)

    # Gradient of score w.r.t. costs gives alignment posteriors
    # For DTW: P[i,j] = beta[i,j] because cost is node-additive
    posteriors = torch.autograd.grad(
        score.sum(), costs, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft Needleman-Wunsch (Linear Gap) - GLOBAL ALIGNMENT
# =============================================================================

def soft_nw_forward_naive(
    scores: torch.Tensor,
    gap: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft Needleman-Wunsch with linear gap penalty (GLOBAL alignment).

    NW differs from SW in key ways:
    - No "sky" restart transition (must align full sequences)
    - Base cases: alpha(i,0) = i*gap, alpha(0,j) = j*gap (not NINF)
    - Final score is alpha(n,m) only (not logsumexp over all cells)

    NW differs from DTW in key ways:
    - Uses softmax (maximize scores) not softmin (minimize costs)
    - Score added to diagonal transition (option-additive), not to cell

    Args:
        scores: [B, L1, L2] similarity scores
        gap: gap penalty (typically negative)
        temperature: softmax temperature

    Returns:
        partition: [B] alignment score (alpha at final cell)
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    NINF = float('-inf')

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)

    # Base cases for GLOBAL alignment:
    # alpha(0,0) = 0
    # alpha(i,0) = i * gap (aligning i characters of seq1 to nothing)
    # alpha(0,j) = j * gap (aligning j characters of seq2 to nothing)
    alpha[:, 0, 0] = 0.0
    for i in range(1, L1 + 1):
        alpha[:, i, 0] = i * gap
    for j in range(1, L2 + 1):
        alpha[:, 0, j] = j * gap

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            score = scores[:, i-1, j-1]

            # Three transitions (GLOBAL alignment - no sky restart):
            # 1. Diagonal: match/mismatch - score added to this transition
            v_diag = alpha[:, i-1, j-1] + score
            # 2. Gap up: gap in seq2 (delete from seq1)
            v_up = alpha[:, i-1, j] + gap
            # 3. Gap left: gap in seq1 (insert into seq1)
            v_left = alpha[:, i, j-1] + gap

            candidates = torch.stack([v_diag, v_up, v_left], dim=-1)
            alpha[:, i, j] = logsumexp_temp(candidates, temperature)

    # Global alignment: score is just the final cell
    partition = alpha[:, L1, L2]

    return partition, alpha


def soft_nw_naive(
    scores: torch.Tensor,
    gap: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Needleman-Wunsch returning soft alignment posteriors.
    Uses autograd to compute gradients.

    For NW, since the match score s_{i,j} is added inside the diagonal option
    (option-additive), the gradient dS/ds_{i,j} = beta(i,j) * w_diag(i,j).

    Args:
        scores: [B, L1, L2] similarity scores
        gap: gap penalty (typically negative)
        temperature: softmax temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (match posteriors)
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    partition, _ = soft_nw_forward_naive(scores, gap, temperature)

    # Gradient of partition w.r.t. scores gives alignment posteriors
    # For NW: P[i,j] = beta[i,j] * w_diag[i,j] because score is option-additive
    posteriors = torch.autograd.grad(
        partition.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft Needleman-Wunsch (Affine Gap) - GLOBAL ALIGNMENT
# =============================================================================

def soft_nw_affine_forward_naive(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Naive soft Needleman-Wunsch with affine gap penalty (GLOBAL alignment).
    Three-state DP: M (match), I (insert/gap in seq2), D (delete/gap in seq1).

    Differs from SW affine:
    - No "sky" restart transition
    - Base cases initialize boundary gaps properly for global alignment
    - Score is LSE of final cell states only

    Args:
        scores: [B, L1, L2] similarity scores
        gap_open: gap opening penalty (typically negative)
        gap_ext: gap extension penalty (typically negative)
        temperature: softmax temperature

    Returns:
        partition: [B] alignment score
        M, I, D: [B, L1+1, L2+1] DP tables for each state
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    NINF = float('-inf')

    # Initialize DP tables
    M = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)
    I = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)
    D = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)

    # Base cases for GLOBAL alignment:
    # M(0,0) = 0, I(0,0) = D(0,0) = NINF
    M[:, 0, 0] = 0.0

    # I(i,0) = gap_open + (i-1)*gap_ext for i > 0
    # (aligning i chars of seq1 to nothing via gap in seq2)
    for i in range(1, L1 + 1):
        I[:, i, 0] = gap_open + (i - 1) * gap_ext

    # D(0,j) = gap_open + (j-1)*gap_ext for j > 0
    # (aligning j chars of seq2 to nothing via gap in seq1)
    for j in range(1, L2 + 1):
        D[:, 0, j] = gap_open + (j - 1) * gap_ext

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            s = scores[:, i-1, j-1]

            # M[i,j]: match/mismatch - come from any state at (i-1, j-1)
            # NO sky transition for global alignment
            m_from_M = M[:, i-1, j-1]
            m_from_I = I[:, i-1, j-1]
            m_from_D = D[:, i-1, j-1]

            m_candidates = torch.stack([m_from_M, m_from_I, m_from_D], dim=-1)
            M[:, i, j] = s + logsumexp_temp(m_candidates, temperature)

            # I[i,j]: gap in seq2 (insertion) - open from M/D or extend from I
            i_from_M = M[:, i-1, j] + gap_open
            i_from_I = I[:, i-1, j] + gap_ext
            i_from_D = D[:, i-1, j] + gap_open

            i_candidates = torch.stack([i_from_M, i_from_I, i_from_D], dim=-1)
            I[:, i, j] = logsumexp_temp(i_candidates, temperature)

            # D[i,j]: gap in seq1 (deletion) - open from M/I or extend from D
            d_from_M = M[:, i, j-1] + gap_open
            d_from_I = I[:, i, j-1] + gap_open
            d_from_D = D[:, i, j-1] + gap_ext

            d_candidates = torch.stack([d_from_M, d_from_I, d_from_D], dim=-1)
            D[:, i, j] = logsumexp_temp(d_candidates, temperature)

    # Global alignment: score is LSE over final cell states
    final_candidates = torch.stack([M[:, L1, L2], I[:, L1, L2], D[:, L1, L2]], dim=-1)
    partition = logsumexp_temp(final_candidates, temperature)

    return partition, M, I, D


def soft_nw_affine_naive(
    scores: torch.Tensor,
    gap_open: float,
    gap_ext: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Needleman-Wunsch (affine) returning soft alignment posteriors.

    Args:
        scores: [B, L1, L2] similarity scores
        gap_open: gap opening penalty
        gap_ext: gap extension penalty
        temperature: softmax temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (match posteriors)
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    partition, _, _, _ = soft_nw_affine_forward_naive(
        scores, gap_open, gap_ext, temperature
    )

    posteriors = torch.autograd.grad(
        partition.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft MAS (Monotonic Alignment Search)
# =============================================================================

def soft_mas_forward_naive(
    scores: torch.Tensor,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Soft Monotonic Alignment Search forward pass.

    Finds monotonic alignment from frames (T) to text (S).
    Each frame aligns to exactly one text position, must be monotonic,
    and all text must be covered.

    Args:
        scores: [B, T, S] frame-to-text similarity scores
        temperature: softmax temperature

    Returns:
        partition: [B] alignment scores
        alpha: [B, T, S] forward DP table
    """
    B, T, S = scores.shape
    NINF = -1e30

    # Alpha table
    alpha = torch.full((B, T, S), NINF, dtype=scores.dtype, device=scores.device)

    # Base case: alpha(0, 0) = score(0, 0)
    alpha[:, 0, 0] = scores[:, 0, 0]

    # Base case: alpha(t, 0) = alpha(t-1, 0) + score(t, 0) for t > 0
    # Must stay on first text token
    for t in range(1, T):
        alpha[:, t, 0] = alpha[:, t-1, 0] + scores[:, t, 0]

    # Fill DP table
    for t in range(1, T):
        for s in range(1, S):
            # Two transitions:
            # 1. Stay on same text token: alpha(t-1, s)
            # 2. Move to next text token: alpha(t-1, s-1)
            stay = alpha[:, t-1, s]
            diag = alpha[:, t-1, s-1]

            candidates = torch.stack([stay, diag], dim=-1)
            alpha[:, t, s] = scores[:, t, s] + logsumexp_temp(candidates, temperature)

    # Score: must reach (T-1, S-1) - all text covered
    partition = alpha[:, T-1, S-1]

    return partition, alpha


def soft_mas_naive(
    scores: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Soft MAS returning alignment posteriors.

    Args:
        scores: [B, T, S] frame-to-text similarity scores
        temperature: softmax temperature

    Returns:
        posteriors: [B, T, S] P(frame t aligns to text s)
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    partition, _ = soft_mas_forward_naive(scores, temperature)

    posteriors = torch.autograd.grad(
        partition.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft Eisner (Projective Dependency Parsing)
# =============================================================================

def soft_eisner_forward_naive(
    arc_scores: torch.Tensor,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Soft Eisner algorithm for projective dependency parsing.

    Uses two types of spans:
    - Complete (C): subtree with all arcs resolved
    - Incomplete (I): span with arc from boundary being formed

    Each type has two directions:
    - Right (0): head at left boundary (i)
    - Left (1): head at right boundary (j)

    Args:
        arc_scores: [B, n, n] where arc_scores[b, i, j] = score of arc i->j
        temperature: softmax temperature

    Returns:
        partition: [B] log partition function
        C: [B, n, n, 2] complete span table
        I: [B, n, n, 2] incomplete span table
    """
    B, n, _ = arc_scores.shape
    device = arc_scores.device
    dtype = arc_scores.dtype
    NINF = -1e30

    # Initialize tables
    # C[b, i, j, d]: complete span [i,j] with head at i (d=0) or j (d=1)
    # I[b, i, j, d]: incomplete span with arc i->j (d=0) or j->i (d=1)
    C = torch.full((B, n, n, 2), NINF, device=device, dtype=dtype)
    I = torch.full((B, n, n, 2), NINF, device=device, dtype=dtype)

    # Base case: single words are complete spans with score 0
    for i in range(n):
        C[:, i, i, 0] = 0.0  # Complete right
        C[:, i, i, 1] = 0.0  # Complete left

    # Process spans by increasing length
    for length in range(1, n):
        for i in range(n - length):
            j = i + length

            # === Incomplete spans ===
            # I[i,j,0] (right arc i->j): combine C[i,k,0] + C[k+1,j,1] + arc[i,j]
            # I[i,j,1] (left arc j->i): combine C[i,k,0] + C[k+1,j,1] + arc[j,i]

            candidates_I = []
            for k in range(i, j):
                val = C[:, i, k, 0] + C[:, k+1, j, 1]
                candidates_I.append(val)

            if candidates_I:
                candidates_I = torch.stack(candidates_I, dim=-1)  # [B, num_splits]
                combined = logsumexp_temp(candidates_I, temperature)  # [B]
                I[:, i, j, 0] = arc_scores[:, i, j] + combined  # i->j
                I[:, i, j, 1] = arc_scores[:, j, i] + combined  # j->i

            # === Complete spans ===
            # C[i,j,0] (complete right, head at i): combine C[i,k,0] + I[k,j,0]
            # C[i,j,1] (complete left, head at j): combine I[i,k,1] + C[k,j,1]

            # Complete right: k ranges from i to j-1
            candidates_CR = []
            for k in range(i, j):
                val = C[:, i, k, 0] + I[:, k, j, 0]
                candidates_CR.append(val)

            if candidates_CR:
                candidates_CR = torch.stack(candidates_CR, dim=-1)
                C[:, i, j, 0] = logsumexp_temp(candidates_CR, temperature)

            # Complete left: k ranges from i+1 to j
            candidates_CL = []
            for k in range(i + 1, j + 1):
                val = I[:, i, k, 1] + C[:, k, j, 1]
                candidates_CL.append(val)

            if candidates_CL:
                candidates_CL = torch.stack(candidates_CL, dim=-1)
                C[:, i, j, 1] = logsumexp_temp(candidates_CL, temperature)

    # Partition: complete span covering [0, n-1] with head at 0
    # This assumes the root is at position 0
    partition = C[:, 0, n-1, 0]

    return partition, C, I


def soft_eisner_naive(
    arc_scores: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Soft Eisner returning arc marginals.

    Args:
        arc_scores: [B, n, n] arc scores
        temperature: softmax temperature

    Returns:
        arc_marginals: [B, n, n] P(arc i->j in tree)
    """
    needs_grad = arc_scores.requires_grad
    if not needs_grad:
        arc_scores = arc_scores.detach().requires_grad_(True)

    partition, _, _ = soft_eisner_forward_naive(arc_scores, temperature)

    arc_marginals = torch.autograd.grad(
        partition.sum(), arc_scores, create_graph=True
    )[0]

    return arc_marginals


# =============================================================================
# Soft Levenshtein (Edit Distance) - GLOBAL ALIGNMENT
# =============================================================================

def soft_levenshtein_forward_naive(
    scores: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft Levenshtein (edit distance) with asymmetric costs.

    Uses softmin since we're minimizing edit distance.
    The scores tensor contains substitution costs (typically 0 for match,
    some positive value for mismatch, or can be learned).

    Args:
        scores: [B, L1, L2] substitution cost matrix (0 = match, positive = mismatch)
        ins_cost: cost of inserting a character (positive)
        del_cost: cost of deleting a character (positive)
        temperature: softmin temperature

    Returns:
        distance: [B] soft edit distance
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    PINF = 1e30  # Use large finite value for numerical stability

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), PINF, device=device, dtype=dtype)

    # Base cases for edit distance:
    # D(0,0) = 0
    # D(i,0) = i * del_cost (deleting i characters from seq1)
    # D(0,j) = j * ins_cost (inserting j characters into seq1)
    alpha[:, 0, 0] = 0.0
    for i in range(1, L1 + 1):
        alpha[:, i, 0] = i * del_cost
    for j in range(1, L2 + 1):
        alpha[:, 0, j] = j * ins_cost

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            sub_cost = scores[:, i-1, j-1]

            # Three transitions:
            # 1. Substitute: D[i-1,j-1] + sub_cost
            v_sub = alpha[:, i-1, j-1] + sub_cost
            # 2. Delete: D[i-1,j] + del_cost
            v_del = alpha[:, i-1, j] + del_cost
            # 3. Insert: D[i,j-1] + ins_cost
            v_ins = alpha[:, i, j-1] + ins_cost

            candidates = torch.stack([v_sub, v_del, v_ins], dim=-1)
            alpha[:, i, j] = softmin_temp(candidates, temperature)

    # Global alignment: distance is the final cell
    distance = alpha[:, L1, L2]

    return distance, alpha


def soft_levenshtein_naive(
    scores: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Levenshtein returning soft alignment posteriors.
    Uses autograd to compute gradients.

    For Levenshtein, since the substitution cost s_{i,j} is added inside
    the substitution option (option-additive), the gradient dD/ds_{i,j} =
    beta(i,j) * w_sub(i,j) where w_sub is the softmin weight for substitution.

    Args:
        scores: [B, L1, L2] substitution cost matrix
        ins_cost: insertion cost
        del_cost: deletion cost
        temperature: softmin temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (substitution posteriors)
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    distance, _ = soft_levenshtein_forward_naive(scores, ins_cost, del_cost, temperature)

    # Gradient of distance w.r.t. scores gives substitution posteriors
    posteriors = torch.autograd.grad(
        distance.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft LCS (Longest Common Subsequence) - GLOBAL ALIGNMENT
# =============================================================================

def soft_lcs_forward_naive(
    scores: torch.Tensor,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft LCS (Longest Common Subsequence) with SOFTMAX (maximization).

    LCS differs from Needleman-Wunsch in that:
    - No gap penalty (skips are free, score 0)
    - Match score added to diagonal transition only
    - Base cases: L(i,0) = 0, L(0,j) = 0 (not gap penalties)

    Args:
        scores: [B, L1, L2] match score matrix (1 = match, 0 = mismatch, or soft)
        temperature: softmax temperature

    Returns:
        lcs_score: [B] soft LCS length
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = scores.shape
    device = scores.device
    dtype = scores.dtype

    NINF = float('-inf')

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), NINF, device=device, dtype=dtype)

    # Base cases for LCS: all boundary cells are 0 (skips are free)
    alpha[:, 0, :] = 0.0
    alpha[:, :, 0] = 0.0

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            match_score = scores[:, i-1, j-1]

            # Three transitions:
            # 1. Match: L[i-1,j-1] + match_score
            v_match = alpha[:, i-1, j-1] + match_score
            # 2. Skip seq1: L[i-1,j] (no penalty)
            v_skip1 = alpha[:, i-1, j]
            # 3. Skip seq2: L[i,j-1] (no penalty)
            v_skip2 = alpha[:, i, j-1]

            candidates = torch.stack([v_match, v_skip1, v_skip2], dim=-1)
            alpha[:, i, j] = logsumexp_temp(candidates, temperature)

    # Global alignment: score is the final cell
    lcs_score = alpha[:, L1, L2]

    return lcs_score, alpha


def soft_lcs_naive(
    scores: torch.Tensor,
    temperature: float
) -> torch.Tensor:
    """
    Soft LCS returning soft alignment posteriors (match marginals).
    Uses autograd to compute gradients.

    For LCS, since the match score s_{i,j} is added inside the match option
    (option-additive), the gradient dL/ds_{i,j} = beta(i,j) * w_match(i,j).

    Args:
        scores: [B, L1, L2] match score matrix
        temperature: softmax temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (match posteriors)
    """
    needs_grad = scores.requires_grad
    if not needs_grad:
        scores = scores.detach().requires_grad_(True)

    lcs_score, _ = soft_lcs_forward_naive(scores, temperature)

    # Gradient of lcs_score w.r.t. scores gives match posteriors
    posteriors = torch.autograd.grad(
        lcs_score.sum(), scores, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft OSA (Optimal String Alignment / Restricted Damerau-Levenshtein)
# =============================================================================

def soft_osa_forward_naive(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft OSA (Optimal String Alignment / Restricted Damerau-Levenshtein).

    OSA extends Levenshtein with adjacent transposition, but with the restriction
    that a transposed pair cannot be edited again (hence "restricted" DL).

    Uses softmin since we're minimizing edit distance.

    Args:
        sub_costs: [B, L1, L2] substitution cost matrix (0 = match, positive = mismatch)
        trans_mask: [B, L1, L2] transposition mask (1 = valid transposition, 0 = invalid)
                    trans_mask[i,j] = 1 means s1[i-1]==s2[j] and s1[i]==s2[j-1]
        ins_cost: cost of inserting a character (positive)
        del_cost: cost of deleting a character (positive)
        trans_cost: cost of transposing adjacent characters (positive)
        temperature: softmin temperature

    Returns:
        distance: [B] soft OSA distance
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = sub_costs.shape
    device = sub_costs.device
    dtype = sub_costs.dtype

    PINF = 1e30  # Use large finite value for numerical stability

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), PINF, device=device, dtype=dtype)

    # Base cases for edit distance:
    # D(0,0) = 0
    # D(i,0) = i * del_cost (deleting i characters from seq1)
    # D(0,j) = j * ins_cost (inserting j characters into seq1)
    alpha[:, 0, 0] = 0.0
    for i in range(1, L1 + 1):
        alpha[:, i, 0] = i * del_cost
    for j in range(1, L2 + 1):
        alpha[:, 0, j] = j * ins_cost

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            sub_c = sub_costs[:, i-1, j-1]

            # Four transitions:
            # 1. Substitute: D[i-1,j-1] + sub_cost
            v_sub = alpha[:, i-1, j-1] + sub_c
            # 2. Delete: D[i-1,j] + del_cost
            v_del = alpha[:, i-1, j] + del_cost
            # 3. Insert: D[i,j-1] + ins_cost
            v_ins = alpha[:, i, j-1] + ins_cost
            # 4. Transpose: D[i-2,j-2] + trans_cost (if valid)
            if i >= 2 and j >= 2:
                trans_valid = trans_mask[:, i-1, j-1]
                v_trans = torch.where(
                    trans_valid > 0.5,
                    alpha[:, i-2, j-2] + trans_cost,
                    torch.full((B,), PINF, device=device, dtype=dtype)
                )
            else:
                v_trans = torch.full((B,), PINF, device=device, dtype=dtype)

            candidates = torch.stack([v_sub, v_del, v_ins, v_trans], dim=-1)
            alpha[:, i, j] = softmin_temp(candidates, temperature)

    # Global alignment: distance is the final cell
    distance = alpha[:, L1, L2]

    return distance, alpha


def soft_osa_naive(
    sub_costs: torch.Tensor,
    trans_mask: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft OSA returning soft alignment posteriors (substitution marginals).
    Uses autograd to compute gradients.

    Args:
        sub_costs: [B, L1, L2] substitution cost matrix
        trans_mask: [B, L1, L2] transposition validity mask
        ins_cost: insertion cost
        del_cost: deletion cost
        trans_cost: transposition cost
        temperature: softmin temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (substitution posteriors)
    """
    needs_grad = sub_costs.requires_grad
    if not needs_grad:
        sub_costs = sub_costs.detach().requires_grad_(True)

    distance, _ = soft_osa_forward_naive(sub_costs, trans_mask, ins_cost, del_cost, trans_cost, temperature)

    # Gradient of distance w.r.t. sub_costs gives substitution posteriors
    posteriors = torch.autograd.grad(
        distance.sum(), sub_costs, create_graph=True
    )[0]

    return posteriors


# =============================================================================
# Soft Damerau (True Damerau-Levenshtein with Unrestricted Transpositions)
# =============================================================================

def soft_damerau_forward_naive(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Naive soft True Damerau-Levenshtein edit distance.

    Unlike OSA which only allows adjacent transpositions, true Damerau-Levenshtein
    allows transpositions of any two characters based on their positions. This is
    implemented via precomputed trans_src indices.

    The transposition cost includes the intermediate characters:
    - trans_cost (for the transposition itself)
    - (i - k - 1) * del_cost (for characters between source and destination in seq1)
    - (j - l - 1) * ins_cost (for characters between source and destination in seq2)

    Uses softmin since we're minimizing edit distance.

    Args:
        sub_costs: [B, L1, L2] substitution cost matrix (0 = match, positive = mismatch)
        trans_src: [B, L1, L2, 2] transposition source indices (k, l)
                   trans_src[b, i-1, j-1, 0] = k, trans_src[b, i-1, j-1, 1] = l
                   If k < 0, transposition is invalid at this position
        ins_cost: cost of inserting a character (positive)
        del_cost: cost of deleting a character (positive)
        trans_cost: cost of transposing characters (positive)
        temperature: softmin temperature

    Returns:
        distance: [B] soft Damerau distance
        alpha: [B, L1+1, L2+1] DP table
    """
    B, L1, L2 = sub_costs.shape
    device = sub_costs.device
    dtype = sub_costs.dtype

    PINF = 1e30  # Use large finite value for numerical stability

    # Initialize alpha table (1-indexed, so L1+1 x L2+1)
    alpha = torch.full((B, L1 + 1, L2 + 1), PINF, device=device, dtype=dtype)

    # Base cases for edit distance:
    # D(0,0) = 0
    # D(i,0) = i * del_cost (deleting i characters from seq1)
    # D(0,j) = j * ins_cost (inserting j characters into seq1)
    alpha[:, 0, 0] = 0.0
    for i in range(1, L1 + 1):
        alpha[:, i, 0] = i * del_cost
    for j in range(1, L2 + 1):
        alpha[:, 0, j] = j * ins_cost

    # Fill DP table
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            sub_c = sub_costs[:, i-1, j-1]

            # Four transitions:
            # 1. Substitute: D[i-1,j-1] + sub_cost
            v_sub = alpha[:, i-1, j-1] + sub_c
            # 2. Delete: D[i-1,j] + del_cost
            v_del = alpha[:, i-1, j] + del_cost
            # 3. Insert: D[i,j-1] + ins_cost
            v_ins = alpha[:, i, j-1] + ins_cost

            # 4. Transpose: D[k,l] + trans_cost + (i-k-1)*del_cost + (j-l-1)*ins_cost
            # where (k, l) comes from trans_src
            trans_k = trans_src[:, i-1, j-1, 0]  # [B]
            trans_l = trans_src[:, i-1, j-1, 1]  # [B]

            v_trans = torch.full((B,), PINF, device=device, dtype=dtype)
            for b in range(B):
                k = int(trans_k[b].item())
                l = int(trans_l[b].item())
                if k >= 0 and l >= 0 and k < i and l < j:
                    extra_del = i - k - 1
                    extra_ins = j - l - 1
                    v_trans[b] = alpha[b, k, l] + trans_cost + extra_del * del_cost + extra_ins * ins_cost

            candidates = torch.stack([v_sub, v_del, v_ins, v_trans], dim=-1)
            alpha[:, i, j] = softmin_temp(candidates, temperature)

    # Global alignment: distance is the final cell
    distance = alpha[:, L1, L2]

    return distance, alpha


def soft_damerau_naive(
    sub_costs: torch.Tensor,
    trans_src: torch.Tensor,
    ins_cost: float,
    del_cost: float,
    trans_cost: float,
    temperature: float
) -> torch.Tensor:
    """
    Soft Damerau returning soft alignment posteriors (substitution marginals).
    Uses autograd to compute gradients.

    Args:
        sub_costs: [B, L1, L2] substitution cost matrix
        trans_src: [B, L1, L2, 2] transposition source indices
        ins_cost: insertion cost
        del_cost: deletion cost
        trans_cost: transposition cost
        temperature: softmin temperature

    Returns:
        posteriors: [B, L1, L2] soft alignment (substitution posteriors)
    """
    needs_grad = sub_costs.requires_grad
    if not needs_grad:
        sub_costs = sub_costs.detach().requires_grad_(True)

    distance, _ = soft_damerau_forward_naive(sub_costs, trans_src, ins_cost, del_cost, trans_cost, temperature)

    # Gradient of distance w.r.t. sub_costs gives substitution posteriors
    posteriors = torch.autograd.grad(
        distance.sum(), sub_costs, create_graph=True
    )[0]

    return posteriors
