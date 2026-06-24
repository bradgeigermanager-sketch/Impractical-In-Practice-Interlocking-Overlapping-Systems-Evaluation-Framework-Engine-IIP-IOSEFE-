"""
fractal_math_engine.py

Fractal Impracticality Analysis Math Core for FRA-01
----------------------------------------------------
Implements:
- Fractal Dimension (box-counting style)
- Self-Similarity Index (SSI)
- Boundary Roughness
- Fractal Instability Index (FII)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Callable, Tuple, List
import math
import itertools


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScalePattern:
    """
    Represents a pattern at a given scale.

    Attributes:
        scale:      Positive float, e.g. epsilon or resolution.
        features:   Arbitrary numeric feature vector describing the pattern
                    at this scale (e.g. interlock density, overlap density).
    """
    scale: float
    features: Sequence[float]


@dataclass
class FractalMetrics:
    """
    Aggregate fractal metrics for a system or subsystem.
    """
    fractal_dimension: float | None
    self_similarity_index: float | None
    boundary_roughness: float | None
    fractal_instability_index: float | None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_log(x: float) -> float:
    if x <= 0:
        raise ValueError(f"Log undefined for non-positive value: {x}")
    return math.log(x)


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# 1. Fractal Dimension (Box-Counting Style)
# ---------------------------------------------------------------------------

def fractal_dimension_box_count(
    scales: Sequence[float],
    counts: Sequence[int],
) -> float | None:
    """
    Estimate fractal dimension D^f using a box-counting-like approach:

        D^f ≈ slope of log(N(epsilon)) vs log(1/epsilon)

    Args:
        scales: sequence of epsilon-like scales (positive floats).
        counts: sequence of N(epsilon) values (positive ints).

    Returns:
        Estimated fractal dimension or None if insufficient data.
    """
    if len(scales) != len(counts):
        raise ValueError("scales and counts must have same length")
    if len(scales) < 2:
        return None

    xs: List[float] = []
    ys: List[float] = []
    for s, c in zip(scales, counts):
        if s <= 0 or c <= 0:
            continue
        xs.append(_safe_log(1.0 / s))
        ys.append(_safe_log(float(c)))

    if len(xs) < 2:
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None

    slope = num / den
    return slope


# ---------------------------------------------------------------------------
# 2. Self-Similarity Index (SSI)
# ---------------------------------------------------------------------------

def self_similarity_index(
    patterns: Sequence[ScalePattern],
    similarity_fn: Callable[[Sequence[float], Sequence[float]], float] = cosine_similarity,
) -> float | None:
    """
    Compute SSI as the average similarity between consecutive scales:

        SSI = (1/k) * sum_{i=1..k} S_i

    where S_i is similarity between pattern at scale i and i+1.

    Args:
        patterns: ordered by scale (coarse -> fine or vice versa).
        similarity_fn: function mapping (features_i, features_j) -> similarity in [−1, 1].

    Returns:
        SSI in [−1, 1] or None if insufficient data.
    """
    if len(patterns) < 2:
        return None

    sims: List[float] = []
    for p1, p2 in zip(patterns[:-1], patterns[1:]):
        sims.append(similarity_fn(p1.features, p2.features))

    if not sims:
        return None
    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# 3. Boundary Roughness
# ---------------------------------------------------------------------------

def boundary_roughness(
    boundary_lengths: Sequence[float],
    reference_length: float | None = None,
) -> float | None:
    """
    Simple boundary roughness estimator.

    Intuition:
        - boundary_lengths: measured boundary length at different resolutions.
        - reference_length: optional baseline (e.g. smooth boundary length).

    A basic roughness measure is the coefficient of variation (CV) of
    normalized lengths.

    Returns:
        Roughness >= 0 or None if insufficient data.
    """
    if len(boundary_lengths) < 2:
        return None

    if reference_length is None:
        # Use mean as implicit reference
        reference_length = sum(boundary_lengths) / len(boundary_lengths)

    if reference_length <= 0:
        return None

    normalized = [b / reference_length for b in boundary_lengths if b > 0]
    if len(normalized) < 2:
        return None

    mean = sum(normalized) / len(normalized)
    var = sum((x - mean) ** 2 for x in normalized) / (len(normalized) - 1)
    std = math.sqrt(var)
    if mean == 0:
        return None

    cv = std / mean
    return cv


# ---------------------------------------------------------------------------
# 4. Fractal Instability Index (FII)
# ---------------------------------------------------------------------------

def fractal_instability_index(
    d_f: float | None,
    ssi: float | None,
    b_r: float | None,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> float | None:
    """
    Composite Fractal Instability Index:

        FII = α * D^f + β * SSI + γ * B_r

    Any missing component is treated as 0 contribution.

    Args:
        d_f: fractal dimension.
        ssi: self-similarity index.
        b_r: boundary roughness.
        alpha, beta, gamma: weights.

    Returns:
        FII (unbounded real) or None if all components are None.
    """
    if d_f is None and ssi is None and b_r is None:
        return None

    d_f_val = d_f if d_f is not None else 0.0
    ssi_val = ssi if ssi is not None else 0.0
    b_r_val = b_r if b_r is not None else 0.0

    return alpha * d_f_val + beta * ssi_val + gamma * b_r_val


# ---------------------------------------------------------------------------
# 5. High-level convenience wrapper
# ---------------------------------------------------------------------------

def compute_fractal_metrics(
    scales: Sequence[float],
    counts: Sequence[int],
    patterns: Sequence[ScalePattern],
    boundary_lengths: Sequence[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> FractalMetrics:
    """
    Compute all core fractal metrics in one call.

    Args:
        scales: scales for box-counting.
        counts: counts for box-counting.
        patterns: scale patterns for SSI.
        boundary_lengths: boundary lengths for roughness.
        alpha, beta, gamma: weights for FII.

    Returns:
        FractalMetrics object.
    """
    d_f = fractal_dimension_box_count(scales, counts)
    ssi = self_similarity_index(patterns)
    b_r = boundary_roughness(boundary_lengths)
    fii = fractal_instability_index(d_f, ssi, b_r, alpha, beta, gamma)
    return FractalMetrics(
        fractal_dimension=d_f,
        self_similarity_index=ssi,
        boundary_roughness=b_r,
        fractal_instability_index=fii,
    )


# ---------------------------------------------------------------------------
# Example usage (can be removed in production)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example synthetic data
    scales = [1.0, 0.5, 0.25, 0.125]
    counts = [10, 20, 40, 80]  # perfect doubling -> D^f ≈ 1

    patterns = [
        ScalePattern(scale=1.0,   features=[1.0, 0.2, 0.1]),
        ScalePattern(scale=0.5,   features=[0.9, 0.25, 0.15]),
        ScalePattern(scale=0.25,  features=[0.88, 0.3, 0.2]),
        ScalePattern(scale=0.125, features=[0.85, 0.32, 0.22]),
    ]

    boundary_lengths = [10.0, 11.5, 13.0, 14.2]

    metrics = compute_fractal_metrics(
        scales=scales,
        counts=counts,
        patterns=patterns,
        boundary_lengths=boundary_lengths,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    print("Fractal Dimension (D^f):", metrics.fractal_dimension)
    print("Self-Similarity Index (SSI):", metrics.self_similarity_index)
    print("Boundary Roughness (B_r):", metrics.boundary_roughness)
    print("Fractal Instability Index (FII):", metrics.fractal_instability_index)
