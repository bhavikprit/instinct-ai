"""
Calibration metrics and loss functions for System 1 Decision Models.
Zero external dependencies for core evaluation, with optional PyTorch hooks.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Union


def brier_score(predictions: List[float], targets: List[int]) -> float:
    """
    Computes the standard Brier Score: Mean Squared Error of probability forecasts.
    Range: [0.0, 1.0]. Lower is better. 0.0 is perfect calibration.
    """
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be non-empty and of equal length.")

    total_error = sum((p - y) ** 2 for p, y in zip(predictions, targets))
    return total_error / len(predictions)


def expected_calibration_error(
    predictions: List[float],
    targets: List[int],
    num_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across uniform confidence bins.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    Range: [0.0, 1.0]. Lower is better.
    """
    if not predictions or not targets or len(predictions) != len(targets):
        raise ValueError("Predictions and targets must be non-empty and of equal length.")

    n = len(predictions)
    bins = [[] for _ in range(num_bins)]

    for p, y in zip(predictions, targets):
        # Clip to [0, 1]
        p_clipped = max(0.0, min(1.0, float(p)))
        bin_idx = min(int(p_clipped * num_bins), num_bins - 1)
        bins[bin_idx].append((p_clipped, y))

    ece = 0.0
    for b in bins:
        if not b:
            continue
        bin_size = len(b)
        avg_conf = sum(p for p, _ in b) / bin_size
        avg_acc = sum(y for _, y in b) / bin_size
        ece += (bin_size / n) * abs(avg_acc - avg_conf)

    return ece


def epistemic_entropy(p: float) -> float:
    """Computes binary entropy H(p) measuring epistemic uncertainty in nats."""
    p_clamped = max(1e-7, min(1.0 - 1e-7, p))
    return -(p_clamped * math.log(p_clamped) + (1.0 - p_clamped) * math.log(1.0 - p_clamped))


def reliability_diagram_data(
    predictions: List[float],
    targets: List[int],
    num_bins: int = 10,
) -> List[Tuple[float, float, int]]:
    """
    Returns (avg_confidence, accuracy, bin_count) tuples for plotting reliability diagrams.
    """
    bins = [[] for _ in range(num_bins)]
    for p, y in zip(predictions, targets):
        p_clipped = max(0.0, min(1.0, float(p)))
        bin_idx = min(int(p_clipped * num_bins), num_bins - 1)
        bins[bin_idx].append((p_clipped, y))

    results = []
    for b in bins:
        if not b:
            continue
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc = sum(y for _, y in b) / len(b)
        results.append((round(avg_conf, 4), round(avg_acc, 4), len(b)))
    return results
