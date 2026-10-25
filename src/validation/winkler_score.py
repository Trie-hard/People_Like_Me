"""Winkler interval sharpness score (PROJECT_PLAN.md §6.2)."""

from __future__ import annotations

import numpy as np


def winkler_score(
    lower: float,
    upper: float,
    observed: float,
    alpha: float,
) -> float:
    """
    Score_alpha(L, U, y) = (U - L) + (2/alpha)(L - y) if y < L + (2/alpha)(y - U) if y > U.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    if lower > upper:
        raise ValueError("lower must be <= upper")
    width = upper - lower
    penalty = 0.0
    if observed < lower:
        penalty = (2.0 / alpha) * (lower - observed)
    elif observed > upper:
        penalty = (2.0 / alpha) * (observed - upper)
    return float(width + penalty)


def mean_winkler(
    lowers: np.ndarray,
    uppers: np.ndarray,
    observed: np.ndarray,
    alpha: float,
) -> float:
    scores = [
        winkler_score(float(lo), float(hi), float(y), alpha)
        for lo, hi, y in zip(lowers, uppers, observed, strict=True)
    ]
    return float(np.mean(scores))
