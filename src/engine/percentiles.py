"""Survey-weighted Hazen ECDF percentiles (PROJECT_PLAN.md §4.3)."""

from __future__ import annotations

import numpy as np

from src.config import PERCENTILE_CEILING, PERCENTILE_FLOOR


def hazen_survey_weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    query: float,
) -> float:
    """
    Continuity-corrected weighted percentile via Hazen's rule and linear interpolation.

    Parameters
    ----------
    values:
        Neighbor analyte values (length k).
    weights:
        Positive NHANES survey weights for each neighbor (length k).
    query:
        User's analyte value y*.

    Returns
    -------
    Percentile in [PERCENTILE_FLOOR, PERCENTILE_CEILING].
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape or v.ndim != 1:
        raise ValueError("values and weights must be 1-D arrays of equal length")
    if v.size == 0:
        raise ValueError("empty neighbor set")
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be strictly positive and finite")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")

    order = np.argsort(v, kind="mergesort")
    y_sorted = v[order]
    w_sorted = w[order]
    w_total = w_sorted.sum()
    cum_w = np.cumsum(w_sorted)
    # Hazen: S_j = (sum_{m<=j} w_m - 0.5 * w_j) / sum(w)
    s = (cum_w - 0.5 * w_sorted) / w_total

    q = float(query)
    if q <= y_sorted[0]:
        pct = 100.0 * s[0]
    elif q >= y_sorted[-1]:
        pct = 100.0 * s[-1]
    else:
        idx = np.searchsorted(y_sorted, q, side="right")
        j_lo = idx - 1
        j_hi = idx
        y_lo, y_hi = y_sorted[j_lo], y_sorted[j_hi]
        s_lo, s_hi = s[j_lo], s[j_hi]
        if y_hi == y_lo:
            frac = 0.0
        else:
            frac = (q - y_lo) / (y_hi - y_lo)
        pct = 100.0 * (s_lo + frac * (s_hi - s_lo))

    return float(np.clip(pct, PERCENTILE_FLOOR, PERCENTILE_CEILING))
