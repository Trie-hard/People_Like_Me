"""MAP and pulse pressure transforms (PROJECT_PLAN.md §4.1)."""

from __future__ import annotations

import numpy as np


def mean_arterial_pressure(sbp: np.ndarray, dbp: np.ndarray) -> np.ndarray:
    """MAP = DBP + (SBP - DBP) / 3."""
    return dbp + (sbp - dbp) / 3.0


def pulse_pressure(sbp: np.ndarray, dbp: np.ndarray) -> np.ndarray:
    """PP = SBP - DBP."""
    return sbp - dbp


def hemodynamic_features(sbp: np.ndarray, dbp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (MAP, PP) with identical broadcasting rules as NumPy ufuncs."""
    map_ = mean_arterial_pressure(sbp, dbp)
    pp = pulse_pressure(sbp, dbp)
    return map_, pp
