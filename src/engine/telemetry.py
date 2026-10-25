"""
Survey-weighted medication telemetry and dysglycemia iceberg (PROJECT_PLAN.md §2.2, §5.1).

All functions operate on a peer neighbor DataFrame plus their survey weights,
producing weighted prevalence estimates within the matched cohort (Lens A).

Functions
---------
peer_medication_rates(peer_df, weights)
    Survey-weighted % on glucose-lowering and lipid-lowering medications.

dysglycemia_iceberg(peer_df, weights)
    Classified categories per §2.2: diagnosed / undiagnosed diabetes /
    prediabetes / normoglycemia — with survey-weighted prevalences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MedicationRates:
    """
    Survey-weighted medication prevalence among peer neighbors (§2.2 / §5.1).

    Attributes
    ----------
    on_glucose_med_pct : % currently on glucose-lowering pharmacotherapy
                         (insulin OR diabetes pills from DIQ/RXQ flags).
    on_lipid_med_pct   : % currently on statin-class lipid-lowering therapy.
    n_valid_weights    : number of peers with valid (positive) survey weights.
    """
    on_glucose_med_pct: float
    on_lipid_med_pct: float
    n_valid_weights: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "on_glucose_med_pct": self.on_glucose_med_pct,
            "on_lipid_med_pct": self.on_lipid_med_pct,
            "n_valid_weights": self.n_valid_weights,
        }


@dataclass(frozen=True)
class DysglycemiaIceberg:
    """
    Dysglycemia classification iceberg per PROJECT_PLAN.md §2.2.

    All fields are survey-weighted prevalences (%) among peer neighbors.

    Categories (mutually exclusive, exhaustive over classified participants):
    - diagnosed_pct    : DIQ010 == 1
    - undiagnosed_pct  : DIQ010 in {2, 3, NaN} AND (HbA1c >= 6.5 OR FPG >= 126)
    - prediabetes_pct  : Not diabetes AND (HbA1c in [5.7, 6.5) OR FPG in [100, 126))
    - normoglycemia_pct: No glycemic criterion met
    - borderline_diq_pct: DIQ010 == 3 prevalence (separate flag; §2.2 note)
    - unclassified_pct : Insufficient lab data to classify (for transparency)
    """
    diagnosed_pct: float
    undiagnosed_pct: float
    prediabetes_pct: float
    normoglycemia_pct: float
    borderline_diq_pct: float
    unclassified_pct: float
    n_classified: int
    n_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosed_pct": self.diagnosed_pct,
            "undiagnosed_pct": self.undiagnosed_pct,
            "prediabetes_pct": self.prediabetes_pct,
            "normoglycemia_pct": self.normoglycemia_pct,
            "borderline_diq_pct": self.borderline_diq_pct,
            "unclassified_pct": self.unclassified_pct,
            "n_classified": self.n_classified,
            "n_total": self.n_total,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, target: str) -> str | None:
    """Case-insensitive column lookup; returns actual column name or None."""
    if target in df.columns:
        return target
    t = target.lower()
    for c in df.columns:
        if c.lower() == t:
            return c
    return None


def _weighted_mean(mask: pd.Series, weights: np.ndarray) -> float:
    """Return survey-weighted proportion (%) for a boolean mask."""
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        return 0.0
    m = np.asarray(mask, dtype=bool)
    return 100.0 * float(np.sum(w[valid] * m[valid])) / float(np.sum(w[valid]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def peer_medication_rates(
    peer_df: pd.DataFrame,
    weights: np.ndarray,
) -> MedicationRates:
    """
    Compute survey-weighted medication prevalence among peer neighbors (§2.2).

    Glucose-lowering detection (any of):
    - ``on_glucose_med`` harmonized flag (True/1) if present in DataFrame
    - ``diq050`` == 1 (taking insulin)
    - ``diq070`` == 1 (taking blood sugar pills)

    Lipid-lowering detection (any of):
    - ``on_lipid_med`` harmonized flag (True/1) if present in DataFrame

    Falls back to zero prevalence gracefully when columns are absent
    (common in synthetic / unit-test data).

    Parameters
    ----------
    peer_df : DataFrame slice of matched peer neighbors
    weights : 1-D array of survey weights aligned to peer_df rows

    Returns
    -------
    MedicationRates
    """
    w = np.asarray(weights, dtype=np.float64)
    n = len(peer_df)

    # --- Glucose medication ---
    glucose_mask = pd.Series(False, index=peer_df.index)

    # Prefer harmonized flag (present in real NHANES parquet after pipeline update)
    gm_col = _find_col(peer_df, "on_glucose_med")
    if gm_col is not None:
        glucose_mask |= peer_df[gm_col].fillna(False).astype(bool)

    # Fallback: DIQ050 (insulin) and DIQ070 (diabetes pills)
    diq050_col = _find_col(peer_df, "diq050")
    if diq050_col is not None:
        glucose_mask |= pd.to_numeric(peer_df[diq050_col], errors="coerce").fillna(0) == 1

    diq070_col = _find_col(peer_df, "diq070")
    if diq070_col is not None:
        glucose_mask |= pd.to_numeric(peer_df[diq070_col], errors="coerce").fillna(0) == 1

    on_glucose_pct = _weighted_mean(glucose_mask, w)

    # --- Lipid medication ---
    lipid_mask = pd.Series(False, index=peer_df.index)
    lm_col = _find_col(peer_df, "on_lipid_med")
    if lm_col is not None:
        lipid_mask |= peer_df[lm_col].fillna(False).astype(bool)

    on_lipid_pct = _weighted_mean(lipid_mask, w)

    n_valid = int(np.sum(np.isfinite(w) & (w > 0)))
    return MedicationRates(
        on_glucose_med_pct=on_glucose_pct,
        on_lipid_med_pct=on_lipid_pct,
        n_valid_weights=n_valid,
    )


def dysglycemia_iceberg(
    peer_df: pd.DataFrame,
    weights: np.ndarray,
) -> DysglycemiaIceberg:
    """
    Survey-weighted dysglycemia iceberg classification per §2.2.

    Classification hierarchy (applied in order; once a row is classified it is
    not re-counted in a lower tier):

    1. **Diagnosed diabetes** — ``diq010 == 1``
    2. **Undiagnosed diabetes** — ``diq010 in {2, 3, NaN}``
       AND (``lbxgh >= 6.5`` OR ``lbxglu >= 126``)
    3. **Prediabetes** — not diabetes tier AND
       (``lbxgh in [5.7, 6.5)`` OR ``lbxglu in [100, 126)``)
    4. **Normoglycemia** — ``diq010 == 2`` AND all available labs below
       prediabetes thresholds
    5. **Unclassified** — ``diq010 not 1`` AND no glycemic lab present
       (cannot distinguish prediabetes from normoglycemia)

    The **borderline DIQ010** flag (``diq010 == 3``) is reported separately
    as an informational prevalence for telemetry (§2.2 note).

    Parameters
    ----------
    peer_df : DataFrame slice of matched peer neighbors
    weights : 1-D array of survey weights aligned to peer_df rows

    Returns
    -------
    DysglycemiaIceberg
    """
    w = np.asarray(weights, dtype=np.float64)
    idx = peer_df.index
    n_total = len(peer_df)

    # Resolve columns
    diq_col = _find_col(peer_df, "diq010")
    hba1c_col = _find_col(peer_df, "lbxgh")
    glu_col = _find_col(peer_df, "lbxglu")

    # Raw series (NaN-safe)
    diq = pd.to_numeric(peer_df[diq_col], errors="coerce") if diq_col else pd.Series(np.nan, index=idx)
    hba1c = pd.to_numeric(peer_df[hba1c_col], errors="coerce") if hba1c_col else pd.Series(np.nan, index=idx)
    glu = pd.to_numeric(peer_df[glu_col], errors="coerce") if glu_col else pd.Series(np.nan, index=idx)

    # Borderline flag (informational)
    borderline_mask = diq == 3

    # --- Tier 1: Diagnosed ---
    diagnosed = diq == 1

    # --- Tier 2: Undiagnosed diabetes ---
    not_diagnosed = ~diagnosed
    diabetes_lab = (hba1c >= 6.5) | (glu >= 126)
    undiagnosed = not_diagnosed & diabetes_lab

    # --- Tier 3: Prediabetes (not already diabetes) ---
    not_diabetes = ~(diagnosed | undiagnosed)
    prediabetes_lab = (
        ((hba1c >= 5.7) & (hba1c < 6.5))
        | ((glu >= 100) & (glu < 126))
    )
    prediabetes = not_diabetes & prediabetes_lab

    # --- Tier 4 / 5: Normoglycemia vs unclassified ---
    not_glycemic = ~(diagnosed | undiagnosed | prediabetes)
    has_any_lab = hba1c.notna() | glu.notna()
    normoglycemia = not_glycemic & has_any_lab
    unclassified = not_glycemic & ~has_any_lab

    n_classified = int((diagnosed | undiagnosed | prediabetes | normoglycemia).sum())

    return DysglycemiaIceberg(
        diagnosed_pct=_weighted_mean(diagnosed, w),
        undiagnosed_pct=_weighted_mean(undiagnosed, w),
        prediabetes_pct=_weighted_mean(prediabetes, w),
        normoglycemia_pct=_weighted_mean(normoglycemia, w),
        borderline_diq_pct=_weighted_mean(borderline_mask, w),
        unclassified_pct=_weighted_mean(unclassified, w),
        n_classified=n_classified,
        n_total=n_total,
    )
