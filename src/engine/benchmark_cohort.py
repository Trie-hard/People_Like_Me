"""
Optimal health reference cohort generator — Lens B (PROJECT_PLAN.md §2.1).

Defines the target manifold of metabolically healthy adults:
- Exact biological sex match
- Age decade match (with 80+ top-coded band [80, 80])
- Normal BMI: [18.5, 25.0) kg/m^2
- Normotensive: SBP < 120 mmHg and DBP < 80 mmHg
- Non-diabetic: DIQ010 == 2 (no prior diagnosis; excludes borderline 3),
  and normoglycemic on available labs (HbA1c < 5.7%, FPG < 100 mg/dL)
- Weighted n check vs LENS_B_MIN_WEIGHTED_N (default 200)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    LENS_B_BMI_MAX,
    LENS_B_BMI_MIN,
    LENS_B_DBP_MAX,
    LENS_B_FPG_MAX,
    LENS_B_HBA1C_MAX,
    LENS_B_MIN_WEIGHTED_N,
    LENS_B_SBP_MAX,
)
from src.engine.matcher import normalize_sex, resolve_weight_column


# ---------------------------------------------------------------------------
# Data structures & helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensBResult:
    """Outcome of optimal health benchmark cohort extraction (§2.1)."""

    mask: pd.Series
    benchmark_df: pd.DataFrame
    weights: np.ndarray
    unweighted_n: int
    effective_n: float
    is_sufficient: bool
    age_decade_bounds: tuple[int, int]
    is_top_coded_age: bool
    status_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unweighted_n": self.unweighted_n,
            "effective_n": self.effective_n,
            "is_sufficient": self.is_sufficient,
            "age_decade_bounds": self.age_decade_bounds,
            "is_top_coded_age": self.is_top_coded_age,
            "status_message": self.status_message,
        }


def get_age_decade_bounds(age: float) -> tuple[int, int, bool]:
    """
    Compute age decade matching bounds per §2.1.

    Rules:
    - User age 45 -> [40, 49], top-coded=False
    - User age >= 80 -> [80, 80] (NHANES public files top-code at 80), top-coded=True

    Returns
    -------
    (age_min, age_max, is_top_coded)
    """
    age_f = float(age)
    if age_f >= 80.0:
        return 80, 80, True
    start = int(np.floor(age_f / 10.0)) * 10
    end = start + 9
    return start, end, False


def compute_effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute Kish's design-effect-adjusted effective sample size:
        n_eff = (sum(w))^2 / sum(w^2)

    Scale-invariant: returns n_eff == N for uniform weights.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    sum_w = float(np.sum(w))
    sum_w_sq = float(np.sum(w ** 2))
    if sum_w_sq <= 0:
        return 0.0
    return float((sum_w ** 2) / sum_w_sq)


def _find_col(df: pd.DataFrame, target: str) -> str | None:
    """Case-insensitive column lookup."""
    if target in df.columns:
        return target
    target_lower = target.lower()
    for c in df.columns:
        if c.lower() == target_lower:
            return c
    return None


# ---------------------------------------------------------------------------
# Core Lens B mask & subset
# ---------------------------------------------------------------------------

def lens_b_mask(
    df: pd.DataFrame,
    age: float,
    sex: int | str,
    analyte: str | None = None,
) -> pd.Series:
    """
    Generate boolean eligibility mask for the Lens B optimal benchmark pool (§2.1).

    Criteria:
    1. Sex: exact match on RIAGENDR.
    2. Age: same age decade [10*floor(age/10), 10*floor(age/10)+9]; if age >= 80, [80, 80].
    3. BMI: 18.5 <= BMXBMI < 25.0 kg/m^2.
    4. BP: measured SBP < 120 mmHg and DBP < 80 mmHg.
    5. Non-diabetic:
       - DIQ010 == 2 (no prior diagnosis; borderline 3 and diagnosed 1 excluded).
       - If LBXGH present in df: LBXGH < 5.7%.
       - If LBXGLU present in df: LBXGLU < 100 mg/dL.
    6. If analyte specified (e.g. 'lbxgh' or 'lbxglu'):
       - Participant must have a valid non-null value for that analyte.

    Parameters
    ----------
    df : NHANES cohort DataFrame (MEC or Fasting)
    age : user age in years
    sex : user biological sex (1/M or 2/F)
    analyte : optional analyte column name (e.g. 'lbxgh', 'lbxglu') to enforce presence

    Returns
    -------
    pd.Series of boolean flags indexed identically to df.
    """
    sex_val = normalize_sex(sex)
    age_min, age_max, _ = get_age_decade_bounds(age)

    # Resolve required column names case-insensitively
    sex_col = _find_col(df, "riagendr")
    age_col = _find_col(df, "ridageyr")
    bmi_col = _find_col(df, "bmxbmi")
    sbp_col = _find_col(df, "sbp")
    dbp_col = _find_col(df, "dbp")
    diq_col = _find_col(df, "diq010")

    missing = [
        name
        for name, col in [
            ("riagendr", sex_col),
            ("ridageyr", age_col),
            ("bmxbmi", bmi_col),
            ("sbp", sbp_col),
            ("dbp", dbp_col),
            ("diq010", diq_col),
        ]
        if col is None
    ]
    if missing:
        raise KeyError(f"Required column(s) for Lens B missing from DataFrame: {missing}")

    # 1. Sex
    mask = df[sex_col] == sex_val

    # 2. Age decade
    mask &= df[age_col].between(age_min, age_max)

    # 3. BMI: [18.5, 25.0)
    bmi = df[bmi_col]
    mask &= (bmi >= LENS_B_BMI_MIN) & (bmi < LENS_B_BMI_MAX)

    # 4. Blood pressure: SBP < 120 and DBP < 80
    sbp = df[sbp_col]
    dbp = df[dbp_col]
    mask &= (sbp < LENS_B_SBP_MAX) & (dbp < LENS_B_DBP_MAX)

    # 5. Non-diabetic survey: DIQ010 == 2
    diq = df[diq_col]
    mask &= diq == 2

    # 5b. Non-diabetic lab: HbA1c < 5.7% (if present in row)
    hba1c_col = _find_col(df, "lbxgh")
    if hba1c_col is not None:
        hba1c = df[hba1c_col]
        # Must not be in prediabetes / diabetes range (>= 5.7)
        mask &= hba1c.isna() | (hba1c < LENS_B_HBA1C_MAX)

    # 5c. Non-diabetic lab: Fasting glucose < 100 mg/dL (if present in row)
    glu_col = _find_col(df, "lbxglu")
    if glu_col is not None:
        glu = df[glu_col]
        mask &= glu.isna() | (glu < LENS_B_FPG_MAX)

    # 6. Specific analyte requirement if provided
    if analyte is not None:
        target_col = _find_col(df, analyte)
        if target_col is None:
            raise KeyError(f"Requested analyte '{analyte}' not found in DataFrame columns")
        vals = df[target_col]
        mask &= vals.notna()
        # Enforce clinical threshold for known glycemic analytes
        target_lower = target_col.lower()
        if target_lower in ("lbxgh", "ghb", "hba1c"):
            mask &= vals < LENS_B_HBA1C_MAX
        elif target_lower in ("lbxglu", "glu", "glucose", "fpg"):
            mask &= vals < LENS_B_FPG_MAX

    return mask


def lens_b_subset(
    df: pd.DataFrame,
    age: float,
    sex: int | str,
    analyte: str | None = None,
    weight_col: str | None = None,
    pool: str = "mec",
    min_weighted_n: float = LENS_B_MIN_WEIGHTED_N,
) -> LensBResult:
    """
    Extract the Lens B benchmark cohort and check sample size adequacy (§2.1).

    Parameters
    ----------
    df : NHANES cohort DataFrame
    age : user age in years
    sex : biological sex (1/M or 2/F)
    analyte : optional target analyte (e.g. 'lbxgh')
    weight_col : survey weight column name (if None, auto-detected)
    pool : 'mec' or 'fasting'
    min_weighted_n : threshold for adequacy check (default LENS_B_MIN_WEIGHTED_N = 200)

    Returns
    -------
    LensBResult with benchmark DataFrame, weights, effective sample size,
    sufficiency flag, and top-coding metadata.
    """
    mask = lens_b_mask(df, age=age, sex=sex, analyte=analyte)
    benchmark_df = df.loc[mask].copy()

    w_col = resolve_weight_column(df, weight_col=weight_col, pool=pool)
    weights = benchmark_df[w_col].to_numpy(dtype=np.float64)

    unweighted_n = len(benchmark_df)
    effective_n = compute_effective_sample_size(weights)
    is_sufficient = effective_n >= min_weighted_n

    age_min, age_max, is_top_coded = get_age_decade_bounds(age)

    if is_sufficient:
        status_message = "Sufficient benchmark sample"
    else:
        status_message = (
            f"Insufficient benchmark sample (effective weighted n={effective_n:.1f} < {min_weighted_n})"
        )

    return LensBResult(
        mask=mask,
        benchmark_df=benchmark_df,
        weights=weights,
        unweighted_n=unweighted_n,
        effective_n=effective_n,
        is_sufficient=is_sufficient,
        age_decade_bounds=(age_min, age_max),
        is_top_coded_age=is_top_coded,
        status_message=status_message,
    )
