"""NHANES cleaning: BP protocol, fasting filter, plausibility bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.hemodynamics import hemodynamic_features

BMI_MIN, BMI_MAX = 12.0, 80.0
SBP_MIN, SBP_MAX = 60.0, 260.0
DBP_MIN, DBP_MAX = 30.0, 150.0
FASTING_HOURS_MIN, FASTING_HOURS_MAX = 8.0, 24.0


def average_blood_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Discard reading 1; mean readings 2 and 3 when both present; else use the single valid reading.
    Expects BPXSY2, BPXSY3, BPXDI2, BPXDI3 (and optionally reading 1 columns).
    """
    out = df.copy()
    sy2 = out.get("BPXSY2")
    sy3 = out.get("BPXSY3")
    di2 = out.get("BPXDI2")
    di3 = out.get("BPXDI3")
    if sy2 is None or sy3 is None or di2 is None or di3 is None:
        raise KeyError("BPX columns BPXSY2/3 and BPXDI2/3 required")

    sy2 = pd.to_numeric(sy2, errors="coerce")
    sy3 = pd.to_numeric(sy3, errors="coerce")
    di2 = pd.to_numeric(di2, errors="coerce")
    di3 = pd.to_numeric(di3, errors="coerce")

    both_sy = sy2.notna() & sy3.notna()
    both_di = di2.notna() & di3.notna()
    sbp = np.where(both_sy, (sy2 + sy3) / 2.0, np.where(sy2.notna(), sy2, sy3))
    dbp = np.where(both_di, (di2 + di3) / 2.0, np.where(di2.notna(), di2, di3))
    out["sbp"] = sbp
    out["dbp"] = dbp
    map_, pp = hemodynamic_features(np.asarray(sbp, dtype=float), np.asarray(dbp, dtype=float))
    out["map"] = map_
    out["pp"] = pp
    return out


def apply_adult_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Adults >= 20; drop pregnant at exam (RIDEXPRG == 1)."""
    out = df.copy()
    out["RIDAGEYR"] = pd.to_numeric(out["RIDAGEYR"], errors="coerce")
    mask = out["RIDAGEYR"] >= 20
    if "RIDEXPRG" in out.columns:
        preg = pd.to_numeric(out["RIDEXPRG"], errors="coerce")
        mask &= preg != 1
    return out.loc[mask]


def apply_matching_coordinate_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing matching coords or outside physiological bounds."""
    out = df.copy()
    out["BMXBMI"] = pd.to_numeric(out["BMXBMI"], errors="coerce")
    for col in ("sbp", "dbp", "BMXBMI", "RIDAGEYR"):
        if col not in out.columns:
            raise KeyError(f"missing {col}")
    valid = (
        out["BMXBMI"].between(BMI_MIN, BMI_MAX)
        & out["sbp"].between(SBP_MIN, SBP_MAX)
        & out["dbp"].between(DBP_MIN, DBP_MAX)
        & out["RIDAGEYR"].notna()
    )
    return out.loc[valid]


def apply_fasting_filter(df: pd.DataFrame) -> pd.DataFrame:
    """PHAFSTHR in [8, 24) hours."""
    out = df.copy()
    hrs = pd.to_numeric(out["PHAFSTHR"], errors="coerce")
    return out.loc[hrs.notna() & (hrs >= FASTING_HOURS_MIN) & (hrs < FASTING_HOURS_MAX)]


def apply_mec_variable_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Align harmonized MEC rows with MecCohortSchema plausibility bounds."""
    out = df.copy()
    diq = pd.to_numeric(out.get("diq010"), errors="coerce")
    out["diq010"] = diq.where(diq.isin([1, 2, 3]))

    mask = out["pp"].between(10, 120)
    for col, lo, hi in (("lbxtc", 50, 600), ("lbdhdd", 10, 150), ("lbxgh", 3, 20)):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            mask &= vals.isna() | vals.between(lo, hi)
    return out.loc[mask]


def apply_fasting_variable_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Align fasting cohort rows with FastingCohortSchema plausibility bounds."""
    out = df.copy()
    glu = pd.to_numeric(out["lbxglu"], errors="coerce")
    mask = glu.between(40, 500)
    tr = pd.to_numeric(out.get("lbxtr"), errors="coerce")
    mask &= tr.isna() | tr.between(20, 2000)
    hrs = pd.to_numeric(out.get("phafsthr"), errors="coerce")
    mask &= hrs.between(FASTING_HOURS_MIN, FASTING_HOURS_MAX, inclusive="left")
    return out.loc[mask]
