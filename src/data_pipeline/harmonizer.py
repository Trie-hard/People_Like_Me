"""Merge NHANES cycles into pooled MEC and fasting Parquet tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NHANES_CYCLES, POOL_CYCLE_DIVISOR
from src.data_pipeline.cleaner import (
    apply_adult_filters,
    apply_fasting_filter,
    apply_matching_coordinate_filters,
    apply_fasting_variable_bounds,
    apply_mec_variable_bounds,
    average_blood_pressure,
)
from src.data_pipeline.contracts import FastingCohortSchema, MecCohortSchema

_RENAME = {
    "SEQN": "seqn",
    "RIDAGEYR": "ridageyr",
    "RIAGENDR": "riagendr",
    "RIDRETH3": "ridreth3",
    "INDFMPIR": "indfmpir",
    "SDMVSTRA": "sdmvstra",
    "SDMVPSU": "sdmvpsu",
    "BMXBMI": "bmxbmi",
    "LBXGH": "lbxgh",
    "LBXTC": "lbxtc",
    "LBDHDD": "lbdhdd",
    "DIQ010": "diq010",
    "LBXGLU": "lbxglu",
    "LBXTR": "lbxtr",
    "PHAFSTHR": "phafsthr",
}

# ---------------------------------------------------------------------------
# RXQ medication classification
# ---------------------------------------------------------------------------
# ATC/ingredient keywords for statin-class lipid-lowering agents (case-insensitive
# substring match on RXDDRUG / generic drug name field in RXQ_RX).
_STATIN_KEYWORDS: frozenset[str] = frozenset({
    "lovastatin", "simvastatin", "pravastatin", "fluvastatin",
    "atorvastatin", "rosuvastatin", "pitavastatin", "cerivastatin",
})


def _build_rxq_flags(
    raw_dir: Path,
    cycle_key: str,
    seqns: pd.Series,
    diq_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Parse RXQ_RX_{H,I,J}.XPT and return per-participant medication flags.

    Returns a DataFrame indexed on seqn with two boolean columns:
    - on_glucose_med : insulin or diabetes-pill prescription in survey period
                       (from RXQ or self-reported in DIQ050/DIQ070)
    - on_lipid_med   : statin-class prescription in survey period

    If the RXQ file is absent (e.g. during unit tests), returns a DataFrame of
    False values (or DIQ-derived flags if present) for all seqns so callers can
    unconditionally merge.
    """
    letter = _suffix_letter(cycle_key)
    rxq_path = raw_dir / f"RXQ_RX_{letter}.XPT"

    glucose_seqns: set[int] = set()
    lipid_seqns: set[int] = set()

    # DIQ050 (insulin) / DIQ070 (pills)
    if diq_df is not None:
        col_map = {c.upper(): c for c in diq_df.columns}
        seqn_c = col_map.get("SEQN", "seqn")
        mask_diq = pd.Series(False, index=diq_df.index)
        for col_name in ("DIQ050", "DIQ070"):
            act_col = col_map.get(col_name)
            if act_col:
                mask_diq |= pd.to_numeric(diq_df[act_col], errors="coerce") == 1
        if mask_diq.any():
            glucose_seqns.update(diq_df.loc[mask_diq, seqn_c].dropna().astype(int).tolist())

    if rxq_path.exists():
        rxq = _read_xpt(rxq_path)
        rxq.columns = [c.upper() for c in rxq.columns]

        seqn_col = "SEQN"
        name_col = "RXDDRUG" if "RXDDRUG" in rxq.columns else None
        cls1_col = "RXDRSC1" if "RXDRSC1" in rxq.columns else None

        # --- Glucose-lowering flag (drug class 87 = antidiabetics in RXDRSC1/2/3) ---
        glucose_class_codes = {87}
        if cls1_col:
            cls_cols = [c for c in rxq.columns if c.startswith("RXDRSC")]
            for col in cls_cols:
                vals = pd.to_numeric(rxq[col], errors="coerce")
                mask = vals.isin(glucose_class_codes)
                glucose_seqns.update(rxq.loc[mask, seqn_col].dropna().astype(int).tolist())

        # Fallback: name-based
        _GLUCOSE_KEYWORDS = frozenset({
            "insulin", "metformin", "glipizide", "glyburide", "glimepiride",
            "sitagliptin", "saxagliptin", "alogliptin", "linagliptin",
            "empagliflozin", "canagliflozin", "dapagliflozin",
            "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
            "exenatide", "liraglutide", "semaglutide", "dulaglutide",
            "nateglinide", "repaglinide",
        })
        if name_col:
            names = rxq[name_col].astype(str).str.lower()
            for kw in _GLUCOSE_KEYWORDS:
                mask = names.str.contains(kw, na=False, regex=False)
                glucose_seqns.update(rxq.loc[mask, seqn_col].dropna().astype(int).tolist())

        # --- Lipid-lowering flag (statins) ---
        lipid_class_codes = {19, 36}
        if cls1_col:
            cls_cols = [c for c in rxq.columns if c.startswith("RXDRSC")]
            for col in cls_cols:
                vals = pd.to_numeric(rxq[col], errors="coerce")
                mask = vals.isin(lipid_class_codes)
                lipid_seqns.update(rxq.loc[mask, seqn_col].dropna().astype(int).tolist())

        if name_col:
            names = rxq[name_col].astype(str).str.lower()
            for kw in _STATIN_KEYWORDS:
                mask = names.str.contains(kw, na=False, regex=False)
                lipid_seqns.update(rxq.loc[mask, seqn_col].dropna().astype(int).tolist())

    # Build per-participant flag DataFrame (one row per SEQN)
    all_seqns = seqns.astype(int)
    result = pd.DataFrame({
        "seqn": all_seqns,
        "on_glucose_med": all_seqns.isin(glucose_seqns),
        "on_lipid_med": all_seqns.isin(lipid_seqns),
    })
    return result.drop_duplicates("seqn").reset_index(drop=True)


def _read_xpt(path: Path) -> pd.DataFrame:
    try:
        return pd.read_sas(path, format="xport", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_sas(path, format="xport", encoding="latin1")


def _suffix_letter(cycle_key: str) -> str:
    return {"2013_2014": "H", "2015_2016": "I", "2017_2018": "J"}[cycle_key]


def build_cycle_mec(raw_dir: Path, cycle_key: str) -> pd.DataFrame:
    letter = _suffix_letter(cycle_key)
    demo = _read_xpt(raw_dir / f"DEMO_{letter}.XPT")
    bmx = _read_xpt(raw_dir / f"BMX_{letter}.XPT")
    bpx = _read_xpt(raw_dir / f"BPX_{letter}.XPT")
    ghb = _read_xpt(raw_dir / f"GHB_{letter}.XPT")[["SEQN", "LBXGH"]]
    tchol = _read_xpt(raw_dir / f"TCHOL_{letter}.XPT")[["SEQN", "LBXTC"]]
    hdl = _read_xpt(raw_dir / f"HDL_{letter}.XPT")[["SEQN", "LBDHDD"]]
    
    diq_raw = _read_xpt(raw_dir / f"DIQ_{letter}.XPT")
    diq_cols = ["SEQN", "DIQ010"]
    for c in ("DIQ050", "DIQ070"):
        if c in diq_raw.columns:
            diq_cols.append(c)
    diq = diq_raw[diq_cols]

    df = demo.merge(bmx, on="SEQN", how="inner")
    df = df.merge(bpx, on="SEQN", how="inner")
    for extra in (ghb, tchol, hdl, diq):
        df = df.merge(extra, on="SEQN", how="left")

    df = apply_adult_filters(df)
    df = average_blood_pressure(df)
    df = apply_matching_coordinate_filters(df)

    df["WTMEC6YR"] = pd.to_numeric(df["WTMEC2YR"], errors="coerce") / POOL_CYCLE_DIVISOR
    df = df.loc[df["WTMEC6YR"] > 0]
    df["cycle"] = cycle_key
    df = df.rename(columns=_RENAME)
    df = df.rename(columns={"WTMEC6YR": "wtmec6yr"})
    df = apply_mec_variable_bounds(df)

    # Medication flags from RXQ and DIQ
    rxq_flags = _build_rxq_flags(raw_dir, cycle_key, df["seqn"], diq_df=df)
    df = df.merge(rxq_flags, on="seqn", how="left")
    df["on_glucose_med"] = df["on_glucose_med"].fillna(False).astype(bool)
    df["on_lipid_med"] = df["on_lipid_med"].fillna(False).astype(bool)

    cols = [
        "seqn",
        "cycle",
        "ridageyr",
        "riagendr",
        "ridreth3",
        "indfmpir",
        "sdmvstra",
        "sdmvpsu",
        "bmxbmi",
        "sbp",
        "dbp",
        "map",
        "pp",
        "lbxgh",
        "lbxtc",
        "lbdhdd",
        "diq010",
        "wtmec6yr",
        "on_glucose_med",
        "on_lipid_med",
    ]
    return df[cols]


def build_cycle_fasting(mec: pd.DataFrame, raw_dir: Path, cycle_key: str) -> pd.DataFrame:
    letter = _suffix_letter(cycle_key)
    glu = _read_xpt(raw_dir / f"GLU_{letter}.XPT")
    trig = _read_xpt(raw_dir / f"TRIGLY_{letter}.XPT")[["SEQN", "LBXTR"]]
    glu = glu.merge(trig, on="SEQN", how="left")
    if "PHAFSTHR" not in glu.columns:
        fastqx_path = raw_dir / f"FASTQX_{letter}.XPT"
        if not fastqx_path.exists():
            raise FileNotFoundError(
                f"{fastqx_path.name} required for fasting hours (PHAFSTHR) in cycle {cycle_key}"
            )
        fastqx = _read_xpt(fastqx_path)[["SEQN", "PHAFSTHR"]]
        glu = glu.merge(fastqx, on="SEQN", how="left")
    glu["WTSAF6YR"] = pd.to_numeric(glu["WTSAF2YR"], errors="coerce") / POOL_CYCLE_DIVISOR
    glu = apply_fasting_filter(glu)
    glu = glu.rename(columns=_RENAME)
    glu = glu.rename(columns={"WTSAF6YR": "wtsaf6yr"})
    merged = mec.merge(
        glu[["seqn", "lbxglu", "lbxtr", "wtsaf6yr", "phafsthr"]],
        on="seqn",
        how="inner",
    )
    merged = merged.loc[merged["wtsaf6yr"] > 0]
    return apply_fasting_variable_bounds(merged)


def pool_cycles(raw_root: Path | str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = Path(raw_root)
    mec_frames: list[pd.DataFrame] = []
    fast_frames: list[pd.DataFrame] = []
    for cycle in NHANES_CYCLES:
        cdir = raw / cycle
        mec = build_cycle_mec(cdir, cycle)
        mec_frames.append(mec)
        fast_frames.append(build_cycle_fasting(mec, cdir, cycle))
    mec_pooled = pd.concat(mec_frames, ignore_index=True)
    fast_pooled = pd.concat(fast_frames, ignore_index=True)
    return mec_pooled, fast_pooled


def export_processed(
    processed_root: Path | str = "data/processed",
    raw_root: Path | str = "data/raw",
) -> tuple[Path, Path]:
    mec, fasting = pool_cycles(raw_root)
    MecCohortSchema.validate(mec, lazy=True)
    FastingCohortSchema.validate(fasting, lazy=True)
    out = Path(processed_root)
    out.mkdir(parents=True, exist_ok=True)
    mec_path = out / "mec_cohort_pooled.parquet"
    fast_path = out / "fasting_cohort_pooled.parquet"
    mec.to_parquet(mec_path, index=False)
    fasting.to_parquet(fast_path, index=False)
    return mec_path, fast_path
