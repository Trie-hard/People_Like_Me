"""Leave-one-out weighted prediction-interval coverage (PROJECT_PLAN.md §6.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import K_NEIGHBORS_DEFAULT
from src.engine.matcher import CohortMatcher, UserProfile, weighted_quantiles
from src.engine.benchmark_cohort import get_age_decade_bounds
from src.validation.winkler_score import mean_winkler


@dataclass(frozen=True)
class CoverageResult:
    alpha: float
    nominal: float
    weighted_coverage: float
    unweighted_coverage: float
    n_evaluated: int


@dataclass(frozen=True)
class ModelEvaluation:
    model_name: str
    coverages: tuple[CoverageResult, ...]
    mean_winkler_90: float


def _central_interval(
    values: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, float]:
    tail = (1.0 - alpha) / 2.0
    lo = weighted_quantiles(values, weights, (tail,))[0]
    hi = weighted_quantiles(values, weights, (1.0 - tail,))[0]
    return float(lo), float(hi)


def _row_profile(row: pd.Series) -> UserProfile:
    return UserProfile(
        age=float(row["ridageyr"]),
        sex=int(row["riagendr"]),
        bmi=float(row["bmxbmi"]),
        sbp=float(row["sbp"]),
        dbp=float(row["dbp"]),
    )


def _baseline0_interval(df: pd.DataFrame, analyte: str, weight_col: str, alpha: float) -> tuple[float, float]:
    y = df[analyte].to_numpy(dtype=float)
    w = df[weight_col].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    return _central_interval(y[mask], w[mask], alpha)


def _baseline1_interval(
    df: pd.DataFrame,
    row: pd.Series,
    analyte: str,
    weight_col: str,
    alpha: float,
) -> tuple[float, float]:
    age_min, age_max, _ = get_age_decade_bounds(float(row["ridageyr"]))
    sub = df.loc[
        (df["riagendr"] == row["riagendr"])
        & (df["ridageyr"] >= age_min)
        & (df["ridageyr"] <= age_max)
    ]
    y = sub[analyte].to_numpy(dtype=float)
    w = sub[weight_col].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 10:
        return _baseline0_interval(df, analyte, weight_col, alpha)
    return _central_interval(y[mask], w[mask], alpha)


def _knn_interval(
    df_loo: pd.DataFrame,
    row: pd.Series,
    analyte: str,
    weight_col: str,
    k: int,
    alpha: float,
) -> tuple[float, float]:
    matcher = CohortMatcher(df_loo, weight_col=weight_col)
    user = _row_profile(row)
    sex_n = int((df_loo["riagendr"] == row["riagendr"]).sum())
    k_eff = min(k, sex_n - 1, len(df_loo) - 1)
    if k_eff < 5:
        return _baseline0_interval(df_loo, analyte, weight_col, alpha)
    match = matcher.query(user, k=k_eff)
    neigh = match.neighbor_df
    y = neigh[analyte].to_numpy(dtype=float)
    w = match.weights
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return _baseline0_interval(df_loo, analyte, weight_col, alpha)
    return _central_interval(y[mask], w[mask], alpha)


def evaluate_model_loo(
    df: pd.DataFrame,
    *,
    analyte: str = "lbxgh",
    weight_col: str = "wtmec6yr",
    model: str = "knn",
    k: int = K_NEIGHBORS_DEFAULT,
    alphas: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    max_units: int | None = 400,
    seed: int = 42,
) -> ModelEvaluation:
    """
    Survey-weighted LOO coverage for k-NN (Lens A neighbors) or baselines 0/1.
    """
    work = df.dropna(subset=[analyte, weight_col, "ridageyr", "riagendr", "bmxbmi", "sbp", "dbp"]).copy()
    if max_units is not None and len(work) > max_units:
        work = work.sample(n=max_units, random_state=seed, weights=work[weight_col])

    lowers_90: list[float] = []
    uppers_90: list[float] = []
    observed: list[float] = []
    weights_obs: list[float] = []

    coverages: list[CoverageResult] = []
    for alpha in alphas:
        w_hit = 0.0
        w_total = 0.0
        hits = 0
        n = 0
        for idx, row in work.iterrows():
            y_i = float(row[analyte])
            w_i = float(row[weight_col])
            df_loo = work.drop(index=idx)
            if model == "knn":
                lo, hi = _knn_interval(df_loo, row, analyte, weight_col, k, alpha)
            elif model == "baseline0":
                lo, hi = _baseline0_interval(df_loo, analyte, weight_col, alpha)
            elif model == "baseline1":
                lo, hi = _baseline1_interval(df_loo, row, analyte, weight_col, alpha)
            else:
                raise ValueError(f"unknown model {model}")
            inside = lo <= y_i <= hi
            w_hit += w_i * int(inside)
            w_total += w_i
            hits += int(inside)
            n += 1
            if alpha == 0.9:
                lowers_90.append(lo)
                uppers_90.append(hi)
                observed.append(y_i)
                weights_obs.append(w_i)
        coverages.append(
            CoverageResult(
                alpha=alpha,
                nominal=alpha,
                weighted_coverage=w_hit / w_total if w_total > 0 else 0.0,
                unweighted_coverage=hits / n if n else 0.0,
                n_evaluated=n,
            )
        )

    winkler = mean_winkler(
        np.asarray(lowers_90),
        np.asarray(uppers_90),
        np.asarray(observed),
        0.9,
    )
    return ModelEvaluation(model_name=model, coverages=tuple(coverages), mean_winkler_90=winkler)
