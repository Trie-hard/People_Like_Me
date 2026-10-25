"""Subgroup coverage parity audit (PROJECT_PLAN.md §6.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import FAIRNESS_COVERAGE_GAP_THRESHOLD_PP, FAIRNESS_MIN_WEIGHTED_N
from src.validation.coverage_evaluator import evaluate_model_loo


@dataclass(frozen=True)
class SubgroupCoverage:
    label: str
    weighted_coverage_90: float
    effective_n: float
    n_units: int
    suppressed: bool


@dataclass(frozen=True)
class FairnessReport:
    groups: tuple[SubgroupCoverage, ...]
    max_gap_pp: float
    flagged: bool
    notes: str


def _age_bracket(age: float) -> str:
    if age < 40:
        return "20-39"
    if age < 60:
        return "40-59"
    if age < 80:
        return "60-79"
    return "80+"


def audit_age_bracket_coverage(
    df: pd.DataFrame,
    *,
    analyte: str = "lbxgh",
    weight_col: str = "wtmec6yr",
    max_units_per_group: int = 120,
    seed: int = 42,
) -> FairnessReport:
    """LOO 90% coverage by age bracket (k-NN model), flag gaps > 3 pp."""
    brackets = ("20-39", "40-59", "60-79", "80+")
    groups: list[SubgroupCoverage] = []
    coverages: list[float] = []

    for br in brackets:
        sub = df.copy()
        sub["_br"] = sub["ridageyr"].apply(_age_bracket)
        sub = sub.loc[sub["_br"] == br]
        eff_n = float(sub[weight_col].sum()) if len(sub) else 0.0
        if eff_n < FAIRNESS_MIN_WEIGHTED_N or len(sub) < 30:
            groups.append(
                SubgroupCoverage(
                    label=br,
                    weighted_coverage_90=float("nan"),
                    effective_n=eff_n,
                    n_units=len(sub),
                    suppressed=True,
                )
            )
            continue
        ev = evaluate_model_loo(
            sub,
            analyte=analyte,
            weight_col=weight_col,
            model="knn",
            alphas=(0.9,),
            max_units=max_units_per_group,
            seed=seed,
        )
        cov90 = ev.coverages[0].weighted_coverage
        groups.append(
            SubgroupCoverage(
                label=br,
                weighted_coverage_90=cov90,
                effective_n=eff_n,
                n_units=ev.coverages[0].n_evaluated,
                suppressed=False,
            )
        )
        coverages.append(cov90)

    if len(coverages) >= 2:
        gap = (max(coverages) - min(coverages)) * 100.0
    else:
        gap = 0.0
    flagged = gap > FAIRNESS_COVERAGE_GAP_THRESHOLD_PP
    note = (
        f"Max weighted 90% coverage spread across age brackets: {gap:.1f} pp "
        f"(threshold {FAIRNESS_COVERAGE_GAP_THRESHOLD_PP} pp)."
    )
    return FairnessReport(groups=tuple(groups), max_gap_pp=gap, flagged=flagged, notes=note)
