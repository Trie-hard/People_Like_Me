"""
Unit tests for Lens B optimal health benchmark cohort (src/engine/benchmark_cohort.py).
Uses synthetic DataFrames to assert all clinical inclusion rules and boundary conditions (§2.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import LENS_B_MIN_WEIGHTED_N
from src.engine.benchmark_cohort import (
    compute_effective_sample_size,
    get_age_decade_bounds,
    lens_b_mask,
    lens_b_subset,
)


def _base_row(**kwargs) -> dict:
    """Return a baseline compliant Lens B participant profile."""
    row = {
        "seqn": 1001,
        "riagendr": 2,  # Female
        "ridageyr": 45,  # 40-49 decade
        "bmxbmi": 22.0,  # 18.5 <= bmi < 25
        "sbp": 110.0,  # < 120
        "dbp": 72.0,  # < 80
        "diq010": 2.0,  # No diabetes
        "lbxgh": 5.2,  # < 5.7
        "lbxglu": 88.0,  # < 100
        "wtmec6yr": 15000.0,
    }
    row.update(kwargs)
    return row


def test_age_decade_bounds_calculation():
    """Verify age decade calculation and 80+ top-coded rule (§2.1)."""
    # Standard decades
    assert get_age_decade_bounds(20) == (20, 29, False)
    assert get_age_decade_bounds(24.5) == (20, 29, False)
    assert get_age_decade_bounds(45) == (40, 49, False)
    assert get_age_decade_bounds(79) == (70, 79, False)

    # Top-coded 80+
    assert get_age_decade_bounds(80) == (80, 80, True)
    assert get_age_decade_bounds(85) == (80, 80, True)


def test_biological_sex_stratification():
    """Lens B must strictly match user's biological sex."""
    df = pd.DataFrame(
        [
            _base_row(seqn=1, riagendr=2),  # Female
            _base_row(seqn=2, riagendr=1),  # Male
        ]
    )

    mask_f = lens_b_mask(df, age=45, sex="F")
    assert mask_f.tolist() == [True, False]

    mask_m = lens_b_mask(df, age=45, sex="M")
    assert mask_m.tolist() == [False, True]


def test_age_decade_filtering_including_80_top_coded():
    """Test age decade matching and the age 80 top-coded edge case (§2.1)."""
    df = pd.DataFrame(
        [
            _base_row(seqn=1, ridageyr=39),  # Previous decade
            _base_row(seqn=2, ridageyr=40),  # In decade
            _base_row(seqn=3, ridageyr=45),  # In decade
            _base_row(seqn=4, ridageyr=49),  # In decade
            _base_row(seqn=5, ridageyr=50),  # Next decade
            _base_row(seqn=6, ridageyr=79),  # 70-79 decade
            _base_row(seqn=7, ridageyr=80),  # 80 top-coded
        ]
    )

    # For 45-year-old user: [40, 49]
    mask_45 = lens_b_mask(df, age=45, sex=2)
    assert df.loc[mask_45, "seqn"].tolist() == [2, 3, 4]

    # For 79-year-old user: [70, 79], must NOT include 80
    mask_79 = lens_b_mask(df, age=79, sex=2)
    assert df.loc[mask_79, "seqn"].tolist() == [6]

    # For 80-year-old user: [80, 80] top-coded, must NOT include 79
    mask_80 = lens_b_mask(df, age=80, sex=2)
    assert df.loc[mask_80, "seqn"].tolist() == [7]

    res_80 = lens_b_subset(df, age=80, sex=2, min_weighted_n=1)
    assert res_80.is_top_coded_age is True
    assert res_80.age_decade_bounds == (80, 80)


def test_bmi_boundaries():
    """Lens B BMI must be in [18.5, 25.0) kg/m^2."""
    df = pd.DataFrame(
        [
            _base_row(seqn=1, bmxbmi=18.4),  # Underweight -> exclude
            _base_row(seqn=2, bmxbmi=18.5),  # Exact lower bound -> include
            _base_row(seqn=3, bmxbmi=22.0),  # Normal -> include
            _base_row(seqn=4, bmxbmi=24.99),  # Just below 25 -> include
            _base_row(seqn=5, bmxbmi=25.0),  # Exact upper bound -> exclude (< 25.0)
            _base_row(seqn=6, bmxbmi=31.0),  # Overweight -> exclude
        ]
    )

    mask = lens_b_mask(df, age=45, sex=2)
    assert df.loc[mask, "seqn"].tolist() == [2, 3, 4]


def test_blood_pressure_boundaries():
    """Lens B requires measured SBP < 120 AND DBP < 80 mmHg."""
    df = pd.DataFrame(
        [
            _base_row(seqn=1, sbp=119.0, dbp=79.0),  # Normotensive -> include
            _base_row(seqn=2, sbp=120.0, dbp=75.0),  # SBP == 120 -> exclude (< 120)
            _base_row(seqn=3, sbp=115.0, dbp=80.0),  # DBP == 80 -> exclude (< 80)
            _base_row(seqn=4, sbp=135.0, dbp=85.0),  # Hypertensive -> exclude
        ]
    )

    mask = lens_b_mask(df, age=45, sex=2)
    assert df.loc[mask, "seqn"].tolist() == [1]


def test_diabetic_status_and_borderline_diq010_3_edge_case():
    """
    DIQ010 == 2 is required for non-diabetic benchmark (§2.1).
    Borderline diabetes (DIQ010 == 3) and diagnosed diabetes (DIQ010 == 1) must be EXCLUDED.
    """
    df = pd.DataFrame(
        [
            _base_row(seqn=1, diq010=2.0),  # No diabetes -> include
            _base_row(seqn=2, diq010=1.0),  # Diagnosed diabetes -> exclude
            _base_row(seqn=3, diq010=3.0),  # Borderline diabetes -> EXCLUDE (§2.1 / §2.2)
            _base_row(seqn=4, diq010=np.nan),  # Missing -> exclude
        ]
    )

    mask = lens_b_mask(df, age=45, sex=2)
    assert df.loc[mask, "seqn"].tolist() == [1]


def test_glycemia_rules_and_analyte_filter():
    """
    Lab glycemia thresholds: HbA1c < 5.7%, FPG < 100 mg/dL.
    Analyte-specific query ensures target biomarker is present and eligible.
    """
    df = pd.DataFrame(
        [
            _base_row(seqn=1, lbxgh=5.4, lbxglu=90.0),  # Optimal -> include
            _base_row(seqn=2, lbxgh=5.7, lbxglu=90.0),  # HbA1c == 5.7 (prediabetic) -> exclude
            _base_row(seqn=3, lbxgh=5.3, lbxglu=100.0),  # FPG == 100 (impaired) -> exclude
            _base_row(seqn=4, lbxgh=np.nan, lbxglu=92.0),  # Missing HbA1c
        ]
    )

    # General Lens B mask (without specific analyte enforcement)
    mask_gen = lens_b_mask(df, age=45, sex=2)
    # Rows 1 and 4 have valid DIQ010==2 and no glycemic violations
    assert df.loc[mask_gen, "seqn"].tolist() == [1, 4]

    # Specific analyte enforcement for HbA1c: row 4 (missing lbxgh) must be excluded
    mask_hba1c = lens_b_mask(df, age=45, sex=2, analyte="lbxgh")
    assert df.loc[mask_hba1c, "seqn"].tolist() == [1]


def test_weighted_n_adequacy_check():
    """
    Adequacy check vs LENS_B_MIN_WEIGHTED_N (200) (§2.1).
    Flags 'Insufficient benchmark sample' when effective n < 200.
    """
    # 1. Under-powered benchmark (e.g. 100 participants with uniform weight)
    df_small = pd.DataFrame([_base_row(seqn=i) for i in range(100)])
    res_small = lens_b_subset(df_small, age=45, sex=2)
    assert res_small.unweighted_n == 100
    assert res_small.effective_n == 100.0
    assert res_small.is_sufficient is False
    assert "Insufficient benchmark sample" in res_small.status_message

    # 2. Sufficient benchmark (e.g. 250 participants with uniform weight)
    df_large = pd.DataFrame([_base_row(seqn=i) for i in range(250)])
    res_large = lens_b_subset(df_large, age=45, sex=2)
    assert res_large.unweighted_n == 250
    assert res_large.effective_n == 250.0
    assert res_large.is_sufficient is True
    assert "Sufficient benchmark sample" in res_large.status_message


def test_kish_effective_sample_size_formula():
    """Verify Kish effective sample size calculation: (sum w)^2 / sum(w^2)."""
    # Uniform weights -> n_eff == N
    w_uni = np.ones(50)
    assert np.isclose(compute_effective_sample_size(w_uni), 50.0)

    # Empty array -> 0.0
    assert compute_effective_sample_size(np.array([])) == 0.0

    # Unequal weights -> n_eff < N
    w_uneq = np.array([1.0, 1.0, 10.0])
    # sum = 12, sum_sq = 1 + 1 + 100 = 102
    # n_eff = 144 / 102 = 1.41176...
    expected = (12.0 ** 2) / 102.0
    assert np.isclose(compute_effective_sample_size(w_uneq), expected)
