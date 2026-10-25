"""
Unit tests for Lens A k-NN peer matching engine (src/engine/matcher.py).
Uses purely synthetic DataFrames (no NHANES download required in CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    COV_RIDGE_TRACE_FRAC,
    MATCHING_FEATURE_COUNT,
    MQI_ADAPTIVE_NEIGHBOR_DISTANCE,
    MQI_HIGH_THRESHOLD,
    MQI_MODERATE_THRESHOLD,
)
from src.engine.hemodynamics import hemodynamic_features
from src.engine.matcher import (
    CohortMatcher,
    DemographicBoundingBox,
    MatchQuality,
    PeerMatchResult,
    UserProfile,
    compute_demographic_bounding_box,
    compute_mqi,
    fit_regularized_covariance,
    normalize_sex,
    query_peer_cohort,
    weighted_quantiles,
)


def make_synthetic_cohort(
    n_per_sex: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic NHANES-like cohort."""
    rng = np.random.default_rng(seed)
    n = n_per_sex * 2

    seqn = np.arange(1000, 1000 + n)
    riagendr = np.array([1] * n_per_sex + [2] * n_per_sex)
    ridageyr = rng.integers(20, 81, size=n)
    bmxbmi = rng.uniform(18.5, 42.0, size=n)
    sbp = rng.uniform(95.0, 160.0, size=n)
    dbp = rng.uniform(60.0, 100.0, size=n)
    map_vals, pp_vals = hemodynamic_features(sbp, dbp)
    wtmec6yr = rng.uniform(5000.0, 45000.0, size=n)
    lbxgh = rng.uniform(4.8, 8.5, size=n)
    diq010 = rng.choice([1.0, 2.0, 3.0], size=n, p=[0.1, 0.85, 0.05])

    return pd.DataFrame(
        {
            "seqn": seqn,
            "riagendr": riagendr,
            "ridageyr": ridageyr,
            "bmxbmi": bmxbmi,
            "sbp": sbp,
            "dbp": dbp,
            "map": map_vals,
            "pp": pp_vals,
            "wtmec6yr": wtmec6yr,
            "lbxgh": lbxgh,
            "diq010": diq010,
        }
    )


def test_sex_stratification_strictness():
    """Matching must strictly isolate the requested biological sex (§4.2)."""
    df = make_synthetic_cohort(n_per_sex=100)

    # Query female (sex=2)
    res_f = query_peer_cohort(
        df=df,
        age=45,
        sex=2,
        bmi=30.0,
        sbp=125.0,
        dbp=82.0,
        k=25,
    )
    assert len(res_f.neighbor_df) == 25
    assert (res_f.neighbor_df["riagendr"] == 2).all()

    # Query male (sex='M')
    res_m = query_peer_cohort(
        df=df,
        age=50,
        sex="M",
        bmi=27.0,
        sbp=120.0,
        dbp=80.0,
        k=25,
    )
    assert len(res_m.neighbor_df) == 25
    assert (res_m.neighbor_df["riagendr"] == 1).all()


def test_k_neighborhood_size_and_monotonicity():
    """Verify returned neighbor size k and sorted distance order."""
    df = make_synthetic_cohort(n_per_sex=100)

    for k_val in (10, 40, 75):
        result = query_peer_cohort(
            df=df,
            age=35,
            sex="F",
            bmi=24.5,
            sbp=115.0,
            dbp=75.0,
            k=k_val,
        )
        assert len(result.neighbor_df) == k_val
        assert len(result.distances) == k_val
        assert len(result.weights) == k_val
        assert result.k == k_val
        # Monotonically non-decreasing distances
        assert np.all(np.diff(result.distances) >= 0.0)
        # Weights must be positive
        assert np.all(result.weights > 0.0)


def test_regularized_covariance_and_whitening_accuracy():
    """Verify trace-based ridge regularization: λ = COV_RIDGE_TRACE_FRAC * tr(Σ) / 4."""
    X = np.array(
        [
            [40.0, 25.0, 90.0, 40.0],
            [50.0, 30.0, 95.0, 45.0],
            [60.0, 28.0, 92.0, 42.0],
            [45.0, 22.0, 88.0, 38.0],
            [55.0, 35.0, 100.0, 50.0],
        ],
        dtype=float,
    )
    cov_sample = np.cov(X, rowvar=False, ddof=1)
    expected_lambda = COV_RIDGE_TRACE_FRAC * np.trace(cov_sample) / MATCHING_FEATURE_COUNT
    expected_cov_reg = cov_sample + expected_lambda * np.eye(MATCHING_FEATURE_COUNT)

    cov_reg, L_inv, lam = fit_regularized_covariance(X)
    assert np.isclose(lam, expected_lambda)
    assert np.allclose(cov_reg, expected_cov_reg)
    # L_inv.T @ L_inv == cov_reg^{-1}
    cov_inv_reconstructed = L_inv.T @ L_inv
    assert np.allclose(cov_inv_reconstructed, np.linalg.inv(expected_cov_reg))


def test_exact_match_zero_distance():
    """A user whose features exactly match a participant in the pool has Mahalanobis distance ~0."""
    df = make_synthetic_cohort(n_per_sex=50)
    # Pick a specific female row
    female_row = df.loc[df["riagendr"] == 2].iloc[0]

    result = query_peer_cohort(
        df=df,
        age=float(female_row["ridageyr"]),
        sex=2,
        bmi=float(female_row["bmxbmi"]),
        sbp=float(female_row["sbp"]),
        dbp=float(female_row["dbp"]),
        k=5,
    )
    # The nearest neighbor should have distance very close to 0
    assert result.distances[0] < 1e-4
    assert result.neighbor_df.iloc[0]["seqn"] == female_row["seqn"]


def test_mqi_thresholds():
    """Verify MQI confidence boundaries per PROJECT_PLAN.md §4.4."""
    # 1. High Confidence (< 1.2)
    d_high = np.array([0.5, 0.7, 0.9, 1.1])
    mqi_high = compute_mqi(d_high)
    assert mqi_high.category == "high"
    assert mqi_high.label == "High Confidence"
    assert not mqi_high.adaptive_guard_triggered

    # 2. Moderate Confidence (1.2 <= mean < 2.0)
    d_mod = np.array([1.2, 1.4, 1.6, 1.8])
    mqi_mod = compute_mqi(d_mod)
    assert mqi_mod.category == "moderate"
    assert mqi_mod.label == "Moderate Confidence"
    assert not mqi_mod.adaptive_guard_triggered

    # 3. Low Support Warning (mean >= 2.0)
    d_low = np.array([2.0, 2.1, 2.3, 2.4])
    mqi_low = compute_mqi(d_low)
    assert mqi_low.category == "low"
    assert mqi_low.label == "Low Support Warning"


def test_mqi_adaptive_guard_triggers_on_kth_distance():
    """
    If the k-th neighbor distance > 2.5, cap displayed MQI at 'low support'
    even when the mean distance is small (§4.2.1 / §4.4).
    """
    # Mean is (0.4 + 0.5 + 0.6 + 2.6) / 4 = 1.025 (< 1.2, which would normally be 'high')
    # But kth distance is 2.6 > 2.5
    d_guarded = np.array([0.4, 0.5, 0.6, 2.6])
    mqi = compute_mqi(d_guarded)

    assert mqi.adaptive_guard_triggered is True
    assert mqi.category == "low"
    assert mqi.label == "Low Support Warning"
    assert mqi.kth_distance == 2.6
    assert mqi.mean_distance < MQI_HIGH_THRESHOLD  # Confirms mean was low


def test_demographic_bounding_box_weighted_quantiles():
    """Verify bounding box calculates correct weighted medians and IQRs."""
    # Simple linear sequence
    n = 21
    df = pd.DataFrame(
        {
            "ridageyr": np.linspace(20, 60, n),
            "bmxbmi": np.linspace(20, 40, n),
            "sbp": np.linspace(100, 140, n),
            "dbp": np.linspace(60, 90, n),
        }
    )
    weights = np.ones(n)

    bbox = compute_demographic_bounding_box(df, weights)

    # Median of linspace(20, 60, 21) is 40
    assert np.isclose(bbox.age_median, 40.0)
    assert bbox.age_q25 < bbox.age_median < bbox.age_q75
    assert np.isclose(bbox.bmi_median, 30.0)
    assert bbox.bmi_q25 < bbox.bmi_median < bbox.bmi_q75
    assert np.isclose(bbox.sbp_median, 120.0)
    assert np.isclose(bbox.dbp_median, 75.0)

    # Check to_dict formatting
    d = bbox.to_dict()
    assert "bp_iqr_string" in d
    assert "age" in d


def test_auto_compute_map_pp_when_missing():
    """Matcher should compute MAP and PP automatically if only SBP/DBP exist."""
    df = make_synthetic_cohort(n_per_sex=50)
    df_no_hemo = df.drop(columns=["map", "pp"])

    matcher = CohortMatcher(df_no_hemo)
    assert "map" in matcher.raw_df.columns
    assert "pp" in matcher.raw_df.columns

    res = matcher.query(age=40, sex=1, bmi=25.0, sbp=120.0, dbp=80.0, k=10)
    assert len(res.neighbor_df) == 10


def test_validation_errors():
    """Matcher should raise informative ValueErrors for invalid parameters."""
    df = make_synthetic_cohort(n_per_sex=30)
    matcher = CohortMatcher(df)

    # Invalid sex string
    with pytest.raises(ValueError, match="Unrecognized biological sex"):
        matcher.query(age=40, sex="Unknown", bmi=25.0, sbp=120.0, dbp=80.0, k=5)

    # k < 1
    with pytest.raises(ValueError, match="k must be >= 1"):
        matcher.query(age=40, sex=1, bmi=25.0, sbp=120.0, dbp=80.0, k=0)

    # k > available stratum size (30)
    with pytest.raises(ValueError, match="Requested k=50 neighbors, but stratum"):
        matcher.query(age=40, sex=1, bmi=25.0, sbp=120.0, dbp=80.0, k=50)

    # Enforce k bounds
    with pytest.raises(ValueError, match="outside allowed bounds"):
        matcher.query(age=40, sex=1, bmi=25.0, sbp=120.0, dbp=80.0, k=10, enforce_k_bounds=True)
