"""
Smoke test / Thin integration: Maya persona evaluation (§1.1, §2.1).

Persona (Maya):
- Biological Sex: Female (RIAGENDR = 2)
- Age: 45 years
- BMI: 31.0 kg/m^2
- Blood Pressure: 128/82 mmHg (SBP=128, DBP=82)
- Lab: HbA1c = 5.8% (Borderline / Prediabetes range per ADA)

Evaluates:
- Lens A: Nearest peer cohort matching (k=300), MQI, and Hazen weighted percentile
- Lens B: Optimal benchmark reference (healthy age/sex matched), adequacy, and Hazen percentile
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import K_NEIGHBORS_DEFAULT
from src.engine import (
    UserProfile,
    hazen_survey_weighted_percentile,
    lens_b_subset,
    query_peer_cohort,
)
from src.engine.hemodynamics import hemodynamic_features


def _generate_synthetic_mec_cohort(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Fallback synthetic MEC cohort if NHANES parquet files have not been built."""
    rng = np.random.default_rng(seed)

    seqn = np.arange(10000, 10000 + n)
    riagendr = rng.choice([1, 2], size=n, p=[0.49, 0.51])
    ridageyr = rng.integers(20, 81, size=n)
    bmxbmi = rng.normal(loc=28.5, scale=6.0, size=n).clip(18.5, 55.0)
    sbp = rng.normal(loc=122.0, scale=15.0, size=n).clip(90.0, 190.0)
    dbp = rng.normal(loc=76.0, scale=10.0, size=n).clip(55.0, 110.0)
    map_vals, pp_vals = hemodynamic_features(sbp, dbp)
    wtmec6yr = rng.uniform(8000.0, 45000.0, size=n)

    # Correlate HbA1c slightly with age, BMI, and BP
    hba1c_base = 4.8 + 0.015 * (ridageyr - 20) + 0.03 * (bmxbmi - 22) + 0.005 * (sbp - 110)
    lbxgh = rng.normal(loc=hba1c_base, scale=0.4).clip(4.5, 12.0)

    # Diabetes diagnosis
    diq010 = np.where(lbxgh >= 6.5, rng.choice([1.0, 2.0], size=n, p=[0.7, 0.3]), 2.0)

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
            "lbxgh": lbxgh,
            "diq010": diq010,
            "wtmec6yr": wtmec6yr,
        }
    )


def run_maya_smoke_test(parquet_path: Path | str = "data/processed/mec_cohort_pooled.parquet") -> None:
    p = Path(parquet_path)
    if p.exists():
        print(f"Loading processed NHANES cohort from {p}...")
        df = pd.read_parquet(p)
    else:
        print(f"File {p} not found. Generating synthetic pooled cohort for demonstration...")
        df = _generate_synthetic_mec_cohort(n=1500)

    # 1. Define Maya Persona
    maya = UserProfile(
        age=45.0,
        sex=2,  # Female
        bmi=31.0,
        sbp=128.0,
        dbp=82.0,
    )
    maya_hba1c = 5.8  # %

    print("\n" + "=" * 60)
    print("PEOPLE LIKE ME — SMOKE TEST: MAYA PERSONA")
    print("=" * 60)
    print(f"Patient Profile: Age {maya.age:.0f}F | BMI {maya.bmi:.1f} | BP {maya.sbp:.0f}/{maya.dbp:.0f} mmHg")
    print(f"Biomarker Evaluated: HbA1c = {maya_hba1c:.1f}%")
    print(f"Hemodynamic Components: MAP = {maya.map:.1f} mmHg | PP = {maya.pp:.1f} mmHg")

    # 2. Lens A: Matched Peer Cohort
    female_pool_size = (df["riagendr"] == 2).sum()
    k_match = min(K_NEIGHBORS_DEFAULT, female_pool_size)

    peer_match = query_peer_cohort(
        df=df,
        user=maya,
        k=k_match,
    )

    print("\n" + "-" * 60)
    print(f"LENS A: Matched Peer Cohort (k={peer_match.k} nearest neighbors)")
    print("-" * 60)
    print(f"Match Quality Index (MQI): {peer_match.mqi.label} (Score: {peer_match.mqi.mean_distance:.2f})")
    print(f"  k-th Distance: {peer_match.mqi.kth_distance:.2f} | Adaptive Guard: {peer_match.mqi.adaptive_guard_triggered}")

    bbox = peer_match.bounding_box
    print(f"Demographic Bounding Box:")
    print(f"  Age: Median {bbox.age_median:.1f} [IQR: {bbox.age_q25:.1f} - {bbox.age_q75:.1f}]")
    print(f"  BMI: Median {bbox.bmi_median:.1f} [IQR: {bbox.bmi_q25:.1f} - {bbox.bmi_q75:.1f}]")
    print(f"  BP:  {bbox.to_dict()['bp_iqr_string']}")

    # Calculate Maya's percentile among peers
    peer_df = peer_match.neighbor_df
    has_hba1c = peer_df["lbxgh"].notna()
    peer_hba1c = peer_df.loc[has_hba1c, "lbxgh"].to_numpy(dtype=float)
    peer_weights = peer_df.loc[has_hba1c, "wtmec6yr"].to_numpy(dtype=float)

    peer_pct = hazen_survey_weighted_percentile(peer_hba1c, peer_weights, maya_hba1c)
    print(f"\n>> Maya's Peer Percentile (Lens A): {peer_pct:.1f}th percentile")

    # 3. Lens B: Optimal Health Benchmark Cohort
    lens_b = lens_b_subset(
        df=df,
        age=maya.age,
        sex=maya.sex,
        analyte="lbxgh",
        min_weighted_n=50,  # Demonstrative threshold for smaller pools
    )

    print("\n" + "-" * 60)
    print("LENS B: Optimal Health Benchmark Cohort")
    print("-" * 60)
    print(f"Benchmark Criteria: Female, Age Decade [40, 49], BMI 18.5-25, BP < 120/80, Non-Diabetic")
    print(f"Sample Size: n={lens_b.unweighted_n} (Effective Weighted n={lens_b.effective_n:.1f})")
    print(f"Benchmark Adequacy: {lens_b.status_message}")

    if len(lens_b.benchmark_df) > 0:
        lens_b_hba1c = lens_b.benchmark_df["lbxgh"].to_numpy(dtype=float)
        lens_b_weights = lens_b.weights
        lens_b_pct = hazen_survey_weighted_percentile(lens_b_hba1c, lens_b_weights, maya_hba1c)
        print(f"\n>> Maya's Optimal Health Percentile (Lens B): {lens_b_pct:.1f}th percentile")

        print("\n" + "=" * 60)
        print("THE DUAL-LENS DELTA INSIGHT:")
        print("=" * 60)
        print(
            f"Maya's HbA1c of {maya_hba1c:.1f}% places her at the {peer_pct:.1f}th percentile among women\n"
            f"with her somatic baseline (BMI ~31, BP ~128/82), but jumps to the {lens_b_pct:.1f}th percentile\n"
            f"when referenced against metabolically healthy peers in her age decade."
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_maya_smoke_test()
