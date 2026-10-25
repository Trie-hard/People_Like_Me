"""
Unit tests for telemetry engine (src/engine/telemetry.py).
All tests use synthetic DataFrames — no NHANES download required in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.telemetry import (
    DysglycemiaIceberg,
    MedicationRates,
    dysglycemia_iceberg,
    peer_medication_rates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_weights(n: int, w: float = 10_000.0) -> np.ndarray:
    return np.full(n, w, dtype=float)


def _make_peer_df(**cols) -> pd.DataFrame:
    """Build a minimal peer DataFrame from keyword column arrays."""
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# peer_medication_rates tests
# ---------------------------------------------------------------------------

class TestPeerMedicationRates:

    def test_zero_when_no_medication_columns(self) -> None:
        """Returns 0.0 for both rates when med columns are absent."""
        df = _make_peer_df(
            riagendr=[1, 2, 2],
            ridageyr=[40, 45, 50],
            diq010=[2.0, 2.0, 1.0],
        )
        w = _uniform_weights(3)
        rates = peer_medication_rates(df, w)
        assert rates.on_glucose_med_pct == 0.0
        assert rates.on_lipid_med_pct == 0.0
        assert rates.n_valid_weights == 3

    def test_on_glucose_med_harmonized_flag(self) -> None:
        """Harmonized on_glucose_med flag drives glucose rate correctly."""
        df = _make_peer_df(
            on_glucose_med=[True, False, True, False],
        )
        w = _uniform_weights(4)
        rates = peer_medication_rates(df, w)
        assert abs(rates.on_glucose_med_pct - 50.0) < 1e-6

    def test_on_lipid_med_harmonized_flag(self) -> None:
        """on_lipid_med flag: 1 of 4 on statins → 25%."""
        df = _make_peer_df(
            on_lipid_med=[False, False, False, True],
        )
        w = _uniform_weights(4)
        rates = peer_medication_rates(df, w)
        assert abs(rates.on_lipid_med_pct - 25.0) < 1e-6

    def test_diq050_fallback_for_insulin(self) -> None:
        """diq050==1 triggers glucose flag when on_glucose_med is absent."""
        df = _make_peer_df(
            diq050=[1.0, 0.0, 0.0],
            diq010=[1.0, 2.0, 2.0],
        )
        w = _uniform_weights(3)
        rates = peer_medication_rates(df, w)
        assert abs(rates.on_glucose_med_pct - (100.0 / 3)) < 0.5

    def test_diq070_fallback_for_pills(self) -> None:
        """diq070==1 (blood sugar pills) contributes to glucose flag."""
        df = _make_peer_df(
            diq070=[1.0, 1.0, 0.0, 0.0],
        )
        w = _uniform_weights(4)
        rates = peer_medication_rates(df, w)
        assert abs(rates.on_glucose_med_pct - 50.0) < 1e-6

    def test_survey_weighted_not_unweighted(self) -> None:
        """When weights differ, weighted rate should differ from simple mean."""
        df = _make_peer_df(
            on_glucose_med=[True, False],
        )
        # First person has 3× the weight → weighted rate ≈ 75%
        w = np.array([30_000.0, 10_000.0])
        rates = peer_medication_rates(df, w)
        assert abs(rates.on_glucose_med_pct - 75.0) < 1e-6

    def test_combined_harmonized_and_diq_fallback(self) -> None:
        """on_glucose_med and diq050/diq070 are OR-combined."""
        df = _make_peer_df(
            on_glucose_med=[False, False, True],
            diq050=[1.0, 0.0, 0.0],
        )
        w = _uniform_weights(3)
        rates = peer_medication_rates(df, w)
        # Rows 0 and 2 are on glucose meds → 66.7%
        assert rates.on_glucose_med_pct > 60.0

    def test_n_valid_weights_excludes_nan(self) -> None:
        """n_valid_weights counts only finite positive weights."""
        df = _make_peer_df(on_glucose_med=[True, False, True])
        w = np.array([10_000.0, np.nan, 0.0])
        rates = peer_medication_rates(df, w)
        assert rates.n_valid_weights == 1


# ---------------------------------------------------------------------------
# dysglycemia_iceberg tests
# ---------------------------------------------------------------------------

class TestDysglycemiaIceberg:

    def test_diagnosed_from_diq010(self) -> None:
        """diq010==1 → 100% diagnosed, no other category."""
        df = _make_peer_df(
            diq010=[1.0, 1.0],
            lbxgh=[7.0, 7.2],
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.diagnosed_pct - 100.0) < 1e-6
        assert ice.undiagnosed_pct == 0.0
        assert ice.prediabetes_pct == 0.0

    def test_undiagnosed_diabetes_via_hba1c(self) -> None:
        """diq010==2 but HbA1c >= 6.5 → undiagnosed diabetes."""
        df = _make_peer_df(
            diq010=[2.0, 2.0],
            lbxgh=[6.8, 5.4],
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.undiagnosed_pct - 50.0) < 1e-6
        assert abs(ice.normoglycemia_pct + ice.prediabetes_pct - 50.0) < 1e-6

    def test_undiagnosed_diabetes_via_fpg(self) -> None:
        """diq010==2 but FPG >= 126 → undiagnosed diabetes."""
        df = _make_peer_df(
            diq010=[2.0],
            lbxgh=[np.nan],
            lbxglu=[135.0],
        )
        w = _uniform_weights(1)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.undiagnosed_pct - 100.0) < 1e-6

    def test_prediabetes_hba1c_range(self) -> None:
        """HbA1c in [5.7, 6.5) with diq010==2 → prediabetes."""
        df = _make_peer_df(
            diq010=[2.0, 2.0],
            lbxgh=[5.7, 6.4],
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.prediabetes_pct - 100.0) < 1e-6

    def test_prediabetes_fpg_range(self) -> None:
        """FPG in [100, 126) → prediabetes tier."""
        df = _make_peer_df(
            diq010=[2.0],
            lbxgh=[np.nan],
            lbxglu=[110.0],
        )
        w = _uniform_weights(1)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.prediabetes_pct - 100.0) < 1e-6

    def test_normoglycemia(self) -> None:
        """diq010==2 AND HbA1c < 5.7 → normoglycemia."""
        df = _make_peer_df(
            diq010=[2.0, 2.0],
            lbxgh=[5.0, 5.5],
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.normoglycemia_pct - 100.0) < 1e-6
        assert ice.prediabetes_pct == 0.0

    def test_unclassified_when_no_labs(self) -> None:
        """No HbA1c or FPG → unclassified (for transparency)."""
        df = _make_peer_df(
            diq010=[2.0, 2.0],
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.unclassified_pct - 100.0) < 1e-6

    def test_borderline_diq010_3_flagged_separately(self) -> None:
        """
        diq010==3 is NOT counted as diagnosed; it IS flagged in borderline_diq_pct.
        Per §2.2: treat borderline as 'not diagnosed' for iceberg counts.
        """
        df = _make_peer_df(
            diq010=[3.0, 2.0],
            lbxgh=[5.9, 5.0],  # row 0 prediabetes by HbA1c, row 1 normo
        )
        w = _uniform_weights(2)
        ice = dysglycemia_iceberg(df, w)
        # Borderline separately flagged
        assert abs(ice.borderline_diq_pct - 50.0) < 1e-6
        # diq010==3 row is classified by lab (HbA1c 5.9 = prediabetes)
        assert ice.diagnosed_pct == 0.0
        assert ice.prediabetes_pct > 0.0

    def test_prevalences_approximately_sum_to_100(self) -> None:
        """Classified rows: diagnosed + undiagnosed + prediabetes + normo ≈ 100% when all rows classified."""
        df = _make_peer_df(
            diq010=[1.0, 2.0, 2.0, 2.0],
            lbxgh=[7.0, 6.6, 5.8, 5.2],
        )
        w = _uniform_weights(4)
        ice = dysglycemia_iceberg(df, w)
        total = ice.diagnosed_pct + ice.undiagnosed_pct + ice.prediabetes_pct + ice.normoglycemia_pct
        assert abs(total - 100.0) < 1e-4

    def test_mixed_cohort_proportions(self) -> None:
        """Validates specific prevalence distribution in a mixed cohort."""
        # 10 peers: 2 diagnosed (20%), 1 undiagnosed (10%), 3 prediabetes (30%), 4 normo (40%)
        df = _make_peer_df(
            diq010=[1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            lbxgh=[7.1, 7.4, 6.8, 6.0, 5.9, 5.8, 5.3, 5.1, 5.0, 4.9],
        )
        w = _uniform_weights(10)
        ice = dysglycemia_iceberg(df, w)
        assert abs(ice.diagnosed_pct - 20.0) < 1e-6
        assert abs(ice.undiagnosed_pct - 10.0) < 1e-6
        assert abs(ice.prediabetes_pct - 30.0) < 1e-6
        assert abs(ice.normoglycemia_pct - 40.0) < 1e-6
        assert ice.n_total == 10
        assert ice.n_classified == 10
