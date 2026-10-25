import numpy as np
import pandas as pd

from src.data_pipeline.cleaner import average_blood_pressure, apply_fasting_filter


def test_bp_averaging_uses_readings_2_and_3():
    df = pd.DataFrame(
        {
            "BPXSY2": [120.0],
            "BPXSY3": [124.0],
            "BPXDI2": [80.0],
            "BPXDI3": [84.0],
        }
    )
    out = average_blood_pressure(df)
    assert out["sbp"].iloc[0] == 122.0
    assert out["dbp"].iloc[0] == 82.0
    assert np.isclose(out["map"].iloc[0], 82.0 + (122.0 - 82.0) / 3.0)


def test_fasting_hours_filter():
    df = pd.DataFrame({"PHAFSTHR": [7.5, 8.0, 12.0, 24.0]})
    out = apply_fasting_filter(df)
    assert list(out["PHAFSTHR"]) == [8.0, 12.0]
