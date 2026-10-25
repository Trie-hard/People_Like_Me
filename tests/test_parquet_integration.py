from pathlib import Path

import pandas as pd
import pytest

PARQUET = Path("data/processed/mec_cohort_pooled.parquet")


@pytest.mark.skipif(not PARQUET.exists(), reason="NHANES parquet not built locally")
def test_mec_parquet_sane_scale():
    df = pd.read_parquet(PARQUET)
    assert len(df) > 10_000
    assert df["wtmec6yr"].gt(0).all()
    assert set(df["cycle"].unique()) == {"2013_2014", "2015_2016", "2017_2018"}
    assert df["lbxgh"].notna().mean() > 0.85
