import numpy as np
import pandas as pd

from src.validation.coverage_evaluator import evaluate_model_loo


def _tiny_cohort(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sbp = rng.uniform(110, 140, n)
    dbp = rng.uniform(70, 90, n)
    return pd.DataFrame(
        {
            "ridageyr": rng.integers(25, 65, n),
            "riagendr": rng.choice([1, 2], n),
            "bmxbmi": rng.uniform(22, 35, n),
            "sbp": sbp,
            "dbp": dbp,
            "map": dbp + (sbp - dbp) / 3,
            "pp": sbp - dbp,
            "lbxgh": rng.normal(5.4, 0.5, n),
            "wtmec6yr": rng.uniform(5000, 20000, n),
        }
    )


def test_loo_coverage_runs_on_tiny_cohort():
    df = _tiny_cohort()
    ev = evaluate_model_loo(df, k=15, max_units=25, alphas=(0.9,))
    assert ev.coverages[0].n_evaluated == 25
    assert 0.0 <= ev.coverages[0].weighted_coverage <= 1.0
