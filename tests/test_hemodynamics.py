import numpy as np

from src.engine.hemodynamics import (
    hemodynamic_features,
    mean_arterial_pressure,
    pulse_pressure,
)


def test_map_and_pp_scalar():
    sbp, dbp = 128.0, 82.0
    assert np.isclose(mean_arterial_pressure(sbp, dbp), 82.0 + (128.0 - 82.0) / 3.0)
    assert pulse_pressure(sbp, dbp) == 46.0


def test_hemodynamic_features_vectorized():
    sbp = np.array([120.0, 140.0])
    dbp = np.array([80.0, 90.0])
    map_, pp = hemodynamic_features(sbp, dbp)
    assert map_.shape == (2,)
    assert pp.shape == (2,)
    assert pp[0] == 40.0
