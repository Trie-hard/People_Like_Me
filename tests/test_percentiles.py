import numpy as np
import pytest

from src.engine.percentiles import hazen_survey_weighted_percentile


def test_hazen_median_uniform_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    pct = hazen_survey_weighted_percentile(values, weights, 2.5)
    assert 40.0 < pct < 60.0


def test_hazen_clamps_extremes():
    values = np.array([10.0, 20.0, 30.0])
    weights = np.array([1.0, 2.0, 1.0])
    low = hazen_survey_weighted_percentile(values, weights, 5.0)
    high = hazen_survey_weighted_percentile(values, weights, 50.0)
    assert low >= 0.1
    assert high <= 99.9


def test_hazen_rejects_bad_weights():
    with pytest.raises(ValueError):
        hazen_survey_weighted_percentile(np.array([1.0]), np.array([0.0]), 1.0)
