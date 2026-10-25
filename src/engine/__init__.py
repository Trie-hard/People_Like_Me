"""
Core mathematical and statistical matching engine for People Like Me.

Exports public API for:
- Hemodynamic transformations (MAP, PP)
- Hazen continuity-corrected survey-weighted percentiles
- k-NN stratified Whitened Mahalanobis peer matching (Lens A)
- Optimal health benchmark cohort extraction and adequacy evaluation (Lens B)
"""

from __future__ import annotations

from src.engine.benchmark_cohort import (
    LensBResult,
    compute_effective_sample_size,
    get_age_decade_bounds,
    lens_b_mask,
    lens_b_subset,
)
from src.engine.hemodynamics import (
    hemodynamic_features,
    mean_arterial_pressure,
    pulse_pressure,
)
from src.engine.matcher import (
    CohortMatcher,
    DemographicBoundingBox,
    MatchQuality,
    PeerMatchResult,
    UserProfile,
    compute_demographic_bounding_box,
    compute_mqi,
    normalize_sex,
    query_peer_cohort,
    weighted_quantiles,
)
from src.engine.percentiles import hazen_survey_weighted_percentile
from src.engine.telemetry import (
    DysglycemiaIceberg,
    MedicationRates,
    dysglycemia_iceberg,
    peer_medication_rates,
)

__all__ = [
    # Hemodynamics
    "mean_arterial_pressure",
    "pulse_pressure",
    "hemodynamic_features",
    # Percentiles
    "hazen_survey_weighted_percentile",
    # Telemetry
    "DysglycemiaIceberg",
    "MedicationRates",
    "dysglycemia_iceberg",
    "peer_medication_rates",
    # Lens A Matching
    "UserProfile",
    "MatchQuality",
    "DemographicBoundingBox",
    "PeerMatchResult",
    "CohortMatcher",
    "query_peer_cohort",
    "compute_mqi",
    "compute_demographic_bounding_box",
    "weighted_quantiles",
    "normalize_sex",
    # Lens B Benchmark
    "LensBResult",
    "lens_b_mask",
    "lens_b_subset",
    "get_age_decade_bounds",
    "compute_effective_sample_size",
]
