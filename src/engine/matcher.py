"""
k-NN peer matching engine on NHANES cohorts (PROJECT_PLAN.md §4.2–§4.4).

Implements:
- Sex-stratified regularized Whitened Mahalanobis matching on [age, bmi, map, pp]
- Sample covariance with ridge regularization: λ = COV_RIDGE_TRACE_FRAC * tr(Σ) / 4
- k-NN query with default k=300 (tunable via src.config)
- Match Quality Index (MQI) with mean distance and adaptive guard if kth > 2.5
- Demographic Bounding Box: weighted median and IQR for age, BMI, SBP, DBP
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.neighbors import NearestNeighbors

from src.config import (
    COV_RIDGE_TRACE_FRAC,
    K_NEIGHBORS_DEFAULT,
    K_NEIGHBORS_MAX,
    K_NEIGHBORS_MIN,
    MATCHING_FEATURE_COUNT,
    MQI_ADAPTIVE_NEIGHBOR_DISTANCE,
    MQI_HIGH_THRESHOLD,
    MQI_MODERATE_THRESHOLD,
)
from src.engine.hemodynamics import hemodynamic_features, mean_arterial_pressure, pulse_pressure


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UserProfile:
    """Demographic and somatic profile for a user query."""

    age: float
    sex: int  # 1 = Male, 2 = Female
    bmi: float
    sbp: float
    dbp: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "age", float(self.age))
        object.__setattr__(self, "sex", normalize_sex(self.sex))
        object.__setattr__(self, "bmi", float(self.bmi))
        object.__setattr__(self, "sbp", float(self.sbp))
        object.__setattr__(self, "dbp", float(self.dbp))

    @property
    def map(self) -> float:
        """Steady-state perfusion resistance."""
        return float(mean_arterial_pressure(np.array(self.sbp), np.array(self.dbp)))

    @property
    def pp(self) -> float:
        """Pulsatile load and arterial stiffness."""
        return float(pulse_pressure(np.array(self.sbp), np.array(self.dbp)))

    @property
    def feature_vector(self) -> np.ndarray:
        """4-dimensional matching coordinate vector [Age, BMI, MAP, PP]."""
        return np.array([self.age, self.bmi, self.map, self.pp], dtype=np.float64)


@dataclass(frozen=True)
class MatchQuality:
    """
    Match Quality Index (MQI) evaluating manifold support density (§4.4).

    Categories:
    - 'high': mean distance < 1.2
    - 'moderate': 1.2 <= mean distance < 2.0
    - 'low': mean distance >= 2.0 OR adaptive guard triggered (k-th distance > 2.5)
    """

    mean_distance: float
    kth_distance: float
    category: str  # "high", "moderate", "low"
    label: str  # "High Confidence", "Moderate Confidence", "Low Support Warning"
    adaptive_guard_triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_distance": self.mean_distance,
            "kth_distance": self.kth_distance,
            "category": self.category,
            "label": self.label,
            "adaptive_guard_triggered": self.adaptive_guard_triggered,
        }


@dataclass(frozen=True)
class DemographicBoundingBox:
    """
    Demographic envelope for the matched peer cohort (§4.4).
    Contains weighted medians and interquartile ranges (Q25, Q75)
    for age, BMI, SBP, and DBP.
    """

    age_median: float
    age_q25: float
    age_q75: float
    bmi_median: float
    bmi_q25: float
    bmi_q75: float
    sbp_median: float
    sbp_q25: float
    sbp_q75: float
    dbp_median: float
    dbp_q25: float
    dbp_q75: float

    @property
    def age_iqr(self) -> tuple[float, float]:
        return (self.age_q25, self.age_q75)

    @property
    def bmi_iqr(self) -> tuple[float, float]:
        return (self.bmi_q25, self.bmi_q75)

    @property
    def sbp_iqr(self) -> tuple[float, float]:
        return (self.sbp_q25, self.sbp_q75)

    @property
    def dbp_iqr(self) -> tuple[float, float]:
        return (self.dbp_q25, self.dbp_q75)

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": {"median": self.age_median, "q25": self.age_q25, "q75": self.age_q75},
            "bmi": {"median": self.bmi_median, "q25": self.bmi_q25, "q75": self.bmi_q75},
            "sbp": {"median": self.sbp_median, "q25": self.sbp_q25, "q75": self.sbp_q75},
            "dbp": {"median": self.dbp_median, "q25": self.dbp_q25, "q75": self.dbp_q75},
            "bp_iqr_string": f"{self.sbp_q25:.0f}/{self.dbp_q25:.0f} - {self.sbp_q75:.0f}/{self.dbp_q75:.0f} mmHg",
        }


@dataclass(frozen=True)
class PeerMatchResult:
    """Complete outcome of a k-NN peer cohort matching query."""

    neighbor_indices: np.ndarray
    neighbor_df: pd.DataFrame
    distances: np.ndarray
    weights: np.ndarray
    mqi: MatchQuality
    bounding_box: DemographicBoundingBox
    k: int
    user_profile: UserProfile

    @property
    def neighbor_seqns(self) -> np.ndarray | None:
        """Array of NHANES SEQN IDs if present in neighbor DataFrame."""
        for col in self.neighbor_df.columns:
            if col.lower() == "seqn":
                return self.neighbor_df[col].to_numpy()
        return None


# ---------------------------------------------------------------------------
# Utility & Statistical helpers
# ---------------------------------------------------------------------------

def normalize_sex(sex: int | str) -> int:
    """Normalize biological sex representation to NHANES coding (1=M, 2=F)."""
    if isinstance(sex, (int, np.integer)):
        if sex in (1, 2):
            return int(sex)
        raise ValueError(f"Numeric sex must be 1 (Male) or 2 (Female), got {sex}")
    if isinstance(sex, str):
        s = sex.strip().upper()
        if s in ("1", "M", "MALE"):
            return 1
        if s in ("2", "F", "FEMALE"):
            return 2
    raise ValueError(f"Unrecognized biological sex: {sex!r}. Expected 1/M/Male or 2/F/Female.")


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> np.ndarray:
    """
    Compute survey-weighted quantiles using Hazen continuity-corrected plotting positions.

    Parameters
    ----------
    values : 1-D array of sample values
    weights : 1-D array of strictly positive survey weights
    quantiles : target quantile positions in [0, 1]

    Returns
    -------
    np.ndarray of interpolated quantile values
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim != 1 or w.ndim != 1 or v.shape != w.shape:
        raise ValueError("values and weights must be 1-D arrays of equal length")
    if v.size == 0:
        raise ValueError("Cannot compute weighted quantiles on empty array")
    if np.any(w <= 0) or not np.all(np.isfinite(w)):
        raise ValueError("Survey weights must be strictly positive and finite")
    if not np.all(np.isfinite(v)):
        raise ValueError("Sample values must be finite")

    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    w_total = w_sorted.sum()
    # Hazen plotting positions: S_j = (sum_{m<=j} w_m - 0.5 * w_j) / sum(w)
    s = (cum_w - 0.5 * w_sorted) / w_total

    q_arr = np.asarray(quantiles, dtype=np.float64)
    return np.interp(q_arr, s, v_sorted)


def compute_mqi(distances: np.ndarray) -> MatchQuality:
    """
    Compute Match Quality Index (MQI) from Mahalanobis neighbor distances (§4.4).

    Parameters
    ----------
    distances : 1-D array of sorted Mahalanobis distances (length k)

    Returns
    -------
    MatchQuality object with category, label, and adaptive guard status.
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        raise ValueError("distances array cannot be empty")
    mean_d = float(np.mean(d))
    kth_d = float(d[-1])
    adaptive_guard = kth_d > MQI_ADAPTIVE_NEIGHBOR_DISTANCE

    if adaptive_guard:
        category = "low"
        label = "Low Support Warning"
    elif mean_d < MQI_HIGH_THRESHOLD:
        category = "high"
        label = "High Confidence"
    elif mean_d < MQI_MODERATE_THRESHOLD:
        category = "moderate"
        label = "Moderate Confidence"
    else:
        category = "low"
        label = "Low Support Warning"

    return MatchQuality(
        mean_distance=mean_d,
        kth_distance=kth_d,
        category=category,
        label=label,
        adaptive_guard_triggered=adaptive_guard,
    )


def compute_demographic_bounding_box(
    df_neighbors: pd.DataFrame,
    weights: np.ndarray,
) -> DemographicBoundingBox:
    """
    Calculate weighted median and IQR for age, BMI, SBP, and DBP (§4.4).
    """
    w = np.asarray(weights, dtype=np.float64)
    col_map = {c.lower(): c for c in df_neighbors.columns}
    for req in ("ridageyr", "bmxbmi", "sbp", "dbp"):
        if req not in col_map:
            raise KeyError(f"Column '{req}' required in neighbor DataFrame for bounding box")

    age_q = weighted_quantiles(df_neighbors[col_map["ridageyr"]].to_numpy(dtype=float), w)
    bmi_q = weighted_quantiles(df_neighbors[col_map["bmxbmi"]].to_numpy(dtype=float), w)
    sbp_q = weighted_quantiles(df_neighbors[col_map["sbp"]].to_numpy(dtype=float), w)
    dbp_q = weighted_quantiles(df_neighbors[col_map["dbp"]].to_numpy(dtype=float), w)

    return DemographicBoundingBox(
        age_median=float(age_q[1]),
        age_q25=float(age_q[0]),
        age_q75=float(age_q[2]),
        bmi_median=float(bmi_q[1]),
        bmi_q25=float(bmi_q[0]),
        bmi_q75=float(bmi_q[2]),
        sbp_median=float(sbp_q[1]),
        sbp_q25=float(sbp_q[0]),
        sbp_q75=float(sbp_q[2]),
        dbp_median=float(dbp_q[1]),
        dbp_q25=float(dbp_q[0]),
        dbp_q75=float(dbp_q[2]),
    )


def resolve_weight_column(
    df: pd.DataFrame,
    weight_col: str | None = None,
    pool: str = "mec",
) -> str:
    """Identify the appropriate survey weight column in the DataFrame."""
    if weight_col is not None:
        if weight_col in df.columns:
            return weight_col
        for c in df.columns:
            if c.lower() == weight_col.lower():
                return c
        raise KeyError(f"Specified weight column '{weight_col}' not found in DataFrame")

    cols_lower = {c.lower(): c for c in df.columns}
    if pool == "fasting" and "wtsaf6yr" in cols_lower:
        return cols_lower["wtsaf6yr"]
    if "wtmec6yr" in cols_lower:
        return cols_lower["wtmec6yr"]
    if "wtsaf6yr" in cols_lower:
        return cols_lower["wtsaf6yr"]
    for candidate in ("weight", "weights", "wtmec2yr", "wtsaf2yr"):
        if candidate in cols_lower:
            return cols_lower[candidate]

    raise KeyError(
        f"No valid survey weight column found (checked 'wtmec6yr', 'wtsaf6yr', etc.). "
        f"Available columns: {list(df.columns)}"
    )


def fit_regularized_covariance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit regularized covariance matrix and return its inverse Cholesky factor.

    Formula:
        Σ_λ = Σ + λ I
        λ = COV_RIDGE_TRACE_FRAC * tr(Σ) / 4
        L_inv such that L_inv^T @ L_inv = Σ_λ^{-1}

    Returns
    -------
    (cov_reg, L_inv, ridge_lambda)
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.shape[1] != MATCHING_FEATURE_COUNT:
        raise ValueError(
            f"Expected {MATCHING_FEATURE_COUNT} features [age, bmi, map, pp], got {X_arr.shape[1]}"
        )
    if X_arr.shape[0] < 2:
        raise ValueError("At least 2 samples required to estimate covariance")

    cov = np.cov(X_arr, rowvar=False, ddof=1)
    if cov.shape != (MATCHING_FEATURE_COUNT, MATCHING_FEATURE_COUNT):
        cov = np.atleast_2d(cov)

    trace_sigma = float(np.trace(cov))
    ridge_lambda = max(COV_RIDGE_TRACE_FRAC * trace_sigma / MATCHING_FEATURE_COUNT, 1e-12)
    cov_reg = cov + ridge_lambda * np.eye(MATCHING_FEATURE_COUNT)

    cov_inv = np.linalg.inv(cov_reg)
    # L_inv is upper-triangular: L_inv.T @ L_inv == cov_inv
    L_inv = scipy.linalg.cholesky(cov_inv, lower=False)

    return cov_reg, L_inv, ridge_lambda


# ---------------------------------------------------------------------------
# CohortMatcher class
# ---------------------------------------------------------------------------

class CohortMatcher:
    """
    Production-ready matching engine for NHANES cohorts (§4.2).

    Precomputes sex-stratified regularized covariance and Cholesky whitening transforms
    to allow sub-millisecond k-NN queries.
    """

    FEATURE_COLS = ("ridageyr", "bmxbmi", "map", "pp")

    def __init__(
        self,
        df: pd.DataFrame,
        pool: str = "mec",
        weight_col: str | None = None,
    ) -> None:
        self.pool = pool
        self.raw_df = df.copy()
        self._ensure_hemodynamic_columns(self.raw_df)

        self.weight_col = resolve_weight_column(self.raw_df, weight_col=weight_col, pool=pool)
        self.sex_col = self._find_column(self.raw_df, "riagendr")

        # Stratify by biological sex (1=Male, 2=Female)
        self.strata_df: dict[int, pd.DataFrame] = {}
        self.cov_reg: dict[int, np.ndarray] = {}
        self.L_inv: dict[int, np.ndarray] = {}
        self.ridge_lambda: dict[int, float] = {}
        self.whitened_Z: dict[int, np.ndarray] = {}
        self.nn_models: dict[int, NearestNeighbors] = {}

        for sex_val in (1, 2):
            sub = self.raw_df.loc[self.raw_df[self.sex_col] == sex_val].copy()
            if len(sub) == 0:
                continue
            self._fit_stratum(sex_val, sub)

    @classmethod
    def from_parquet(
        cls,
        pool: str = "mec",
        processed_dir: Path | str = "data/processed",
        weight_col: str | None = None,
    ) -> CohortMatcher:
        """Instantiate CohortMatcher directly from processed Parquet table."""
        path = Path(processed_dir) / f"{pool}_cohort_pooled.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Cohort parquet table not found at {path}. Run data pipeline first."
            )
        df = pd.read_parquet(path)
        return cls(df, pool=pool, weight_col=weight_col)

    def _find_column(self, df: pd.DataFrame, target: str) -> str:
        for c in df.columns:
            if c.lower() == target.lower():
                return c
        raise KeyError(f"Required column '{target}' not found in DataFrame columns: {list(df.columns)}")

    def _ensure_hemodynamic_columns(self, df: pd.DataFrame) -> None:
        """Compute MAP and PP if not already present."""
        cols_lower = {c.lower(): c for c in df.columns}
        if "map" not in cols_lower or "pp" not in cols_lower:
            if "sbp" in cols_lower and "dbp" in cols_lower:
                sbp_vals = df[cols_lower["sbp"]].to_numpy(dtype=float)
                dbp_vals = df[cols_lower["dbp"]].to_numpy(dtype=float)
                map_vals, pp_vals = hemodynamic_features(sbp_vals, dbp_vals)
                if "map" not in cols_lower:
                    df["map"] = map_vals
                if "pp" not in cols_lower:
                    df["pp"] = pp_vals
            else:
                raise KeyError("Columns 'map' and 'pp' (or 'sbp' and 'dbp') are required for matching")

    def _fit_stratum(self, sex_val: int, sub_df: pd.DataFrame) -> None:
        col_map = {c.lower(): c for c in sub_df.columns}
        for f in self.FEATURE_COLS:
            if f not in col_map:
                raise KeyError(f"Matching feature '{f}' missing from stratum DataFrame")

        feature_col_names = [col_map[f] for f in self.FEATURE_COLS]
        X = sub_df[feature_col_names].to_numpy(dtype=np.float64)

        cov_reg, L_inv, ridge_lam = fit_regularized_covariance(X)
        # Whitened coordinates Z: Euclidean distance in Z space equals Mahalanobis distance in X space
        Z = X @ L_inv.T

        nn = NearestNeighbors(metric="euclidean", algorithm="auto")
        nn.fit(Z)

        self.strata_df[sex_val] = sub_df
        self.cov_reg[sex_val] = cov_reg
        self.L_inv[sex_val] = L_inv
        self.ridge_lambda[sex_val] = ridge_lam
        self.whitened_Z[sex_val] = Z
        self.nn_models[sex_val] = nn

    def query(
        self,
        user: UserProfile | None = None,
        *,
        age: float | None = None,
        sex: int | str | None = None,
        bmi: float | None = None,
        sbp: float | None = None,
        dbp: float | None = None,
        k: int = K_NEIGHBORS_DEFAULT,
        enforce_k_bounds: bool = False,
    ) -> PeerMatchResult:
        """
        Query the k-nearest peer cohort for a user profile.

        Parameters
        ----------
        user : UserProfile instance, or pass individual fields (age, sex, bmi, sbp, dbp)
        k : number of nearest neighbors (default K_NEIGHBORS_DEFAULT = 300)
        enforce_k_bounds : if True, validate that K_NEIGHBORS_MIN <= k <= K_NEIGHBORS_MAX

        Returns
        -------
        PeerMatchResult containing neighbor indices, dataframe slice, distances,
        weights, Match Quality Index, and demographic bounding box.
        """
        if user is None:
            if any(v is None for v in (age, sex, bmi, sbp, dbp)):
                raise ValueError("Must provide either a UserProfile or all of (age, sex, bmi, sbp, dbp)")
            user = UserProfile(age=age, sex=sex, bmi=bmi, sbp=sbp, dbp=dbp)  # type: ignore

        sex_val = user.sex
        if sex_val not in self.strata_df:
            raise ValueError(f"No cohort data available for sex {sex_val}")

        stratum = self.strata_df[sex_val]
        n_available = len(stratum)

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if enforce_k_bounds and not (K_NEIGHBORS_MIN <= k <= K_NEIGHBORS_MAX):
            raise ValueError(
                f"k={k} outside allowed bounds [{K_NEIGHBORS_MIN}, {K_NEIGHBORS_MAX}] (§4.2.1)"
            )
        if k > n_available:
            raise ValueError(
                f"Requested k={k} neighbors, but stratum (sex={sex_val}) only contains {n_available} records"
            )

        # Transform user feature vector into whitened space
        x_user = user.feature_vector  # [age, bmi, map, pp]
        L_inv = self.L_inv[sex_val]
        z_user = x_user @ L_inv.T

        # Query nearest neighbors
        nn = self.nn_models[sex_val]
        distances, positional_indices = nn.kneighbors(z_user.reshape(1, -1), n_neighbors=k)
        dists = distances[0]
        pos_idx = positional_indices[0]

        neighbor_df = stratum.iloc[pos_idx].copy()
        neighbor_indices = neighbor_df.index.to_numpy()
        weights = neighbor_df[self.weight_col].to_numpy(dtype=np.float64)

        # Match Quality Index & Demographic Bounding Box
        mqi = compute_mqi(dists)
        bbox = compute_demographic_bounding_box(neighbor_df, weights)

        return PeerMatchResult(
            neighbor_indices=neighbor_indices,
            neighbor_df=neighbor_df,
            distances=dists,
            weights=weights,
            mqi=mqi,
            bounding_box=bbox,
            k=k,
            user_profile=user,
        )


# ---------------------------------------------------------------------------
# High-level functional API
# ---------------------------------------------------------------------------

def query_peer_cohort(
    df: pd.DataFrame | None = None,
    user: UserProfile | None = None,
    *,
    age: float | None = None,
    sex: int | str | None = None,
    bmi: float | None = None,
    sbp: float | None = None,
    dbp: float | None = None,
    k: int = K_NEIGHBORS_DEFAULT,
    pool: str = "mec",
    weight_col: str | None = None,
    processed_dir: Path | str = "data/processed",
    enforce_k_bounds: bool = False,
) -> PeerMatchResult:
    """
    Top-level entry point to query the peer cohort.

    If df is None, loads from data/processed/{pool}_cohort_pooled.parquet.
    """
    if df is None:
        matcher = CohortMatcher.from_parquet(
            pool=pool,
            processed_dir=processed_dir,
            weight_col=weight_col,
        )
    else:
        matcher = CohortMatcher(df, pool=pool, weight_col=weight_col)

    return matcher.query(
        user=user,
        age=age,
        sex=sex,
        bmi=bmi,
        sbp=sbp,
        dbp=dbp,
        k=k,
        enforce_k_bounds=enforce_k_bounds,
    )
