"""
People Like Me — Streamlit MVP Application (PROJECT_PLAN.md §5.1–5.4).

Multi-Biomarker Routing:
- MEC Pool (mec_cohort_pooled.parquet): HbA1c (lbxgh), Total Cholesterol (lbxtc), HDL-C (lbdhdd)
- Fasting Pool (fasting_cohort_pooled.parquet): Fasting Plasma Glucose (lbxglu) with wtsaf6yr
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="People Like Me",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make sure project root is importable regardless of launch directory
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import K_NEIGHBORS_DEFAULT
from src.engine import (
    UserProfile,
    hazen_survey_weighted_percentile,
    lens_b_subset,
    query_peer_cohort,
)
from src.engine.hemodynamics import hemodynamic_features
from src.ui.charts import dual_lens_density, ecdf_chart
from src.ui.components import (
    render_bounding_box_card,
    render_confounder_panel,
    render_lens_b_status,
    render_mqi_badge,
    render_no_data_warning,
    render_percentile_delta_card,
    render_safety_disclaimer,
)
from src.ui.pdf_generator import generate_pdf

# ---------------------------------------------------------------------------
# Multi-Biomarker Specifications (PROJECT_PLAN.md §3.2, §5.1, §5.2)
# ---------------------------------------------------------------------------

BIOMARKERS_CONFIG: dict[str, dict[str, Any]] = {
    "lbxgh": {
        "display_name": "HbA1c",
        "unit": "%",
        "pool": "mec",
        "weight_col": "wtmec6yr",
        "min_val": 3.0,
        "max_val": 15.0,
        "default_val": 5.5,
        "step": 0.1,
        "format": "%.1f",
        "maya_val": 5.8,
        "parquet_name": "mec_cohort_pooled.parquet",
    },
    "lbxglu": {
        "display_name": "Fasting Plasma Glucose",
        "unit": "mg/dL",
        "pool": "fasting",
        "weight_col": "wtsaf6yr",
        "min_val": 40.0,
        "max_val": 500.0,
        "default_val": 95.0,
        "step": 1.0,
        "format": "%.0f",
        "maya_val": 104.0,
        "parquet_name": "fasting_cohort_pooled.parquet",
    },
    "lbxtc": {
        "display_name": "Total Cholesterol",
        "unit": "mg/dL",
        "pool": "mec",
        "weight_col": "wtmec6yr",
        "min_val": 50.0,
        "max_val": 600.0,
        "default_val": 190.0,
        "step": 1.0,
        "format": "%.0f",
        "maya_val": 210.0,
        "parquet_name": "mec_cohort_pooled.parquet",
    },
    "lbdhdd": {
        "display_name": "HDL-Cholesterol",
        "unit": "mg/dL",
        "pool": "mec",
        "weight_col": "wtmec6yr",
        "min_val": 10.0,
        "max_val": 150.0,
        "default_val": 50.0,
        "step": 1.0,
        "format": "%.0f",
        "maya_val": 48.0,
        "parquet_name": "mec_cohort_pooled.parquet",
    },
}

# Maya demo values (§1.1)
MAYA = {
    "age": 45,
    "sex_label": "Female",
    "height_cm": 165.0,
    "weight_kg": 84.6,  # BMI ≈ 31.1
    "sbp": 128.0,
    "dbp": 82.0,
    "selected_markers": ["lbxgh", "lbxglu"],
}


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading NHANES cohort…")
def load_cohort_table(pool: str = "mec") -> pd.DataFrame:
    parquet_path = Path("data/processed") / f"{pool}_cohort_pooled.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return load_synthetic_cohort(pool=pool)


@st.cache_data(show_spinner="Generating synthetic demo cohort…")
def load_synthetic_cohort(pool: str = "mec") -> pd.DataFrame:
    """Synthetic cohort fallback for local dev when raw data is unpooled."""
    rng = np.random.default_rng(42)
    n = 1500
    riagendr = rng.choice([1, 2], size=n, p=[0.49, 0.51])
    ridageyr = rng.integers(20, 81, size=n)
    bmxbmi = rng.normal(loc=28.5, scale=6.0, size=n).clip(18.5, 55.0)
    sbp = rng.normal(loc=122.0, scale=15.0, size=n).clip(90.0, 190.0)
    dbp = rng.normal(loc=76.0, scale=10.0, size=n).clip(55.0, 110.0)
    map_vals, pp_vals = hemodynamic_features(sbp, dbp)
    wtmec6yr = rng.uniform(8_000.0, 45_000.0, size=n)
    hba1c_base = 4.8 + 0.015 * (ridageyr - 20) + 0.03 * (bmxbmi - 22) + 0.005 * (sbp - 110)
    lbxgh = rng.normal(loc=hba1c_base, scale=0.4).clip(4.5, 12.0)
    diq010 = np.where(lbxgh >= 6.5, rng.choice([1.0, 2.0], size=n, p=[0.7, 0.3]), 2.0)
    lbxtc = rng.normal(loc=195.0, scale=35.0, size=n).clip(100.0, 320.0)
    lbdhdd = rng.normal(loc=52.0, scale=14.0, size=n).clip(20.0, 110.0)
    on_glucose_med = (diq010 == 1) | (lbxgh >= 7.0)
    on_lipid_med = (lbxtc > 220) & (rng.uniform(size=n) > 0.5)

    data = {
        "seqn": np.arange(10_000, 10_000 + n),
        "cycle": ["2017_2018"] * n,
        "riagendr": riagendr,
        "ridageyr": ridageyr,
        "ridreth3": [1.0] * n,
        "indfmpir": [2.5] * n,
        "sdmvstra": [1] * n,
        "sdmvpsu": [1] * n,
        "bmxbmi": bmxbmi,
        "sbp": sbp,
        "dbp": dbp,
        "map": map_vals,
        "pp": pp_vals,
        "lbxgh": lbxgh,
        "lbxtc": lbxtc,
        "lbdhdd": lbdhdd,
        "diq010": diq010,
        "wtmec6yr": wtmec6yr,
        "on_glucose_med": on_glucose_med,
        "on_lipid_med": on_lipid_med,
    }

    if pool == "fasting":
        fpg_base = 80.0 + 0.5 * (ridageyr - 20) + 1.2 * (bmxbmi - 22) + 0.2 * (sbp - 110)
        lbxglu = rng.normal(loc=fpg_base, scale=12.0).clip(55.0, 350.0)
        data["lbxglu"] = lbxglu
        data["lbxtr"] = rng.normal(loc=125.0, scale=50.0, size=n).clip(30.0, 500.0)
        data["wtsaf6yr"] = wtmec6yr * 0.45
        data["phafsthr"] = [12.0] * n

    return pd.DataFrame(data)


def bmi_from_height_weight(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100.0
    return weight_kg / (h_m ** 2)


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, Any] = {
        "submitted": False,
        "age": 45,
        "sex_label": "Female",
        "height_cm": 170.0,
        "weight_kg": 70.0,
        "sbp": 120.0,
        "dbp": 80.0,
        "selected_markers": ["lbxgh"],
    }
    for marker_key, cfg in BIOMARKERS_CONFIG.items():
        defaults[f"val_{marker_key}"] = cfg["default_val"]

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------------------------

def sidebar_inputs() -> None:
    with st.sidebar:
        st.title("🧬 People Like Me")
        st.caption("Enter your biometric profile and lab values, then click **Analyse**.")

        # --- Maya Demo Preset Button ---
        if st.button("🌟 Load Maya demo (45F · BMI 31 · 128/82 · HbA1c 5.8% · FPG 104)", use_container_width=True):
            st.session_state["age"] = MAYA["age"]
            st.session_state["sex_label"] = MAYA["sex_label"]
            st.session_state["height_cm"] = MAYA["height_cm"]
            st.session_state["weight_kg"] = MAYA["weight_kg"]
            st.session_state["sbp"] = MAYA["sbp"]
            st.session_state["dbp"] = MAYA["dbp"]
            st.session_state["selected_markers"] = list(MAYA["selected_markers"])
            for m_key, m_cfg in BIOMARKERS_CONFIG.items():
                st.session_state[f"val_{m_key}"] = m_cfg["maya_val"]
            st.session_state["submitted"] = True
            st.rerun()

        st.markdown("---")
        st.subheader("Step 1 · Demographics")
        st.session_state["age"] = st.number_input(
            "Age (years)", min_value=20, max_value=80, value=int(st.session_state["age"]), step=1,
        )
        st.session_state["sex_label"] = st.selectbox(
            "Biological sex", ["Female", "Male"],
            index=0 if st.session_state["sex_label"] == "Female" else 1,
        )

        st.subheader("Step 2 · Body Measurements")
        col_h, col_w = st.columns(2)
        with col_h:
            st.session_state["height_cm"] = st.number_input(
                "Height (cm)", min_value=100.0, max_value=230.0,
                value=float(st.session_state["height_cm"]), step=0.5, format="%.1f",
            )
        with col_w:
            st.session_state["weight_kg"] = st.number_input(
                "Weight (kg)", min_value=30.0, max_value=300.0,
                value=float(st.session_state["weight_kg"]), step=0.5, format="%.1f",
            )
        bmi = bmi_from_height_weight(st.session_state["height_cm"], st.session_state["weight_kg"])
        st.info(f"Calculated BMI: **{bmi:.1f} kg/m²**")

        st.subheader("Step 3 · Blood Pressure")
        col_s, col_d = st.columns(2)
        with col_s:
            st.session_state["sbp"] = st.number_input(
                "Systolic (mmHg)", min_value=70.0, max_value=250.0,
                value=float(st.session_state["sbp"]), step=1.0, format="%.0f",
            )
        with col_d:
            st.session_state["dbp"] = st.number_input(
                "Diastolic (mmHg)", min_value=40.0, max_value=150.0,
                value=float(st.session_state["dbp"]), step=1.0, format="%.0f",
            )

        st.subheader("Step 4 · Laboratory Markers")
        st.session_state["selected_markers"] = st.multiselect(
            "Select Biomarkers to Contextualize",
            options=list(BIOMARKERS_CONFIG.keys()),
            default=st.session_state.get("selected_markers", ["lbxgh"]),
            format_func=lambda k: f"{BIOMARKERS_CONFIG[k]['display_name']} ({BIOMARKERS_CONFIG[k]['unit']})",
        )

        for marker_key in st.session_state["selected_markers"]:
            cfg = BIOMARKERS_CONFIG[marker_key]
            st.session_state[f"val_{marker_key}"] = st.number_input(
                f"{cfg['display_name']} ({cfg['unit']})",
                min_value=float(cfg["min_val"]),
                max_value=float(cfg["max_val"]),
                value=float(st.session_state.get(f"val_{marker_key}", cfg["default_val"])),
                step=float(cfg["step"]),
                format=cfg["format"],
            )

        st.markdown("---")
        if st.button("🔍 Analyse Biomarkers", type="primary", use_container_width=True):
            if not st.session_state["selected_markers"]:
                st.warning("Please select at least one biomarker to analyze.")
            else:
                st.session_state["submitted"] = True


# ---------------------------------------------------------------------------
# Main Panel — Analysis Execution
# ---------------------------------------------------------------------------

def run_analysis() -> None:
    age = int(st.session_state["age"])
    sex = 2 if st.session_state["sex_label"] == "Female" else 1
    sex_label = st.session_state["sex_label"]
    bmi = bmi_from_height_weight(st.session_state["height_cm"], st.session_state["weight_kg"])
    sbp = float(st.session_state["sbp"])
    dbp = float(st.session_state["dbp"])
    selected_markers = st.session_state.get("selected_markers", [])

    if not selected_markers:
        st.warning("No biomarkers selected. Select at least one in the sidebar.")
        return

    user = UserProfile(age=age, sex=sex, bmi=bmi, sbp=sbp, dbp=dbp)

    st.title("People Like Me 🧬")
    st.caption(
        f"Epidemiological context for: **{sex_label}, Age {age}** · "
        f"BMI {bmi:.1f} kg/m² · BP {sbp:.0f}/{dbp:.0f} mmHg · "
        f"MAP {user.map:.1f} mmHg · PP {user.pp:.1f} mmHg"
    )

    # 1. Determine which NHANES pools are required
    pools_needed = {BIOMARKERS_CONFIG[m]["pool"] for m in selected_markers}

    peer_matches: dict[str, Any] = {}
    lens_b_results: dict[str, dict[str, Any]] = {}
    cohort_dfs: dict[str, pd.DataFrame] = {}

    for pool in pools_needed:
        df_pool = load_cohort_table(pool=pool)
        cohort_dfs[pool] = df_pool

        # k-NN Peer matching on pool
        w_col = "wtsaf6yr" if pool == "fasting" else "wtmec6yr"
        sex_pool_size = int((df_pool["riagendr"] == sex).sum())
        k_val = min(K_NEIGHBORS_DEFAULT, sex_pool_size)

        peer_matches[pool] = query_peer_cohort(
            df=df_pool,
            user=user,
            k=k_val,
            pool=pool,
            weight_col=w_col,
        )

        lens_b_results[pool] = {}
        for m in selected_markers:
            if BIOMARKERS_CONFIG[m]["pool"] == pool:
                lens_b_results[pool][m] = lens_b_subset(
                    df=df_pool,
                    age=age,
                    sex=sex,
                    analyte=m,
                    weight_col=w_col,
                    pool=pool,
                )

    # Primary peer match for bounding box / telemetry (prefer MEC, else Fasting)
    primary_pool = "mec" if "mec" in peer_matches else "fasting"
    primary_peer_match = peer_matches[primary_pool]

    # Render MQI badge for primary match
    render_mqi_badge(primary_peer_match.mqi)
    st.markdown("")

    # 2. Compute percentiles and bundle results for each biomarker
    biomarker_evaluations: list[dict[str, Any]] = []

    for m in selected_markers:
        cfg = BIOMARKERS_CONFIG[m]
        pool = cfg["pool"]
        val = float(st.session_state.get(f"val_{m}", cfg["default_val"]))
        p_match = peer_matches[pool]
        l_b = lens_b_results[pool][m]

        # Peer percentile (Lens A)
        peer_df = p_match.neighbor_df
        c_map = {c.lower(): c for c in peer_df.columns}
        col_actual = c_map.get(m, m)
        w_actual = cfg["weight_col"]

        lens_a_pct: float | None = None
        has_lab = peer_df[col_actual].notna() if col_actual in peer_df.columns else pd.Series(False, index=peer_df.index)
        if has_lab.any():
            try:
                lens_a_pct = hazen_survey_weighted_percentile(
                    peer_df.loc[has_lab, col_actual].to_numpy(dtype=float),
                    peer_df.loc[has_lab, w_actual].to_numpy(dtype=float),
                    val,
                )
            except Exception:
                pass

        # Optimal benchmark percentile (Lens B)
        lens_b_pct: float | None = None
        if l_b.is_sufficient and len(l_b.benchmark_df) > 0:
            b_df = l_b.benchmark_df
            b_cmap = {c.lower(): c for c in b_df.columns}
            b_col_act = b_cmap.get(m, m)
            b_has = b_df[b_col_act].notna() if b_col_act in b_df.columns else pd.Series(False, index=b_df.index)
            if b_has.any():
                try:
                    lens_b_pct = hazen_survey_weighted_percentile(
                        b_df.loc[b_has, b_col_act].to_numpy(dtype=float),
                        l_b.weights[b_has.to_numpy()],
                        val,
                    )
                except Exception:
                    pass

        biomarker_evaluations.append({
            "key": m,
            "display_name": cfg["display_name"],
            "unit": cfg["unit"],
            "pool": pool,
            "value": val,
            "lens_a_pct": lens_a_pct,
            "lens_b_pct": lens_b_pct,
            "lens_b_sufficient": l_b.is_sufficient,
            "peer_match": p_match,
            "lens_b": l_b,
        })

    # 3. Multi-Biomarker Tabs / Sections
    st.markdown("### 🔬 Laboratory Analysis & Dual-Lens Context")

    if len(biomarker_evaluations) == 1:
        bm = biomarker_evaluations[0]
        render_lens_b_status(bm["lens_b"])
        render_percentile_delta_card(
            display_name=bm["display_name"],
            unit=bm["unit"],
            user_value=bm["value"],
            lens_a_pct=bm["lens_a_pct"],
            lens_b_pct=bm["lens_b_pct"],
            is_lens_b_sufficient=bm["lens_b_sufficient"],
        )
        t_dens, t_ecdf = st.tabs(["📊 Dual-Lens Distribution", "📈 Weighted ECDF"])
        with t_dens:
            fig_d = dual_lens_density(
                peer_match=bm["peer_match"],
                lens_b=bm["lens_b"],
                user_value=bm["value"],
                analyte_col=bm["key"],
                lens_a_pct=bm["lens_a_pct"],
                lens_b_pct=bm["lens_b_pct"],
            )
            st.plotly_chart(fig_d, use_container_width=True)
        with t_ecdf:
            fig_e = ecdf_chart(
                peer_match=bm["peer_match"],
                user_value=bm["value"],
                analyte_col=bm["key"],
                lens_a_pct=bm["lens_a_pct"],
            )
            st.plotly_chart(fig_e, use_container_width=True)
    else:
        marker_tabs = st.tabs([f"{bm['display_name']} ({bm['value']:.1f} {bm['unit']})" for bm in biomarker_evaluations])
        for tab, bm in zip(marker_tabs, biomarker_evaluations):
            with tab:
                render_lens_b_status(bm["lens_b"])
                render_percentile_delta_card(
                    display_name=bm["display_name"],
                    unit=bm["unit"],
                    user_value=bm["value"],
                    lens_a_pct=bm["lens_a_pct"],
                    lens_b_pct=bm["lens_b_pct"],
                    is_lens_b_sufficient=bm["lens_b_sufficient"],
                )
                sub_dens, sub_ecdf = st.tabs(["📊 Dual-Lens Distribution", "📈 Weighted ECDF"])
                with sub_dens:
                    fig_d = dual_lens_density(
                        peer_match=bm["peer_match"],
                        lens_b=bm["lens_b"],
                        user_value=bm["value"],
                        analyte_col=bm["key"],
                        lens_a_pct=bm["lens_a_pct"],
                        lens_b_pct=bm["lens_b_pct"],
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                with sub_ecdf:
                    fig_e = ecdf_chart(
                        peer_match=bm["peer_match"],
                        user_value=bm["value"],
                        analyte_col=bm["key"],
                        lens_a_pct=bm["lens_a_pct"],
                    )
                    st.plotly_chart(fig_e, use_container_width=True)

    st.markdown("---")

    # 4. Peer Demographic Profile (Bounding Box)
    render_bounding_box_card(primary_peer_match.bounding_box, k=primary_peer_match.k)

    st.markdown("---")

    # 5. Confounder & Medication Telemetry Panel (§2.2, §5.1)
    render_confounder_panel(primary_peer_match.neighbor_df, primary_peer_match.weights)

    st.markdown("---")

    # 6. Clinical Provider Brief Export (PDF)
    st.subheader("📄 Clinical Provider Brief")
    st.caption("Generate a personalized 1-page summary to share with your primary care provider.")

    try:
        pdf_biomarkers = [
            {
                "display_name": bm["display_name"],
                "unit": bm["unit"],
                "value": bm["value"],
                "lens_a_pct": bm["lens_a_pct"],
                "lens_b_pct": bm["lens_b_pct"],
                "lens_b_sufficient": bm["lens_b_sufficient"],
            }
            for bm in biomarker_evaluations
        ]

        pdf_bytes = generate_pdf(
            user=user,
            display_name=biomarker_evaluations[0]["display_name"],
            unit=biomarker_evaluations[0]["unit"],
            user_value=biomarker_evaluations[0]["value"],
            lens_a_pct=biomarker_evaluations[0]["lens_a_pct"],
            lens_b_pct=biomarker_evaluations[0]["lens_b_pct"],
            lens_a_k=primary_peer_match.k,
            lens_b_n=biomarker_evaluations[0]["lens_b"].effective_n,
            lens_b_sufficient=biomarker_evaluations[0]["lens_b"].is_sufficient,
            biomarker_results=pdf_biomarkers,
        )
        st.download_button(
            label="⬇️ Download Clinical Provider Brief (PDF)",
            data=pdf_bytes,
            file_name="people_like_me_clinical_brief.pdf",
            mime="application/pdf",
            type="primary",
        )
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")

    # 7. Persistent Safety Disclaimer (§5.4)
    render_safety_disclaimer()


# ---------------------------------------------------------------------------
# Landing Page State
# ---------------------------------------------------------------------------

def landing_page() -> None:
    st.title("People Like Me 🧬")
    st.subheader("Your lab results, in epidemiological context.")
    st.markdown(
        """
        Enter your demographic, body measurements, and laboratory values in the **sidebar**, 
        then click **Analyse Biomarkers**.

        **How People Like Me contextualizes your labs:**
        - **Lens A (Peer Cohort):** Shows where your laboratory values sit among US adults
          with a similar somatic baseline (same biological sex, similar age, BMI, and hemodynamics).
        - **Lens B (Optimal Health Benchmark):** Juxtaposes your peer reality against metabolically
          healthy adults of your same age and sex (normative BMI 18.5–25, BP < 120/80, non-diabetic).
        - **The Confounder Iceberg:** Surfaces the prevalence of pharmacotherapy (statins, antidiabetic pills/insulin)
          and undiagnosed dysglycemia among matched peers, explaining why 'average' is not always 'healthy'.

        > 💡 Click **Load Maya demo** in the sidebar to view a clinical worked example.
        """
    )

    mec_exists = Path("data/processed/mec_cohort_pooled.parquet").exists()
    fast_exists = Path("data/processed/fasting_cohort_pooled.parquet").exists()

    if not mec_exists or not fast_exists:
        render_no_data_warning()

    render_safety_disclaimer()


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

sidebar_inputs()

if st.session_state.get("submitted"):
    run_analysis()
else:
    landing_page()
