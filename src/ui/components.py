"""
Streamlit UI component helpers for People Like Me (PROJECT_PLAN.md §5.1–5.4).

All functions return None — they call Streamlit display primitives directly.
Designed for import into app.py with no circular dependencies.

Public API
----------
render_mqi_badge(mqi)                   — Coloured MQI confidence badge.
render_bounding_box_card(bbox, k)       — Peer demographic IQR card.
render_lens_b_status(lens_b)            — Lens B adequacy chip.
render_safety_disclaimer()              — §5.4 mandatory safety block.
render_no_data_warning()                — Missing parquet prompt.
render_percentile_delta_card(...)       — Dual-lens delta summary card.
"""

from __future__ import annotations

import streamlit as st

from src.engine.benchmark_cohort import LensBResult
from src.engine.matcher import DemographicBoundingBox, MatchQuality, PeerMatchResult
from src.engine.telemetry import dysglycemia_iceberg, peer_medication_rates


# ---------------------------------------------------------------------------
# Colour palette (mirrors charts.py)
# ---------------------------------------------------------------------------

_MQI_COLOURS: dict[str, str] = {
    "high": "#16a34a",      # green
    "moderate": "#d97706",  # amber
    "low": "#dc2626",       # red
}

_MQI_ICONS: dict[str, str] = {
    "high": "✅",
    "moderate": "⚠️",
    "low": "🔴",
}


# ---------------------------------------------------------------------------
# MQI Badge
# ---------------------------------------------------------------------------

def render_mqi_badge(mqi: MatchQuality) -> None:
    """
    Render a colour-coded Match Quality Index badge (§4.4).

    High Confidence (green) / Moderate Confidence (amber) / Low Support Warning (red).
    Includes mean Mahalanobis distance and adaptive guard indicator.
    """
    colour = _MQI_COLOURS.get(mqi.category, "#6b7280")
    icon = _MQI_ICONS.get(mqi.category, "ℹ️")

    guard_note = " · Sparse region (k-th neighbour > 2.5)" if mqi.adaptive_guard_triggered else ""
    st.markdown(
        f"""
        <div style="
            display:inline-flex;align-items:center;gap:8px;
            background:#f9fafb;border:1.5px solid {colour};
            border-radius:8px;padding:6px 14px;margin-bottom:4px;
        ">
          <span style="font-size:18px">{icon}</span>
          <div>
            <span style="font-weight:600;color:{colour};font-size:14px">{mqi.label}</span>
            <span style="color:#6b7280;font-size:12px;margin-left:8px">
              mean distance {mqi.mean_distance:.2f}{guard_note}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Demographic Bounding Box Card
# ---------------------------------------------------------------------------

def render_bounding_box_card(bbox: DemographicBoundingBox, k: int) -> None:
    """
    Render the demographic envelope for the k-nearest peer cohort (§4.4 / §5.1).

    Shows weighted medians and interquartile ranges for Age, BMI, SBP, and DBP.
    """
    st.markdown("#### 👥 Peer Demographic Profile")
    st.caption(
        f"Showing the weighted IQR (Q₂₅–Q₇₅) and median among your {k} nearest NHANES peers."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Age (years)",
            value=f"{bbox.age_median:.0f}",
            delta=f"IQR {bbox.age_q25:.0f}–{bbox.age_q75:.0f}",
            delta_color="off",
        )
    with c2:
        st.metric(
            label="BMI (kg/m²)",
            value=f"{bbox.bmi_median:.1f}",
            delta=f"IQR {bbox.bmi_q25:.1f}–{bbox.bmi_q75:.1f}",
            delta_color="off",
        )
    with c3:
        st.metric(
            label="SBP (mmHg)",
            value=f"{bbox.sbp_median:.0f}",
            delta=f"IQR {bbox.sbp_q25:.0f}–{bbox.sbp_q75:.0f}",
            delta_color="off",
        )
    with c4:
        st.metric(
            label="DBP (mmHg)",
            value=f"{bbox.dbp_median:.0f}",
            delta=f"IQR {bbox.dbp_q25:.0f}–{bbox.dbp_q75:.0f}",
            delta_color="off",
        )


# ---------------------------------------------------------------------------
# Confounder & Telemetry Panel (§2.2, §5.1)
# ---------------------------------------------------------------------------

def render_confounder_panel(peer_df: pd.DataFrame, weights: np.ndarray) -> None:
    """
    Render Confounder Panel: medication rates and dysglycemia decomposition (§2.2, §5.1).
    Surfaces the proportion of matched peers receiving pharmacotherapy and
    the distribution of diagnosed/undiagnosed dysglycemia.
    """
    med_rates = peer_medication_rates(peer_df, weights)
    iceberg = dysglycemia_iceberg(peer_df, weights)

    st.markdown("#### 💊 Confounder & Medication Telemetry")
    st.caption(
        "Pharmacotherapy and underlying metabolic status among your matched NHANES peers (Lens A). "
        "Laboratory values among peers often appear normal solely due to active medications."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="Peers on Glucose Medication",
            value=f"{med_rates.on_glucose_med_pct:.1f}%",
            help="Insulin or oral diabetes pills reported in survey or prescription records",
        )
    with c2:
        st.metric(
            label="Peers on Lipid-Lowering Medication",
            value=f"{med_rates.on_lipid_med_pct:.1f}%",
            help="Statin-class lipid-lowering prescriptions from RXQ",
        )

    st.markdown("##### 🧊 The Diagnostic Iceberg (Peer Cohort)")
    st.caption(
        "Decomposition of glycemic status among peers with your demographic and somatic profile (§2.2)."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Diagnosed Diabetes", f"{iceberg.diagnosed_pct:.1f}%")
    with col2:
        st.metric(
            "Undiagnosed Diabetes",
            f"{iceberg.undiagnosed_pct:.1f}%",
            help="Survey negative but laboratory test indicates diabetes (HbA1c ≥ 6.5% or FPG ≥ 126 mg/dL)",
        )
    with col3:
        st.metric(
            "Prediabetes",
            f"{iceberg.prediabetes_pct:.1f}%",
            help="HbA1c 5.7–6.4% or FPG 100–125 mg/dL, without diabetes diagnosis",
        )
    with col4:
        st.metric("Normoglycemia", f"{iceberg.normoglycemia_pct:.1f}%")

    if iceberg.borderline_diq_pct > 0:
        st.caption(
            f"ℹ️ {iceberg.borderline_diq_pct:.1f}% of peers reported 'borderline' diabetes upon physician questioning."
        )

    st.info(
        "**Epidemiological Context:** High rates of pharmacotherapy or dysglycemia among peers mean "
        "that being 'average' in this cohort does not equal optimal metabolic health. "
        "Medications can normalize lab numbers without altering underlying somatic baseline.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
# Lens B Status Chip
# ---------------------------------------------------------------------------

def render_lens_b_status(lens_b: LensBResult) -> None:
    """
    Render Lens B benchmark adequacy status (§2.1).

    Shows effective weighted n, age decade bounds, and sufficiency flag.
    """
    age_lo, age_hi = lens_b.age_decade_bounds
    if age_lo == 80 and age_hi == 80:
        decade_label = "80+ (top-coded)"
    else:
        decade_label = f"{age_lo}–{age_hi}"

    n_label = f"n={lens_b.unweighted_n} (eff. weighted n={lens_b.effective_n:.0f})"

    if lens_b.is_sufficient:
        st.success(
            f"**Lens B benchmark:** Age decade {decade_label} · {n_label} · ✅ Sufficient sample",
            icon="🧪",
        )
    else:
        st.warning(
            f"**Lens B benchmark:** Age decade {decade_label} · {n_label} · "
            f"⚠️ {lens_b.status_message}. "
            f"Lens B distribution is hidden; Lens A peer context still shown.",
        )
        if lens_b.is_top_coded_age:
            st.caption(
                "ℹ️ NHANES public-use files top-code age at 80. "
                "The Lens B cohort is restricted to RIDAGEYR == 80."
            )


# ---------------------------------------------------------------------------
# Percentile Delta Card
# ---------------------------------------------------------------------------

def render_percentile_delta_card(
    display_name: str,
    unit: str,
    user_value: float,
    lens_a_pct: float | None,
    lens_b_pct: float | None,
    is_lens_b_sufficient: bool,
) -> None:
    """
    Summary card showing the user's value, Lens A peer percentile,
    and Lens B optimal benchmark percentile side-by-side.
    """
    st.markdown("#### 📊 Percentile Summary")

    value_str = f"{user_value:.1f}{unit}" if unit else f"{user_value:.1f}"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(label=f"Your {display_name}", value=value_str)
    with c2:
        if lens_a_pct is not None:
            st.metric(label="Peer Percentile (Lens A)", value=f"{lens_a_pct:.1f}th")
        else:
            st.metric(label="Peer Percentile (Lens A)", value="—")
    with c3:
        if lens_b_pct is not None and is_lens_b_sufficient:
            st.metric(label="Optimal Percentile (Lens B)", value=f"{lens_b_pct:.1f}th")
        else:
            st.metric(label="Optimal Percentile (Lens B)", value="—",
                      help="Lens B has insufficient sample size for this age decade.")

    if lens_a_pct is not None and lens_b_pct is not None and is_lens_b_sufficient:
        delta = lens_b_pct - lens_a_pct
        direction = "higher" if delta > 0 else "lower"
        st.info(
            f"**The Delta Insight:** Your {display_name} sits at the "
            f"**{lens_a_pct:.1f}th percentile** among somatic peers, but at the "
            f"**{lens_b_pct:.1f}th percentile** among metabolically healthy adults — "
            f"**{abs(delta):.1f} percentage points {direction}**.",
        )


# ---------------------------------------------------------------------------
# No-data warning / parquet missing
# ---------------------------------------------------------------------------

def render_no_data_warning(parquet_path: str = "data/processed/mec_cohort_pooled.parquet") -> None:
    """
    Shown when the processed NHANES parquet file is not yet available.
    """
    st.error(
        f"**NHANES cohort data not found** at `{parquet_path}`.\n\n"
        "Run the data pipeline to build the processed tables:\n"
        "```bash\n"
        "python scripts/build_nhanes_parquet.py\n"
        "```\n"
        "Or use the **Maya demo** button in the sidebar to explore with synthetic data.",
        icon="🗄️",
    )


# ---------------------------------------------------------------------------
# Safety Disclaimer Block (§5.4)
# ---------------------------------------------------------------------------

def render_safety_disclaimer() -> None:
    """
    Mandatory safety and framing statement (PROJECT_PLAN.md §5.4).

    Displayed persistently — not hidden behind an expander.
    """
    st.markdown("---")
    st.markdown(
        """
        <div style="
            background:#f0f9ff;border-left:4px solid #0ea5e9;
            padding:12px 16px;border-radius:4px;font-size:13px;color:#0c4a6e;
        ">
        <b>⚕️ Important: Descriptive context only</b><br>
        Percentiles describe where your values sit among matched NHANES peers (Lens A) or
        among adults meeting optimal benchmark criteria (Lens B). This tool is
        <b>not a diagnosis</b>, not individual risk prediction, and not a treatment plan.<br><br>
        <b>Population:</b> US civilian non-institutionalized adults, NHANES cycles 2013–2018.
        Results do not generalise to other populations or time periods.<br><br>
        <b>Privacy:</b> No biometric or lab data is stored or transmitted. All processing
        is performed in-memory in your current session only.
        </div>
        """,
        unsafe_allow_html=True,
    )
