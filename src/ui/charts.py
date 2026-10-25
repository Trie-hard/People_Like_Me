"""
Plotly dual-lens visualizations for People Like Me (PROJECT_PLAN.md §5.1–5.3).

Public functions
----------------
dual_lens_density(...)  — Peer (Lens A) vs Optimal (Lens B) KDE density chart with
                          ADA/AHA hazard bands, user marker, and percentile annotations.
ecdf_chart(...)         — Hazen-weighted ECDF curve with user marker.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from src.engine.benchmark_cohort import LensBResult
from src.engine.matcher import PeerMatchResult


# ---------------------------------------------------------------------------
# Clinical banding specs (§5.2 — ADA HbA1c / AHA-BP authorities)
# ---------------------------------------------------------------------------

# Each band: (x_start, x_end, fill_colour, label, text_x_fraction)
# colour is an rgba string; text_x_fraction positions the label in [0,1] of band width
_HBA1C_BANDS: list[tuple[float, float, str, str]] = [
    (0.0, 5.7, "rgba(34, 197, 94, 0.10)", "Normal\n(<5.7%)"),
    (5.7, 6.5, "rgba(250, 204, 21, 0.15)", "Prediabetes\n(5.7–6.4%)"),
    (6.5, 15.0, "rgba(239, 68, 68, 0.12)", "Diabetes\n(≥6.5%)"),
]

_FPG_BANDS: list[tuple[float, float, str, str]] = [
    (0.0, 100.0, "rgba(34, 197, 94, 0.10)", "Normal\n(<100)"),
    (100.0, 126.0, "rgba(250, 204, 21, 0.15)", "Prediabetes\n(100–125)"),
    (126.0, 600.0, "rgba(239, 68, 68, 0.12)", "Diabetes\n(≥126)"),
]

_TOTAL_CHOLESTEROL_BANDS: list[tuple[float, float, str, str]] = [
    (0.0, 200.0, "rgba(34, 197, 94, 0.10)", "Desirable\n(<200)"),
    (200.0, 240.0, "rgba(250, 204, 21, 0.15)", "Borderline\n(200–239)"),
    (240.0, 600.0, "rgba(239, 68, 68, 0.12)", "High Risk\n(≥240)"),
]

_HDL_BANDS: list[tuple[float, float, str, str]] = [
    (0.0, 40.0, "rgba(239, 68, 68, 0.12)", "Low/Risk\n(<40)"),
    (40.0, 60.0, "rgba(250, 204, 21, 0.15)", "Acceptable\n(40–59)"),
    (60.0, 200.0, "rgba(34, 197, 94, 0.10)", "Optimal\n(≥60)"),
]

_LENS_A_COLOUR = "rgba(59, 130, 246, 0.85)"   # blue
_LENS_B_COLOUR = "rgba(16, 185, 129, 0.85)"   # teal/green
_LENS_A_FILL = "rgba(59, 130, 246, 0.18)"
_LENS_B_FILL = "rgba(16, 185, 129, 0.18)"
_USER_LINE_COLOUR = "rgba(220, 38, 38, 0.9)"  # red


def _hazard_bands(analyte_col: str) -> list[tuple[float, float, str, str]]:
    """Return the appropriate clinical banding list for the given analyte."""
    col = analyte_col.lower()
    if col in ("lbxgh", "hba1c", "ghb"):
        return _HBA1C_BANDS
    if col in ("lbxglu", "fpg", "glu", "glucose"):
        return _FPG_BANDS
    if col in ("lbxtc", "chol", "tchol"):
        return _TOTAL_CHOLESTEROL_BANDS
    if col in ("lbdhdd", "hdl"):
        return _HDL_BANDS
    return []


def _analyte_label(analyte_col: str) -> tuple[str, str]:
    """Return (display_name, unit_string) for an analyte column."""
    col = analyte_col.lower()
    if col in ("lbxgh", "hba1c", "ghb"):
        return "HbA1c", "%"
    if col in ("lbxglu", "fpg", "glu", "glucose"):
        return "Fasting Plasma Glucose", "mg/dL"
    if col in ("lbxtc", "chol", "tchol"):
        return "Total Cholesterol", "mg/dL"
    if col in ("lbdhdd", "hdl"):
        return "HDL-C", "mg/dL"
    return analyte_col, ""


def _kde_trace(
    values: np.ndarray,
    weights: np.ndarray,
    x_grid: np.ndarray,
    name: str,
    line_colour: str,
    fill_colour: str,
    dash: str = "solid",
) -> list[go.BaseTraceType]:
    """Return a filled KDE line trace from survey-weighted data."""
    if values.size < 3:
        return []
    w_norm = weights / weights.sum()
    # Weighted bandwidth via Scott's rule on effective sample
    n_eff = float(np.sum(w_norm) ** 2 / np.sum(w_norm ** 2))
    bw_factor = n_eff ** (-1.0 / 5.0)  # Scott's rule exponent
    weighted_std = float(np.sqrt(np.average((values - np.average(values, weights=w_norm)) ** 2, weights=w_norm)))
    bw = bw_factor * weighted_std * max(n_eff, 1) ** (1 / 5) / max(n_eff, 1) ** (1 / 5)
    # Fallback to scipy gaussian_kde with bandwidth rescaling
    try:
        kde = gaussian_kde(values, weights=w_norm)
        kde.set_bandwidth(bw_factor)
        y_grid = kde(x_grid)
    except Exception:
        return []

    fill_trace = go.Scatter(
        x=np.concatenate([x_grid, x_grid[::-1]]),
        y=np.concatenate([y_grid, np.zeros_like(y_grid)]),
        fill="toself",
        fillcolor=fill_colour,
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
        name=f"{name} fill",
    )
    line_trace = go.Scatter(
        x=x_grid,
        y=y_grid,
        mode="lines",
        line=dict(color=line_colour, width=2, dash=dash),
        name=name,
        hovertemplate=f"<b>{name}</b><br>Value: %{{x:.2f}}<extra></extra>",
    )
    return [fill_trace, line_trace]


def _add_hazard_band_shapes(
    fig: go.Figure,
    bands: list[tuple[float, float, str, str]],
    x_min: float,
    x_max: float,
) -> None:
    """Add vertical hazard band shading and labels to a Plotly figure."""
    for band_min, band_max, colour, label in bands:
        b_lo = max(band_min, x_min)
        b_hi = min(band_max, x_max)
        if b_lo >= b_hi:
            continue
        fig.add_vrect(
            x0=b_lo,
            x1=b_hi,
            fillcolor=colour,
            opacity=1.0,
            layer="below",
            line_width=0,
        )
        # Band boundary line (dashed)
        if band_min > x_min:
            fig.add_vline(
                x=band_min,
                line=dict(color="rgba(107,114,128,0.45)", width=1, dash="dot"),
            )
        # Label annotation in middle of visible band segment
        mid = (b_lo + b_hi) / 2
        top_label = label.split("\n")[0]
        sub_label = "\n".join(label.split("\n")[1:]) if "\n" in label else ""
        fig.add_annotation(
            x=mid,
            y=1.0,
            yref="paper",
            text=f"<span style='font-size:10px;color:#6b7280'>{top_label}</span>"
                 + (f"<br><span style='font-size:9px;color:#9ca3af'>{sub_label}</span>" if sub_label else ""),
            showarrow=False,
            yanchor="bottom",
            xanchor="center",
            align="center",
        )


def dual_lens_density(
    peer_match: PeerMatchResult,
    lens_b: LensBResult,
    user_value: float,
    analyte_col: str = "lbxgh",
    lens_a_pct: float | None = None,
    lens_b_pct: float | None = None,
    height: int = 480,
) -> go.Figure:
    """
    Interactive Plotly dual-lens density chart (§5.1).

    Shows Peer cohort (Lens A, blue) and Optimal Benchmark (Lens B, teal) KDE
    distributions for a single analyte, with ADA/AHA clinical hazard bands, a
    vertical marker for the user's value, and Lens A / Lens B percentile annotations.

    Parameters
    ----------
    peer_match : PeerMatchResult from query_peer_cohort
    lens_b     : LensBResult from lens_b_subset
    user_value : The user's analyte value (e.g. HbA1c = 5.8)
    analyte_col: NHANES column name for the analyte (default 'lbxgh')
    lens_a_pct : Pre-computed Lens A percentile (0–100); computed internally if None
    lens_b_pct : Pre-computed Lens B percentile (0–100); computed internally if None
    height     : Figure height in pixels

    Returns
    -------
    plotly.graph_objects.Figure
    """
    display_name, unit = _analyte_label(analyte_col)
    bands = _hazard_bands(analyte_col)

    # --- Gather Lens A data ---
    peer_df = peer_match.neighbor_df
    col_map = {c.lower(): c for c in peer_df.columns}
    a_col = col_map.get(analyte_col.lower(), analyte_col)
    w_col = col_map.get("wtmec6yr") or col_map.get("wtsaf6yr")

    a_mask = peer_df[a_col].notna() if a_col in peer_df.columns else pd.Series(False, index=peer_df.index)
    peer_vals = peer_df.loc[a_mask, a_col].to_numpy(dtype=float) if a_mask.any() else np.array([])
    peer_w = peer_df.loc[a_mask, w_col].to_numpy(dtype=float) if (a_mask.any() and w_col) else np.ones_like(peer_vals)

    # --- Gather Lens B data ---
    b_df = lens_b.benchmark_df
    b_col_map = {c.lower(): c for c in b_df.columns}
    b_col = b_col_map.get(analyte_col.lower(), analyte_col)
    bw_col = b_col_map.get("wtmec6yr") or b_col_map.get("wtsaf6yr")

    b_mask = b_df[b_col].notna() if (b_col in b_df.columns and not b_df.empty) else pd.Series(dtype=bool)
    bench_vals = b_df.loc[b_mask, b_col].to_numpy(dtype=float) if b_mask.any() else np.array([])
    bench_w = b_df.loc[b_mask, bw_col].to_numpy(dtype=float) if (b_mask.any() and bw_col) else np.ones_like(bench_vals)

    # --- X-axis range ---
    all_vals = np.concatenate([v for v in [peer_vals, bench_vals, np.array([user_value])] if v.size > 0])
    if all_vals.size == 0:
        return go.Figure().update_layout(title="No data available")
    x_min = max(float(np.percentile(all_vals, 0.5)) - 0.2, 0.0)
    x_max = float(np.percentile(all_vals, 99.5)) + 0.2
    x_grid = np.linspace(x_min, x_max, 400)

    fig = go.Figure()

    # --- Hazard band background ---
    _add_hazard_band_shapes(fig, bands, x_min, x_max)

    # --- Lens A KDE ---
    if peer_vals.size >= 3:
        a_traces = _kde_trace(peer_vals, peer_w, x_grid, f"Lens A — Peers (n={len(peer_vals)})", _LENS_A_COLOUR, _LENS_A_FILL)
        for t in a_traces:
            fig.add_trace(t)

    # --- Lens B KDE (if sufficient) ---
    if lens_b.is_sufficient and bench_vals.size >= 3:
        b_traces = _kde_trace(bench_vals, bench_w, x_grid, f"Lens B — Optimal Benchmark (n={len(bench_vals)})", _LENS_B_COLOUR, _LENS_B_FILL, dash="dash")
        for t in b_traces:
            fig.add_trace(t)
    elif not lens_b.is_sufficient:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=f"<b>Lens B:</b> {lens_b.status_message}",
            showarrow=False,
            font=dict(size=11, color="#6b7280"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d1d5db",
            borderwidth=1,
        )

    # --- User value vertical line ---
    fig.add_vline(
        x=user_value,
        line=dict(color=_USER_LINE_COLOUR, width=2.5, dash="solid"),
    )

    # --- Percentile annotations ---
    annot_parts = [f"<b>Your {display_name}: {user_value:.1f}{unit}</b>"]
    if lens_a_pct is not None:
        annot_parts.append(f"Peer percentile (A): <b>{lens_a_pct:.1f}</b>th")
    if lens_b_pct is not None and lens_b.is_sufficient:
        annot_parts.append(f"Optimal percentile (B): <b>{lens_b_pct:.1f}</b>th")
    annot_text = "<br>".join(annot_parts)

    x_frac = (user_value - x_min) / max(x_max - x_min, 1e-6)
    ax_offset = 60 if x_frac < 0.6 else -60
    fig.add_annotation(
        x=user_value,
        y=0.95,
        yref="paper",
        text=annot_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=_USER_LINE_COLOUR,
        ax=ax_offset,
        ay=-30,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=_USER_LINE_COLOUR,
        borderwidth=1.5,
        align="left",
    )

    # --- Layout ---
    x_title = f"{display_name} ({unit})" if unit else display_name
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        title=dict(
            text=f"<b>Dual-Lens Distribution — {display_name}</b>",
            font=dict(size=16),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title=x_title, range=[x_min, x_max], showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="Density", showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    return fig


def ecdf_chart(
    peer_match: PeerMatchResult,
    user_value: float,
    analyte_col: str = "lbxgh",
    lens_a_pct: float | None = None,
    height: int = 380,
) -> go.Figure:
    """
    Hazen-weighted ECDF curve (§4.3, §5.1) with user value marker.

    Shows the empirical CDF of the peer cohort (Lens A) for the given analyte,
    with the user's value marked on the curve.
    """
    display_name, unit = _analyte_label(analyte_col)

    peer_df = peer_match.neighbor_df
    col_map = {c.lower(): c for c in peer_df.columns}
    a_col = col_map.get(analyte_col.lower(), analyte_col)
    w_col = col_map.get("wtmec6yr") or col_map.get("wtsaf6yr")

    mask = peer_df[a_col].notna() if a_col in peer_df.columns else pd.Series(False, index=peer_df.index)
    if not mask.any():
        return go.Figure().update_layout(title="No analyte data in peer cohort")

    vals = peer_df.loc[mask, a_col].to_numpy(dtype=float)
    w = peer_df.loc[mask, w_col].to_numpy(dtype=float) if w_col else np.ones_like(vals)

    order = np.argsort(vals, kind="mergesort")
    y_sorted = vals[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    w_total = w_sorted.sum()
    s = (cum_w - 0.5 * w_sorted) / w_total  # Hazen

    fig = go.Figure()

    # ECDF line
    fig.add_trace(go.Scatter(
        x=y_sorted, y=s * 100,
        mode="lines",
        line=dict(color=_LENS_A_COLOUR, width=2),
        name="Peer ECDF (Lens A)",
        hovertemplate=f"{display_name}: %{{x:.2f}}{unit}<br>Percentile: %{{y:.1f}}<extra></extra>",
    ))

    # User marker
    if lens_a_pct is not None:
        fig.add_trace(go.Scatter(
            x=[user_value], y=[lens_a_pct],
            mode="markers",
            marker=dict(color=_USER_LINE_COLOUR, size=12, symbol="circle", line=dict(color="white", width=2)),
            name=f"Your value ({user_value:.1f}{unit})",
            hovertemplate=f"Your {display_name}: {user_value:.1f}{unit}<br>Peer Percentile: {lens_a_pct:.1f}th<extra></extra>",
        ))

    # Clinical band lines
    bands = _hazard_bands(analyte_col)
    for band_min, band_max, colour, label in bands:
        if vals.min() <= band_min <= vals.max():
            fig.add_vline(x=band_min, line=dict(color="rgba(107,114,128,0.4)", width=1, dash="dot"))

    fig.update_layout(
        height=height,
        title=dict(text=f"<b>Weighted ECDF — {display_name} (Peer Cohort)</b>", font=dict(size=14), x=0.0, xanchor="left"),
        xaxis=dict(title=f"{display_name} ({unit})" if unit else display_name, showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(title="Percentile", range=[0, 100], ticksuffix="th", showgrid=True, gridcolor="#f3f4f6"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=50, b=50),
    )
    return fig
