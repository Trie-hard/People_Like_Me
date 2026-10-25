"""
ReportLab 1-page Clinical Provider Brief stub (PROJECT_PLAN.md §5.3).

Generates a compact, print-ready PDF with:
- Patient biometric header
- HbA1c value, Lens A peer percentile, Lens B optimal percentile
- NHANES cohort sample sizes (Lens A k and Lens B effective n)
- Three evidence-grounded physician discussion questions

Usage
-----
    pdf_bytes = generate_pdf(profile, hba1c=5.8, lens_a_pct=76.1, lens_b_pct=94.6, ...)
    with open("brief.pdf", "wb") as f:
        f.write(pdf_bytes)

Or from Streamlit:
    st.download_button("Download PDF Brief", data=pdf_bytes, file_name="plm_brief.pdf")
"""

from __future__ import annotations

import io
from datetime import date
from typing import Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.engine.matcher import PeerMatchResult, UserProfile


# ---------------------------------------------------------------------------
# Colour palette (low-ink print-safe)
# ---------------------------------------------------------------------------
_BRAND_BLUE = colors.HexColor("#1e40af")
_BRAND_TEAL = colors.HexColor("#0d9488")
_BRAND_RED = colors.HexColor("#b91c1c")
_LIGHT_GREY = colors.HexColor("#f3f4f6")
_MID_GREY = colors.HexColor("#9ca3af")
_DARK = colors.HexColor("#111827")


# ---------------------------------------------------------------------------
# Discussion question templates (§5.3)
# ---------------------------------------------------------------------------

def _discussion_questions(
    display_name: str,
    user_value: float,
    unit: str,
    lens_a_pct: float | None,
    lens_b_pct: float | None,
    lens_b_sufficient: bool,
) -> list[str]:
    """
    Generate three targeted physician discussion questions based on clinical context.
    Template strings with light personalisation.
    """
    val_str = f"{user_value:.1f}{unit}"

    # Question 1 — Screening / monitoring frequency
    q1 = (
        f"Given a {display_name} of {val_str}, what is the recommended rescreening "
        f"interval (e.g. annual HbA1c, fasting glucose), and should we consider "
        f"additional workup (e.g. oral glucose tolerance test)?"
    )

    # Question 2 — Population context and risk trajectory
    if lens_a_pct is not None and lens_b_pct is not None and lens_b_sufficient:
        q2 = (
            f"My {display_name} is at the {lens_a_pct:.0f}th percentile among somatic peers "
            f"but at the {lens_b_pct:.0f}th percentile among metabolically healthy adults of "
            f"the same age and sex. What lifestyle or pharmacological interventions have the "
            f"strongest evidence for shifting this trajectory toward the optimal reference range?"
        )
    elif lens_a_pct is not None:
        q2 = (
            f"My {display_name} is at the {lens_a_pct:.0f}th percentile among people with "
            f"a similar somatic profile (age, BMI, blood pressure). "
            f"What interventions have the strongest evidence for improving this?"
        )
    else:
        q2 = (
            f"What lifestyle or pharmacological interventions have the strongest evidence "
            f"for improving {display_name} from {val_str} toward the optimal range?"
        )

    # Question 3 — Medication and confounders
    q3 = (
        "Are any current medications (e.g. corticosteroids, atypical antipsychotics, "
        "thiazide diuretics, beta-blockers) likely to be masking or exacerbating this "
        "result, and should we account for this when interpreting the trend?"
    )

    return [q1, q2, q3]


# ---------------------------------------------------------------------------
# PDF generator
# ---------------------------------------------------------------------------

def generate_pdf(
    user: UserProfile,
    *,
    display_name: str = "HbA1c",
    unit: str = "%",
    user_value: float = 0.0,
    lens_a_pct: float | None = None,
    lens_b_pct: float | None = None,
    lens_a_k: int = 300,
    lens_b_n: float = 0.0,
    lens_b_sufficient: bool = False,
    session_date: date | None = None,
    extra_questions: Sequence[str] = (),
    biomarker_results: Sequence[dict] | None = None,
) -> bytes:
    """
    Render a 1-page Clinical Provider Brief PDF (§5.3).

    Parameters
    ----------
    user            : UserProfile (age, sex, bmi, sbp, dbp)
    display_name    : Human-readable analyte name (e.g. 'HbA1c')
    unit            : Measurement unit string (e.g. '%' or 'mg/dL')
    user_value      : The user's analyte measurement
    lens_a_pct      : Peer cohort percentile (Lens A)
    lens_b_pct      : Optimal benchmark percentile (Lens B)
    lens_a_k        : k used in peer matching
    lens_b_n        : Effective weighted n in Lens B benchmark
    lens_b_sufficient: Whether Lens B sample meets adequacy threshold
    session_date    : Defaults to today
    extra_questions : Up to 3 additional custom discussion questions (replaces defaults)

    Returns
    -------
    bytes — raw PDF content for download or file I/O.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )

    styles = getSampleStyleSheet()
    session_str = (session_date or date.today()).strftime("%B %d, %Y")
    sex_label = "Male" if user.sex == 1 else "Female"
    val_str = f"{user_value:.1f}{unit}"

    # ---- Style definitions ----
    title_style = ParagraphStyle(
        "plm_title",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=_BRAND_BLUE,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "plm_subtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=_MID_GREY,
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "plm_section",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=_BRAND_BLUE,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "plm_body",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=_DARK,
    )
    small_style = ParagraphStyle(
        "plm_small",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=_MID_GREY,
    )
    q_style = ParagraphStyle(
        "plm_q",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        leftIndent=14,
        textColor=_DARK,
        spaceBefore=5,
        spaceAfter=2,
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("People Like Me", title_style))
    story.append(Paragraph("Clinical Provider Brief — Epidemiological Lab Contextualization", subtitle_style))
    story.append(Paragraph(f"Session date: {session_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_BRAND_BLUE, spaceAfter=8))

    # ---- Biometrics table ----
    story.append(Paragraph("Patient Biometric Summary", section_style))
    biometrics = [
        ["Age", f"{user.age:.0f} years", "Biological Sex", sex_label],
        ["BMI", f"{user.bmi:.1f} kg/m²", "Blood Pressure", f"{user.sbp:.0f}/{user.dbp:.0f} mmHg"],
        ["MAP", f"{user.map:.1f} mmHg", "Pulse Pressure", f"{user.pp:.1f} mmHg"],
    ]
    bio_table = Table(
        biometrics,
        colWidths=[1.2 * inch, 1.6 * inch, 1.4 * inch, 1.8 * inch],
    )
    bio_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _LIGHT_GREY),
        ("TEXTCOLOR", (0, 0), (-1, -1), _DARK),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LIGHT_GREY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, _MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 10))

    # ---- Biomarker matrix ----
    story.append(Paragraph("Biomarker Analysis", section_style))

    def _pct_str(p: float | None) -> str:
        return f"{p:.1f}th" if p is not None else "—"

    def _clinical_category(v: float, name: str) -> str:
        name_lc = name.lower()
        if "hba1c" in name_lc or "glyco" in name_lc:
            if v < 5.7:
                return "Normal"
            if v < 6.5:
                return "Prediabetes"
            return "Diabetes (ADA)"
        if "glucose" in name_lc or "fpg" in name_lc:
            if v < 100:
                return "Normal"
            if v < 126:
                return "Prediabetes"
            return "Diabetes (ADA)"
        if "cholesterol" in name_lc or "tchol" in name_lc or name_lc == "tc":
            if v < 200:
                return "Desirable (AHA)"
            if v < 240:
                return "Borderline High"
            return "High Risk"
        if "hdl" in name_lc:
            if v >= 60:
                return "Optimal (AHA)"
            if v >= 40:
                return "Acceptable"
            return "Low / Risk Factor"
        return "—"

    biomarker_data = [
        ["Biomarker", "Measured Value", "Lens A Percentile", "Lens B Percentile", "Clinical Category"],
    ]
    if biomarker_results:
        for bm in biomarker_results:
            bm_name = bm.get("display_name", "Biomarker")
            bm_unit = bm.get("unit", "")
            bm_val = bm.get("value", 0.0)
            bm_v_str = f"{bm_val:.1f}{bm_unit}" if bm_unit else f"{bm_val:.1f}"
            bm_a = bm.get("lens_a_pct")
            bm_b = bm.get("lens_b_pct")
            bm_suff = bm.get("lens_b_sufficient", False)
            bm_cat = _clinical_category(bm_val, bm_name)
            biomarker_data.append([
                bm_name,
                bm_v_str,
                _pct_str(bm_a),
                _pct_str(bm_b) if bm_suff else "—",
                bm_cat,
            ])
    else:
        category = _clinical_category(user_value, display_name)
        biomarker_data.append([
            display_name,
            val_str,
            _pct_str(lens_a_pct),
            _pct_str(lens_b_pct) if lens_b_sufficient else "—",
            category,
        ])
    bm_table = Table(
        biomarker_data,
        colWidths=[1.0 * inch, 1.15 * inch, 1.25 * inch, 1.25 * inch, 1.55 * inch],
    )
    bm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _BRAND_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, _MID_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT_GREY]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(bm_table)
    story.append(Spacer(1, 6))

    # Cohort sizes footnote
    lens_b_label = (
        f"Lens B: n={lens_b_n:.0f} (effective weighted)"
        if lens_b_sufficient
        else "Lens B: Insufficient sample (see below)"
    )
    story.append(Paragraph(
        f"NHANES cohort sizes — Lens A: k={lens_a_k} matched peers · {lens_b_label}",
        small_style,
    ))
    story.append(Spacer(1, 10))

    # ---- Physician Discussion Questions ----
    story.append(Paragraph("Physician Discussion Questions", section_style))
    questions_list = (
        list(extra_questions)[:3]
        if extra_questions
        else _discussion_questions(display_name, user_value, unit, lens_a_pct, lens_b_pct, lens_b_sufficient)
    )
    for idx, q in enumerate(questions_list[:3], start=1):
        story.append(Paragraph(f"<b>{idx}.</b> {q}", q_style))
    story.append(Spacer(1, 10))

    # ---- Safety / Legal Footer ----
    story.append(HRFlowable(width="100%", thickness=0.5, color=_MID_GREY, spaceAfter=4))
    story.append(Paragraph(
        "<b>⚕ Important:</b> This brief provides descriptive epidemiological context only — "
        "it is <b>not a diagnosis</b>, not individual risk prediction, and not a treatment plan. "
        "Percentiles reflect NHANES 2013–2018 US survey data. "
        "No patient data is stored or transmitted beyond this session.",
        small_style,
    ))

    doc.build(story)
    return buf.getvalue()
