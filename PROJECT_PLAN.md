# People Like Me: Comprehensive Project Specification & Architecture

**Tagline:** *Your lab results, in epidemiological context.*  
**System Classification:** Evidence-Based Health Data Product & Patient Contextualization Engine  
**Target Deployment (MVP):** Streamlit application, Plotly visualization layer, ReportLab 1-page clinical brief (no separate API server in v1).

**Documentation:** This file is the **canonical** engineering specification. [PEOPLE_LIKE_ME.md](PEOPLE_LIKE_ME.md) is narrative/product copy; when they disagree, follow `PROJECT_PLAN.md`.

---

## 1. Executive Summary & Core Philosophy

### 1.1 The Problem
When patients receive standard clinical laboratory reports (e.g., Quest, Labcorp, hospital EHRs), values are presented against monolithic reference intervals that treat nearly all adults identically (e.g., Fasting Plasma Glucose < 100 mg/dL = Normal; 100–125 mg/dL = Impaired/Prediabetes; $\ge$ 126 mg/dL = Diabetes). 

While clinically standardized, these static binary flags fail in two critical ways:
1. **Lack of Population Grounding:** A 45-year-old woman with a BMI of 31 and blood pressure of 128/82 mmHg who receives a fasting glucose result of 104 mg/dL has no idea whether she is an extreme outlier or typical among people with her physiological baseline.
2. **The "Sick Population" Trap (Geoffrey Rose Paradox):** Conversely, if an app simply tells her *"You are at the 50th percentile among peers like you,"* she risks false reassurance. In the United States, over 40% of adults with her demographic and biometric profile have dysglycemia. Being "average" in an unhealthy population does not equal optimal metabolic health.

### 1.2 The Solution
**People Like Me** provides personalized, population-representative lab contextualization using 6 continuous years of CDC NHANES (National Health and Nutrition Examination Survey) microdata. 

It introduces a **Dual-Lens Architectural Framework**:
* **Lens A (Peer Cohort):** Shows how the user compares to their nearest demographic and biometric neighbors (same sex, similar age, BMI, and hemodynamics).
* **Lens B (Optimal Health Benchmark):** Juxtaposes the peer distribution against an age- and sex-matched cohort of metabolically healthy adults (normal BMI, normotensive, non-diabetic).

This dual framing eliminates complacency while replacing anxiety with objective, non-stigmatizing, epidemiological clarity.

---

## 2. Clinical & Epidemiological Foundations

```
                        ┌────────────────────────────────────────────────┐
                        │              User Biomarkers                   │
                        │    Age: 45 | Sex: F | BMI: 31 | BP: 128/82     │
                        └───────────────────────┬────────────────────────┘
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        ▼                                                ▼
         ┌──────────────────────────────┐                ┌──────────────────────────────┐
         │     LENS A: Peer Cohort      │                │  LENS B: Optimal Benchmark   │
         │  k-NN on Age, BMI, MAP, PP   │                │ Sex + age decade; BMI 18.5–25  │
         │ (Demographic & Biometric)    │                │  BP < 120/80; non-diabetic   │
         └──────────────┬───────────────┘                └──────────────┬───────────────┘
                        │                                                │
                        ▼                                                ▼
         ┌──────────────────────────────┐                ┌──────────────────────────────┐
         │ • Matched Peer Percentile    │                │ • Healthy Cohort Percentile  │
         │ • Peer Prediabetes: 41.2%    │                │ • Healthy Prediabetes: 8.4%  │
         │ • Peer Medicated: 28.1%      │                │ • Peer vs Optimal Gap Delta  │
         └──────────────┬───────────────┘                └──────────────┬───────────────┘
                        │                                                │
                        └───────────────────────┬────────────────────────┘
                                                │
                                                ▼
                        ┌────────────────────────────────────────────────┐
                        │     Clinical Provider Brief & Action Prompts   │
                        │     1-Page PDF Export for Shared Decisions     │
                        └────────────────────────────────────────────────┘
```

### 2.1 The Dual-Lens Paradigm
1. **Peer Contextualization (Local Manifold):** Identifies the empirical distribution of $k$ nearest neighbors in NHANES. Tells the user where they sit relative to biological reality for their current somatic baseline.
2. **Optimal Reference Grounding (Target Manifold):** Subsets NHANES participants who meet **Lens B eligibility** (all required; see below). Percentiles and densities use this fixed reference pool, not k-NN.
3. **The Delta Insight:** Demonstrates the gap between peer reality and optimal biology, turning abstract clinical advice into an intuitive visual distance.

**Lens B eligibility (canonical rule):** Participant $i$ is in the optimal benchmark pool iff:
* Same **biological sex** as the user (`RIAGENDR`).
* **Age decade** matches the user (e.g. user age 45 → ages $[40, 49]$; users with `RIDAGEYR == 80` use the $[80, 80]$ top-coded band and UI notes top-coding).
* BMI $18.5 \le \text{BMXBMI} < 25$ $\text{kg/m}^2$.
* Measured BP at exam: SBP $< 120$ mmHg and DBP $< 80$ mmHg (after BP cleaning protocol; medication status is **not** used to exclude—treated individuals with low measured BP remain eligible).
* **Non-diabetic** by survey + lab:
  * `DIQ010 == 2` (no prior diabetes diagnosis), **and**
  * if `LBXGH` present: `LBXGH < 5.7`%; if `LBXGLU` present (fasting pool only): `LBXGLU < 100` mg/dL; if both missing, exclude from Lens B glycemia summaries for that analyte.

**Lens B minimum sample size:** If weighted $n < 200$ in the user’s sex × age-decade × pool (MEC or fasting), do not render a KDE for Lens B; show *“Insufficient benchmark sample”* and widen age band to full sex-stratified optimal pool only in **offline validation**, not in production UX without explicit labeling.

### 2.2 Confounder Management & Pharmacological Masking
A significant percentage of US adults with high BMI or hypertension have "normal" total cholesterol (< 200 mg/dL) or HbA1c (< 5.7%) solely because they take statins, metformin, or ACE inhibitors/ARBs.
* **The Diagnostic Iceberg:** Decomposes dysglycemia in the matched cohort into:
  * Diagnosed Diabetes (`DIQ010 == 1`)
  * Undiagnosed Diabetes (`DIQ010` in $\{2, 3, \text{missing}\}$ AND (`LBXGH >= 6.5` OR `LBXGLU >= 126`))
  * Prediabetes (`LBXGH` between 5.7–6.4% OR `LBXGLU` between 100–125 mg/dL), excluding those classified as diabetes above
  * Normoglycemia
* **`DIQ010` coding:** `1` = Yes (diagnosed diabetes); `2` = No; `3` = Borderline → treat as **not diagnosed** for iceberg “diagnosed” counts, but flag borderline separately in telemetry when `DIQ010 == 3`.
* **Medication Telemetry:** Tracks and explicitly surfaces the proportion of matched peers currently receiving pharmacotherapy (`RXQ_RX`).

---

## 3. Data Architecture & Ingestion Pipeline

### 3.1 Data Source & Multi-Cycle Pooling
The engine pools three continuous NHANES cycles:
* **Cycle H (2013–2014)**
* **Cycle I (2015–2016)**
* **Cycle J (2017–2018)**

*(Pre-pandemic 2017–March 2020 cycle is excluded due to sampling interruption and anomalous weighting adjustments).*

Pooling 3 cycles increases statistical power in the fasting subsample from $\approx 3,100$ to $\approx 8,400$ adults, ensuring dense local neighborhoods in continuous metric space.

### 3.2 Dual-Pool Survey Architecture
NHANES employs a complex, multistage, stratified probability sampling design. Analytes belong to different subsamples with distinct statistical weights. Pooling them incorrectly introduces massive selection bias.

| Pipeline Pool | Target Analytes | NHANES Files | Primary Weight Column | Multi-Cycle Weight Formula |
|---|---|---|---|---|
| **MEC Full Sample Pool** | HbA1c (`LBXGH`), Total Cholesterol (`LBXTC`), HDL (`LBDHDD`), Systolic/Diastolic BP, BMI | `DEMO`, `BMX`, `BPX`, `GHB`, `TCHOL`, `HDL`, `DIQ`, `RXQ_RX` | `WTMEC2YR` | $\text{WTMEC6YR} = \frac{\text{WTMEC2YR}}{3}$ |
| **Fasting Subsample Pool** | Fasting Plasma Glucose (`LBXGLU`); *pipeline also ingests* `LBXTR`, LDL for future use | `GLU`, `TRIGLY` + MEC files above | `WTSAF2YR` | $\text{WTSAF6YR} = \frac{\text{WTSAF2YR}}{3}$ |

*MVP UI biomarkers (§5.1):* fasting glucose, HbA1c, total cholesterol, HDL-C. Triglycerides and LDL stay in the harmonized fasting table for Phase 2+ analytics only until added to the product surface.

### 3.3 Variable Dictionary & Protocol Rules

```
NHANES Variable Harmonization Spec:
├── DEMO_*:
│   ├── SEQN: Participant identification number (Merge Key)
│   ├── RIDAGEYR: Age in years (Filter: >= 20, Top-coded at 80 in public files)
│   ├── RIAGENDR: Biological sex (1 = Male, 2 = Female)
│   ├── RIDEXPRG: Pregnancy status at exam (Filter: Drop if == 1)
│   ├── RIDRETH3: Detailed race/ethnicity code (Audit feature)
│   ├── INDFMPIR: Ratio of family income to poverty (Audit feature)
│   ├── SDMVSTRA: Masked variance pseudo-stratum (Survey design)
│   └── SDMVPSU: Masked variance pseudo-PSU (Survey design)
├── BMX_*:
│   ├── BMXBMI: Body Mass Index (kg/m^2)
│   ├── BMXWT: Weight (kg)
│   └── BMXHT: Standing Height (cm)
├── BPX_*:
│   ├── BPXSY1, BPXSY2, BPXSY3: Systolic Blood Pressure readings 1, 2, 3
│   └── BPXDI1, BPXDI2, BPXDI3: Diastolic Blood Pressure readings 1, 2, 3
│   └── Cleaning Protocol: Discard reading 1 (white-coat attenuation).
│       Average valid readings 2 and 3. If only one reading exists, use it.
├── GLU_*:
│   ├── LBXGLU: Fasting plasma glucose (mg/dL)
│   ├── WTSAF2YR: Fasting subsample weight
│   └── PHAFSTHR: Fasting duration in hours (Filter: >= 8.0 and < 24.0)
├── GHB_*:
│   └── LBXGH: Glycohemoglobin / HbA1c (%)
├── TCHOL_* & HDL_*:
│   ├── LBXTC: Total Cholesterol (mg/dL)
│   └── LBDHDD: Direct HDL-Cholesterol (mg/dL)
└── DIQ_*:
    ├── DIQ010: "Doctor told you have diabetes" (1=Yes, 2=No, 3=Borderline)
    ├── DIQ050: Currently taking insulin
    └── DIQ070: Currently taking blood sugar pills
```

### 3.4 Data Validation Contract (Pandera Schema)
All raw files pass through strict validation boundaries before ingestion:
* Missing values in matching coordinates (`age`, `bmi`, `sbp`, `dbp`) are dropped.
* Physiologically implausible outliers are scrubbed ($\text{BMI} \notin [12, 80]$, $\text{SBP} \notin [60, 260]$, $\text{DBP} \notin [30, 150]$).
* All survey weights must be strictly positive ($W_i > 0$).

### 3.5 Survey Inference Policy (Pooling & Validation)
* **Pooled point weights:** For descriptive percentiles and prevalences among pooled cycles H–J, use $\text{WTMEC6YR} = \text{WTMEC2YR}/3$ and $\text{WTSAF6YR} = \text{WTSAF2YR}/3$ per [NCHS guidance on combining NHANES cycles](https://wwwn.cdc.gov/nchs/nhanes/tutorials/default.aspx).
* **Design variables:** Retain `SDMVSTRA` and `SDMVPSU` in processed tables for offline variance estimation; MVP UI percentiles are **survey-weighted among the selected neighbor set or Lens B subset**, not full-population marginal estimates.
* **Validation coverage (§6):** Primary metric is **survey-weighted** coverage: each leave-one-out unit contributes with its pooled weight; report unweighted coverage as a secondary diagnostic.
* **Documentation:** User-facing copy must say percentiles are *among matched NHANES peers* (Lens A) or *among NHANES adults meeting optimal benchmark criteria* (Lens B), not personalized risk predictions.

---

## 4. Mathematical Engine & Algorithmic Matching

### 4.1 Hemodynamic Decoupling
Systolic and Diastolic Blood Pressure have an empirical collinearity of $r \approx 0.70 - 0.80$. Placing both directly into Euclidean distance spaces skews the distance metric by double-counting vascular resistance.

To resolve this, the engine converts raw blood pressure into two physiologically independent orthogonal components:
1. **Mean Arterial Pressure (MAP):** Measures steady-state perfusion resistance:
   $$\text{MAP} = \text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})$$
2. **Pulse Pressure (PP):** Measures pulsatile load, stroke volume, and large-artery stiffness:
   $$\text{PP} = \text{SBP} - \text{DBP}$$

### 4.2 Distance Space & Metric Formulation
Matching operates over the coordinate vector:
$$\mathbf{x} = \begin{bmatrix} \text{Age}, & \text{BMI}, & \text{MAP}, & \text{PP} \end{bmatrix}^T$$

Exact stratification is applied to **Biological Sex** ($\text{Pool} = \{i : \text{Sex}_i == \text{Sex}_{\text{user}}\}$).

Within the stratified pool, distance is calculated via **regularized Whitened Mahalanobis Distance**:
$$D_M(\mathbf{x}_{\text{user}}, \mathbf{x}_i) = \sqrt{(\mathbf{x}_{\text{user}} - \mathbf{x}_i)^T \mathbf{\Sigma}_\lambda^{-1} (\mathbf{x}_{\text{user}} - \mathbf{x}_i)}$$
Where $\mathbf{\Sigma}_\lambda = \mathbf{\Sigma} + \lambda \mathbf{I}$ and $\mathbf{\Sigma}$ is the sample covariance of the stratified pool. Default $\lambda = 10^{-3} \times \mathrm{tr}(\mathbf{\Sigma})/4$ (tunable in `src/config.py`). Use Cholesky of $\mathbf{\Sigma}_\lambda^{-1}$ for production queries.

**Covariance fit policy:** Production: fit $\mathbf{\Sigma}$ once per sex on the full pooled MEC (or fasting) pool. LOO validation (§6): refit $\mathbf{\Sigma}$ on the stratified pool **excluding** holdout $i$.

### 4.2.1 Neighborhood Size ($k$)
* **Default:** $k = 300$ nearest neighbors by $D_M$ within the same sex.
* **Allowed range:** $k \in [100, 500]$ (advanced setting / sensitivity analysis only in v1).
* **Adaptive guard:** If the $k$-th neighbor has $D_M > 2.5$, cap displayed MQI at “low support” regardless of $\bar{d}_k$ thresholds.

### 4.3 Survey-Weighted Percentile Estimation (Local Neighbor ECDF)
Percentiles are computed from the **neighbor set only** (not the full national weighted marginal). This is intentional for interpretability; compare against Baseline 2 (quantile regression) in validation and optional UI footnotes.
For a query value $y^*$ (e.g. HbA1c = 5.8%) evaluated against neighbor values $\{y_1, \dots, y_k\}$ with associated NHANES survey weights $\{w_1, \dots, w_k\}$:

1. Sort the neighbor indices such that $y_{(1)} \le y_{(2)} \le \dots \le y_{(k)}$.
2. Compute the continuity-corrected empirical cumulative survey weight (Hazen's rule):
   $$S_{(j)} = \frac{\sum_{m=1}^{j} w_{(m)} - 0.5 \cdot w_{(j)}}{\sum_{m=1}^{k} w_{(m)}}$$
3. Evaluate the continuous percentile rank via linear interpolation:
   $$\text{Percentile}(y^*) = 100 \times \text{Interp}(y^*, \{y_{(j)}\}, \{S_{(j)}\})$$
   *Values outside the neighbor support are bounded to $[0.1, 99.9]$.*

### 4.4 Demographic Bounding Box & Support Density Index
To guarantee algorithmic transparency, the system computes the demographic envelope for the matched cohort:
* $\text{Age}_{\text{IQR}} = [Q_{25}, Q_{75}]$, $\text{Median}(\text{Age})$
* $\text{BMI}_{\text{IQR}} = [Q_{25}, Q_{75}]$, $\text{Median}(\text{BMI})$
* $\text{BP}_{\text{IQR}} = [\text{SBP}_{Q25}/\text{DBP}_{Q25}, \text{SBP}_{Q75}/\text{DBP}_{Q75}]$

**Match Quality Index (MQI):**
Calculates the mean Mahalanobis radius $\bar{d}_k = \frac{1}{k}\sum_{i=1}^k D_M(\mathbf{x}_{\text{user}}, \mathbf{x}_i)$.
* $\bar{d}_k < 1.2$: **High Confidence** (Dense manifold support)
* $1.2 \le \bar{d}_k < 2.0$: **Moderate Confidence**
* $\bar{d}_k \ge 2.0$: **Low Support Warning** (User has an unusual profile, e.g. age 80 [top-coded] with BMI 48; display sparsity caution).

---

## 5. Product Features & User Experience

### 5.1 User Journey
```
[ Step 1: Baseline Demographics ]
├── Age (years), Biological Sex (M/F)
├── Height & Weight (calculates BMI automatically)
└── Blood Pressure (Systolic & Diastolic)
      │
      ▼
[ Step 2: Lab Marker Input (Optional / Multi-select) ]
├── Fasting Plasma Glucose (mg/dL)
├── Glycohemoglobin / HbA1c (%)
└── Total Cholesterol & HDL-C (mg/dL)
      │
      ▼
[ Step 3: Dual-Lens Visualization Dashboard ]
├── Interactive Dual-Distribution Density Plot (Peer vs. Optimal)
├── Interactive Hazen-Weighted ECDF Curve with User Marker
├── Clinical Hazard Banding (ADA / AHA / ACC reference zones)
├── Peer Demographic Bounding Box Card (Explainable k-NN)
└── Confounder Panel (% on Statins, % Undiagnosed Dysglycemia)
      │
      ▼
[ Step 4: Metabolic Trajectory Modeler ("Counterfactual Lens") ]
└── Sliders for target BMI / Blood pressure showing cross-sectional cohort shift
    (Mandatory copy: "Association in other NHANES adults—not a forecast of your future labs.")
      │
      ▼
[ Step 5: Export Clinical Appointment Brief ]
└── Download 1-Page Provider SBAR PDF with targeted discussion questions
```

### 5.2 Clinical Banding Specifications

| Biomarker | Optimal / Normal | Borderline / Prediabetes | High Risk / Diabetes | Clinical Authority |
|---|---|---|---|---|
| **Fasting Glucose** | $< 100\text{ mg/dL}$ | $100 - 125\text{ mg/dL}$ | $\ge 126\text{ mg/dL}$ | American Diabetes Association (ADA) |
| **HbA1c** | $< 5.7\%$ | $5.7\% - 6.4\%$ | $\ge 6.5\%$ | American Diabetes Association (ADA) |
| **Total Cholesterol** | $< 200\text{ mg/dL}$ | $200 - 239\text{ mg/dL}$ | $\ge 240\text{ mg/dL}$ | NCEP ATP III / AHA |
| **Blood Pressure** | $< 120/80\text{ mmHg}$ | $120-129/<80$ (Elevated), $130-139/80-89$ (Stage 1) | $\ge 140/90\text{ mmHg}$ (Stage 2) | ACC/AHA 2017 Guidelines |

### 5.3 The Clinical Provider Brief (1-Page PDF)
Engineered for efficient provider review during a standard 15-minute primary care visit:
* **Header:** Patient biometric summary, session date, and NHANES cohort sample sizes.
* **Biomarker Matrix:** Measured values, Peer Percentile, Optimal Cohort Percentile, and Clinical Category.
* **Epidemiological Context:** Cohort prevalence of metabolic syndrome, prediabetes, and pharmacotherapy.
* **Physician Discussion Questions:** Three evidence-grounded questions tailored to the patient's specific lab position (e.g., screening recommendations, lifestyle intervention targets, re-testing frequency).

### 5.4 Safety, Ethics, and Regulatory Framing
* **Descriptive context only:** Not a diagnosis, not treatment advice, not individual risk prediction.
* **Population:** US civilian NHANES 2013–2018; not UK/global reference ranges.
* **Privacy:** No server-side persistence of biometrics or labs in MVP (in-memory session only).
* **Counterfactual sliders:** Cross-sectional cohort comparison only; prohibit causal wording in UI and PDF.

---

## 6. Rigorous Statistical Validation Suite

To ensure research-grade integrity, the engine is evaluated using a formal statistical validation battery:

### 6.1 Prediction Interval Calibration (Leave-One-Out)
For every adult $i$ in the benchmark evaluation set:
1. Exclude individual $i$.
2. Form their $k$-NN neighborhood.
3. Compute the weighted prediction interval at nominal levels $\alpha \in \{0.50, 0.80, 0.90, 0.95\}$.
4. Measure **survey-weighted** empirical coverage (primary):
   $$\text{Coverage}(\alpha) = \frac{\sum_{i=1}^N w_i \mathbb{I}\left(y_i \in [L_i(\alpha), U_i(\alpha)]\right)}{\sum_{i=1}^N w_i}$$
   *Target: Weighted coverage within $\pm 1.5\%$ of nominal $\alpha$ on primary endpoints (HbA1c on MEC pool; fasting glucose on fasting pool). Report unweighted coverage alongside.*

### 6.2 Scoring Rules (Winkler Score & CRPS)
Coverage alone can be achieved by arbitrarily widening intervals. To measure true information gain, we compute the **Winkler Interval Score**:
$$\text{Score}_{\alpha}(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)$$

### 6.3 Baseline Comparative Benchmarking
The matching engine is validated against three formal baselines:
1. **Baseline 0 (Null Population):** Unconditional distribution of all US adults.
2. **Baseline 1 (Demographic Binning):** Same age group (10-year bucket) and sex.
3. **Baseline 2 (Linear Quantile Regression):** Parametric quantile regression conditioned on Age, BMI, and BP.
4. **People Like Me (Regularized Mahalanobis k-NN):** Must demonstrate lower **paired** Winkler scores at 90% nominal coverage vs. Baselines 0–1 on the same holdout units.

**Significance / reporting:** Pre-specify primary biomarkers (HbA1c, LBXGLU). Use paired differences in mean Winkler score across LOO folds; bootstrap by `SDMVPSU` strata where feasible. “Significant” = 95% CI of mean improvement excludes 0 for the primary endpoint.

**Medication sensitivity (offline):** Recompute Lens A prevalences with and without peers on glucose/lipid-lowering therapy (`RXQ_RX`); document direction of bias in technical appendix.

### 6.4 Algorithmic Fairness & Subgroup Equity Audit
The engine executes automated subgroup fairness audits:
* Calibration and coverage verified across:
  * Race / Hispanic Origin (`RIDRETH3`: Non-Hispanic White, Non-Hispanic Black, Mexican American, Other Hispanic, Non-Hispanic Asian).
  * Socioeconomic status (`INDFMPIR` quartiles).
  * Age brackets ($20-39, 40-59, 60-79, 80+$).
* Discrepancies in **weighted** coverage $> 3$ percentage points between subgroups are flagged, surfaced in documentation, and displayed as humility notes in the user interface.
* **Minimum cell size:** Suppress subgroup UI flags when effective weighted $n < 50$; still log in validation reports.

---

## 7. Technical Implementation & Directory Layout

```
People_Like_Me/
├── README.md                           # System overview & setup guide
├── PROJECT_PLAN.md                     # This master architectural blueprint
├── pyproject.toml                      # Modern Python packaging configuration
├── requirements.txt                    # Pinned core production dependencies
├── data/
│   ├── raw/                            # Auto-downloaded NHANES .XPT files
│   │   ├── 2013_2014/
│   │   ├── 2015_2016/
│   │   └── 2017_2018/
│   ├── processed/                      # Harmonized, cleaned Parquet tables
│   │   ├── mec_cohort_pooled.parquet
│   │   └── fasting_cohort_pooled.parquet
│   └── metadata/                       # Variable schemas & codebooks
│       └── nhanes_schema.yaml
├── src/
│   ├── __init__.py
│   ├── data_pipeline/                  # Ingestion, Harmonization & Pandera Validation
│   │   ├── __init__.py
│   │   ├── downloader.py               # Asynchronous CDC XPT scraper
│   │   ├── cleaner.py                  # BP averaging, fasting filters, outlier removal
│   │   ├── harmonizer.py               # Multi-cycle SEQN merging & weight scaling
│   │   └── contracts.py                # Pandera DataFrame schemas
│   ├── engine/                         # Mathematical & Statistical Matching Core
│   │   ├── __init__.py
│   │   ├── hemodynamics.py             # MAP, Pulse Pressure transformations
│   │   ├── matcher.py                  # Stratified Whitened Mahalanobis k-NN
│   │   ├── percentiles.py              # Hazen survey-weighted ECDF & CDF interpolation
│   │   └── benchmark_cohort.py         # Optimal health reference cohort generator
│   ├── validation/                     # Validation & Fairness Auditing
│   │   ├── __init__.py
│   │   ├── coverage_evaluator.py       # Leave-one-out prediction interval testing
│   │   ├── winkler_score.py            # Interval sharpness scoring vs. baselines
│   │   └── fairness_audit.py           # Demographic subgroup parity analyzer
│   ├── ui/                             # Presentation Layer (Streamlit / Components)
│   │   ├── __init__.py
│   │   ├── components.py               # Cards, bounding boxes, telemetry indicators
│   │   ├── charts.py                   # Plotly Dual-Lens density & ECDF visualizers
│   │   └── pdf_generator.py            # Clinical Provider Brief generator (ReportLab)
│   └── config.py                       # Global application settings & clinical cutoffs
├── tests/                              # Comprehensive Unit & Integration Tests
│   ├── test_hemodynamics.py
│   ├── test_matcher.py
│   ├── test_percentiles.py
│   └── test_data_pipeline.py
└── app.py                              # Main application entry point
```

---

## 8. Development Roadmap (Production Engineering Sprint)

### Phase 0: Repository Bootstrap & Vertical Slice Definition
* Add `pyproject.toml`, `app.py` (Streamlit entry), and `src/config.py` (defaults: $k=300$, $\lambda$, clinical cutoffs).
* **MVP vertical slice:** HbA1c on MEC pool only — downloader → parquet → matcher → dual Lens chart → PDF stub.
* Golden integration test: fixed `SEQN` list per cycle for merge/weight sanity.

### Phase 1: Data Infrastructure & Pipeline Verification
* Build automated CDC NHANES multi-cycle downloader with SHA-256 verification.
* Implement cleaning pipelines: blood pressure protocol (discard reading 1, average 2 & 3), fasting compliance filter (`PHAFSTHR >= 8`), weight scaling ($\text{Weight} / 3$).
* Write Pandera data contracts and export `mec_cohort_pooled.parquet` and `fasting_cohort_pooled.parquet`.

### Phase 2: Core Algorithmic Engine & Metric Space
* Implement `hemodynamics.py` (MAP and Pulse Pressure conversions).
* Implement `matcher.py` (Sex-stratified regularized Mahalanobis; $k=300$ default).
* Implement `percentiles.py` (Hazen continuity-corrected weighted percentiles with boundary safety).
* Implement `benchmark_cohort.py` (Optimal Health reference cohort extractor).

### Phase 3: Statistical Validation & Subgroup Audit
* Run leave-one-out cross-validation on 8,000+ records to compute empirical coverage (50%, 80%, 90%, 95%).
* Compute Winkler scores against Unconditional and Age/Sex baselines.
* Run subgroup fairness audit across race/ethnicity and `INDFMPIR` **quartiles** (aligned with §6.4).

### Phase 4: UI, Dual-Lens Visualizations, & PDF Brief Engine
* Build Streamlit UI with intuitive biological inputs.
* Construct interactive Plotly charts with shaded ADA/AHA clinical bands, dual density curves (Peer vs. Optimal), and Hazen ECDF curves.
* Build Demographic Bounding Box component (displaying Q1–Q3 ranges of matched peers).
* Build ReportLab 1-page Clinical Provider Brief PDF generator with automated discussion prompts.

### Phase 5: Hardening, Safety Panels, & Production Deployment
* Embed safety panels per §5.4 (descriptive vs. diagnostic; US NHANES era; privacy; counterfactual non-causal copy).
* Write complete automated test suite (`pytest`) covering pipeline, math, and UI states.
* Containerize application (`Dockerfile`) and deploy to cloud runtime.
