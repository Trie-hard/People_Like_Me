# People Like Me: Master Project Specification

> **Note:** [PROJECT_PLAN.md](PROJECT_PLAN.md) is the canonical engineering specification (algorithms, survey policy, validation). This document is product narrative and demo storytelling; if anything conflicts, use `PROJECT_PLAN.md`.

**Tagline:** *Your lab results, in epidemiological context.*  
**System Classification:** Evidence-Based Health Data Product & Patient Contextualization Engine  

---

## 1. Executive Summary & Core Concept

When patients receive routine laboratory test results (e.g., fasting blood glucose, HbA1c, lipid panels), their numbers are typically compared against monolithic, population-wide reference ranges that treat a 22-year-old athlete and an 80-year-old sedentary adult identically. A single binary label—"Normal" or "High"—provides zero personal context, inducing either unnecessary anxiety or false reassurance.

**People Like Me** translates static clinical laboratory results into intuitive, population-grounded epidemiological distributions. By querying de-identified, nationally representative microdata from the CDC's **National Health and Nutrition Examination Survey (NHANES)**, the engine identifies real individuals who share the user's biological sex, age, body mass index (BMI), and hemodynamic profile. It then reveals where the user's lab values sit along the empirical distribution of their peers.

### The Central Innovation: The "Dual-Lens" Architecture
A fatal flaw of naive peer matching is the **"Sick Population Trap"** (Geoffrey Rose's epidemiological paradox). In the United States, over 42% of adults are obese and nearly 50% have diabetes or prediabetes. If an overweight user with elevated fasting glucose (e.g., 104 mg/dL) is told they are at the "50th percentile among people like them," they may conclude that their glucose is safe or optimal.

To solve this, **People Like Me** introduces a simultaneous **Dual-Lens Framework**:
1. **Lens A (The Peer Cohort):** The empirical distribution of the user's nearest demographic and biometric neighbors.
2. **Lens B (The Optimal Health Benchmark):** An age- and sex-matched cohort of adults who have achieved optimal metabolic health (normal BMI $18.5 - 24.9\text{ kg/m}^2$, normotensive BP $< 120/80\text{ mmHg}$, non-diabetic).

Juxtaposing these two distributions replaces binary panic with constructive clarity: the user sees that while their value is common among peers with their current physical profile, a distinct metabolic gap exists compared to optimal health.

---

## 2. The Anchor Persona: The Demo Story

> **Maya, 45, BMI 31, Blood Pressure 128/82 mmHg**  
> Maya just received a routine fasting plasma glucose result of **104 mg/dL**. Her lab portal flagged it in red as *"Slightly High / Impaired Fasting Glucose"*. 
>
> Anxious, she opens **People Like Me**:
> * **Lens A (Her Peers):** Among 300 matched female peers (median age 45, median BMI 31, median BP 126/82), Maya sits at the **52nd percentile**. In this cohort, **41%** meet criteria for prediabetes or diabetes.
> * **Lens B (Optimal Health Benchmark):** Among 45-year-old women with normal BMI and normal blood pressure, Maya sits at the **88th percentile**, where only **8%** have elevated blood glucose.
> * **Transparency Panel:** The app explicitly reports: *"Your peers are aged 42–48, have BMIs from 29.5 to 32.8, and blood pressure from 122/78 to 134/86 mmHg. Note: 28% of adults in this group take cholesterol- or glucose-regulating medication."*
> * **Actionable Output:** Maya exports a 1-page **Clinical Provider Brief** to bring to her physician, equipped with three targeted questions for shared decision-making.

---

## 3. Core Feature Architecture

### 3.1 Input Protocol
* **Biometrics:** Age (years), Biological Sex (Male/Female), Height & Weight (calculates BMI automatically), Systolic and Diastolic Blood Pressure (mmHg).
* **Target Biomarkers (User provides one or more):**
  * Fasting Plasma Glucose ($\text{mg/dL}$)
  * Glycohemoglobin / HbA1c ($\%$)
  * Total Cholesterol & HDL-C ($\text{mg/dL}$)

### 3.2 Dual-Lens Interactive Visualizer
* **Distribution Overlays:** Visualizes smooth kernel density estimates and empirical cumulative distributions for both the **Peer Cohort** and the **Optimal Health Benchmark**.
* **Clinical Hazard Shading:** Shaded reference bands reflecting consensus guidelines:
  * *Fasting Glucose (ADA):* Normal $< 100\text{ mg/dL}$, Prediabetes $100–125\text{ mg/dL}$, Diabetes $\ge 126\text{ mg/dL}$.
  * *HbA1c (ADA):* Normal $< 5.7\%$, Prediabetes $5.7\%–6.4\%$, Diabetes $\ge 6.5\%$.
  * *Total Cholesterol (NCEP/AHA):* Desirable $< 200\text{ mg/dL}$, Borderline $200–239\text{ mg/dL}$, High $\ge 240\text{ mg/dL}$.
* **User Marker & Dynamic Percentile:** A prominent vertical line indicating the user's exact value, with immediate percentile calculations across both lenses.

### 3.3 Demographic Bounding Box (Explainable k-NN)
Rather than displaying abstract distance scores ($d = 1.42$), the app displays the interquartile range (IQR) and median of the matched cohort:
* Matched Age Range: $[Q_{25}, Q_{75}]$ and Median
* Matched BMI Range: $[Q_{25}, Q_{75}]$ and Median
* Matched Blood Pressure Range: $[Q_{25}, Q_{75}]$ and Median
* **Support Quality Indicator:** Evaluates local sample density; displays a sparsity caution if the user has a rare biometric profile (e.g., age 85 with BMI 48).

### 3.4 Confounder & Diagnostic Telemetry
* **Diagnosed vs. Undiagnosed Dysglycemia:** Decomposes peer diabetes into individuals previously diagnosed by a physician (`DIQ010 == 1`) versus individuals with undiagnosed diabetes identified during NHANES lab testing (`LBXGH >= 6.5` or `LBXGLU >= 126`).
* **The Medication Mask:** Reports the percentage of matched peers actively taking lipid-lowering agents (statins) or anti-hyperglycemics, explaining why some peers exhibit surprisingly low cholesterol or glucose despite higher body weight.

### 3.5 The "Trajectory Lens" (Counterfactual Cohort Shift)
An interactive slider allowing users to shift hypothetical BMI (e.g., 31 $\to$ 26) or blood pressure.
* **Epidemiological Integrity:** Explicitly labeled as *"How metabolic distributions look in cohorts with lower BMI"*, avoiding invalid causal claims (*"If you lose 10 lbs, your glucose will drop to X"*).

### 3.6 The Clinical Provider Brief (1-Page PDF Export)
A structured medical brief designed for primary care appointments:
1. **Patient Biometric Baseline:** Measured biometrics and entered lab values.
2. **Epidemiological Summary:** Percentile ranking within peer cohort vs. optimal cohort, plus peer prevalence rates.
3. **Three Evidence-Based Provider Questions:** Context-specific prompts (e.g., *"Given my fasting glucose of 104 mg/dL and BMI of 31, should we confirm with an HbA1c test?"*, *"Would a continuous glucose monitor or oral glucose tolerance test be appropriate?"*).

---

## 4. CDC NHANES Data Architecture

### 4.1 Multi-Cycle Pooling
The engine pools three continuous cycles of CDC NHANES data:
* **Cycle H (2013–2014)**
* **Cycle I (2015–2016)**
* **Cycle J (2017–2018)**

Pooling across 6 years provides statistical power: $\approx 15,000$ examined adults for general examination labs (HbA1c, Lipids) and $\approx 8,400$ adults for the morning fasting subsample (Fasting Glucose).

### 4.2 Dual-Pool Survey Architecture
NHANES utilizes complex, multistage probability sampling. Using the wrong survey weights introduces severe statistical bias. The pipeline partitions data into two isolated analytical pools:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CDC NHANES Microdata                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│     MEC Examination Pool        │   │    Fasting Subsample Pool       │
│  (HbA1c, Total/HDL Cholesterol) │   │    (Fasting Plasma Glucose)     │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│ • Weight: WTMEC6YR = WTMEC2YR/3 │   │ • Weight: WTSAF6YR = WTSAF2YR/3 │
│ • Sample Size: N ≈ 15,000 adults│   │ • Filter: PHAFSTHR in [8, 24)   │
│ • Files: DEMO, BMX, BPX, GHB,   │   │ • Sample Size: N ≈ 8,400 adults │
│   TCHOL, HDL, DIQ               │   │ • Files: GLU, TRIGLY + MEC      │
└─────────────────────────────────┘   └─────────────────────────────────┘
```

### 4.3 Variable Harmonization & Cleaning Rules
1. **Demographics (`DEMO_*`):**
   * Filter: Adults aged $\ge 20$ years (`RIDAGEYR >= 20`).
   * Exclusions: Exclude pregnant individuals (`RIDEXPRG == 1`).
   * Complex Design Variables: Retain Strata (`SDMVSTRA`) and Primary Sampling Units (`SDMVPSU`).
2. **Anthropometry (`BMX_*`):**
   * Body Mass Index: `BMXBMI` ($\text{kg/m}^2$). Plausibility range: $[12, 80]$.
3. **Blood Pressure Protocol (`BPX_*`):**
   * Consecutive auscultatory readings: Systolic (`BPXSY1–3`) and Diastolic (`BPXDI1–3`).
   * **Cleaning Protocol:** Discard reading 1 to eliminate white-coat artifact. Compute the arithmetic mean of readings 2 and 3. If only one valid reading exists, use that reading.
4. **Fasting Glucose (`GLU_*`):**
   * Fasting glucose: `LBXGLU` ($\text{mg/dL}$).
   * **Compliance Filter:** Participant must have completed a documented fast between 8 and 24 hours (`PHAFSTHR >= 8 & PHAFSTHR < 24`).
5. **HbA1c & Lipids (`GHB_*`, `TCHOL_*`, `HDL_*`):**
   * Glycohemoglobin: `LBXGH` ($\%$).
   * Total Cholesterol: `LBXTC` ($\text{mg/dL}$), HDL: `LBDHDD` ($\text{mg/dL}$).

---

## 5. Mathematical & Algorithmic Engine

### 5.1 Hemodynamic Decoupling
Systolic (SBP) and Diastolic (DBP) blood pressure exhibit high collinearity ($r \approx 0.70 - 0.80$). Feeding both directly into Euclidean space artificially overweights vascular resistance. The engine decouples blood pressure into two physiologically independent components:
1. **Mean Arterial Pressure (MAP):** Reflects steady-state organ perfusion resistance:
   $$\text{MAP} = \text{DBP} + \frac{1}{3}(\text{SBP} - \text{DBP})$$
2. **Pulse Pressure (PP):** Reflects pulsatile load, stroke volume, and large-artery stiffness:
   $$\text{PP} = \text{SBP} - \text{DBP}$$

### 5.2 Distance Space & Stratified Mahalanobis Matching
Matching operates over the coordinate vector:
$$\mathbf{x} = \begin{bmatrix} \text{Age}, & \text{BMI}, & \text{MAP}, & \text{PP} \end{bmatrix}^T$$

* **Exact Stratification:** Matching is partitioned strictly within the same biological sex:
  $$\text{Pool} = \{i : \text{Sex}_i == \text{Sex}_{\text{user}}\}$$
* **Regularized Whitened Mahalanobis Distance:** Within the stratified pool, distance uses $\mathbf{\Sigma}_\lambda = \mathbf{\Sigma} + \lambda \mathbf{I}$ (see `PROJECT_PLAN.md` §4.2).
* **Neighborhood Selection:** Default $k=300$ nearest neighbors (allowed range 100–500 per `src/config.py`).

### 5.3 Hazen Continuity-Corrected Weighted Percentile
For a user value $y^*$ evaluated against neighbor values $\{y_1, \dots, y_k\}$ with survey weights $\{w_1, \dots, w_k\}$:
1. Sort the neighbors such that $y_{(1)} \le y_{(2)} \le \dots \le y_{(k)}$.
2. Calculate the continuity-corrected empirical cumulative weight (Hazen's rule):
   $$S_{(j)} = \frac{\sum_{m=1}^{j} w_{(m)} - 0.5 \cdot w_{(j)}}{\sum_{m=1}^{k} w_{(m)}}$$
3. Estimate the percentile rank via linear interpolation:
   $$\text{Percentile}(y^*) = 100 \times \text{Interp}(y^*, \{y_{(j)}\}, \{S_{(j)}\})$$
   *Out-of-bounds values are constrained to $[0.1, 99.9]$ with an extreme outlier notice.*

---

## 6. Statistical Validation & Research Rigor

To guarantee scientific credibility, the system is calibrated using an empirical validation suite:

### 6.1 Leave-One-Out Prediction Interval Coverage
For every participant in the validation set:
1. Exclude participant $i$ from the matching pool.
2. Form their nearest neighbor cohort.
3. Compute the weighted prediction interval at nominal levels $\alpha \in \{0.50, 0.80, 0.90, 0.95\}$.
4. Measure empirical coverage across the population:
   $$\text{Coverage}(\alpha) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}\left(y_i \in [L_i(\alpha), U_i(\alpha)]\right)$$
   *Target: Empirical coverage matches nominal level within $\pm 1.5\%$.*

### 6.2 Scoring Rules (Winkler Sharpness Score)
Unconditional national percentiles can achieve 90% coverage by being excessively wide. To evaluate information gain, the engine computes the **Winkler Interval Score**:
$$\text{Score}_{\alpha}(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)$$
*Target: The k-NN matched model must achieve a lower Winkler score (higher sharpness at equivalent coverage) than both the "All US Adults" baseline and the "Age/Sex Only" baseline.*

### 6.3 Algorithmic Fairness & Subgroup Equity Audit
Automated scripts audit prediction coverage across:
* Race / Hispanic Origin (`RIDRETH3`: Non-Hispanic White, Non-Hispanic Black, Mexican American, Other Hispanic, Non-Hispanic Asian).
* Income-to-Poverty Ratio (`INDFMPIR` quartiles).
* Age decades ($20–39, 40–59, 60–79, 80+$).
* Discrepancies $> 3\%$ are documented and surfaced in the application's scientific transparency disclosures.

---

## 7. Technical Architecture & Tech Stack

```
People_Like_Me/
├── data/
│   ├── raw/                  # Auto-downloaded NHANES .XPT files
│   └── processed/            # Harmonized Parquet files (MEC & Fasting pools)
├── src/
│   ├── data_pipeline/        # Ingestion scraper, BP cleaning, Pandera validation
│   ├── engine/               # Hemodynamics, Mahalanobis matching, Hazen percentiles
│   ├── validation/           # Leave-one-out coverage & Winkler scoring suite
│   └── ui/                   # Streamlit interface, Plotly charts, ReportLab PDF engine
├── tests/                    # Pytest test suite for pipeline, math, and UI
├── requirements.txt          # Pinned production dependencies
└── app.py                    # Main application launcher
```

* **Data Processing:** Python 3.10+, Pandas, NumPy, SciPy, PyArrow.
* **Data Contracts:** Pandera (enforcing non-null biometrics and physiological bounds).
* **Algorithmic Engine:** Scikit-Learn `NearestNeighbors` (Mahalanobis metric).
* **User Interface:** Streamlit (rapid, clean, responsive reactive interface).
* **Visualization:** Plotly (interactive continuous density and cumulative distribution curves with hover telemetry).
* **PDF Engine:** ReportLab (clinical 1-page SBAR PDF generator).
* **Testing:** Pytest (verifying survey weight math, hemodynamics, and interpolation).

---

## 8. Safety, Ethics, and Clinical Disclaimers

1. **Descriptive, Not Diagnostic:** The application explicitly states that it does not diagnose medical conditions or predict individual risk trajectories. It provides empirical context.
2. **US Population Representation:** NHANES is representative of the civilian non-institutionalized United States population. The interface clearly notes that baseline distributions may differ for non-US populations.
3. **Zero Data Retention / Privacy-by-Design:** User biometrics and lab inputs exist only in memory during the active session. No inputs are stored in a database, logged to disk, or transmitted to third parties.

---

## 9. Anticipated Inquiries & Defenses

* **"Why not simply use standard laboratory reference ranges?"**  
  *Response:* Clinical reference ranges are essential for diagnosing acute disease, but they are binary and monolithic. They fail to show where an individual sits along the spectrum of people with their somatic baseline, obscuring whether a "borderline" value is typical or exceptional.
* **"Why k-NN over deep learning or complex neural networks?"**  
  *Response:* In healthcare, interpretability is paramount. k-NN is non-parametric and transparent: we can inspect the exact individuals comprising the cohort, verify their demographic bounding box, and explain every percentile calculation directly to a physician.
* **"Does telling someone they are 'average' discourage healthy lifestyle changes?"**  
  *Response:* This is precisely why we created the **Dual-Lens Architecture**. By simultaneously displaying the **Optimal Health Benchmark**, users immediately see the gap between population reality and physiological health.
