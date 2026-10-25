---
title: People Like Me
emoji: 🧬
colorFrom: blue
colorTo: teal
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# People Like Me

> **Your lab results, in epidemiological context.**

Hackathon-ready Streamlit app that places lab values on **two** NHANES-based distributions:

1. **Lens A — Peers like you:** $k$-NN match on age, sex, BMI, and hemodynamics (MAP/PP).
2. **Lens B — Optimal benchmark:** Same sex and age decade, healthy BMI, normotensive, non-diabetic.

Avoids the **“sick population trap”** (looking “average” among peers who are already high-risk).

**Try it:** open the Hugging Face Space (if linked) or run locally — sidebar **Maya demo**, then export a 1-page PDF for a clinician.

---

## Demo (local)

Python **3.10+** required.

```bash
git clone https://github.com/Trie-hard/People_Like_Me.git
cd People_Like_Me
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"   # Windows: set PYTHONPATH=%CD%
streamlit run app.py
```

Processed CDC NHANES cohorts (2013–2018, cycles H–J) ship in `data/processed/`. To rebuild from CDC XPT files:

```bash
python scripts/build_nhanes_parquet.py --purge-invalid
```

---

## Features (MVP)

| Feature | Description |
|--------|-------------|
| Biomarkers | HbA1c, fasting glucose, total cholesterol, HDL-C |
| Survey weights | MEC (`wtmec6yr`) vs fasting (`wtsaf6yr`) pools |
| Transparency | Demographic bounding box, match quality index |
| Confounders | Peer medication rates + dysglycemia iceberg |
| Export | Clinical provider brief (PDF) |

---

## Safety

Descriptive context only — **not** a diagnosis, treatment plan, or individual risk prediction. US civilian NHANES 2013–2018; inputs stay in session memory (no server-side persistence in this MVP).

---

## Docs

- Engineering spec: [PROJECT_PLAN.md](PROJECT_PLAN.md)
- Product story (Maya): [PEOPLE_LIKE_ME.md](PEOPLE_LIKE_ME.md)
- Hosting: [DEPLOY.md](DEPLOY.md)

```bash
python -m pytest tests/ -q
```
