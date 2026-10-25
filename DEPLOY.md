# Deploy (Hackathon)

## Hugging Face Spaces (recommended)

1. Sign in at [huggingface.co](https://huggingface.co) (free).
2. **New Space** → SDK **Streamlit** → link **GitHub** repo `Trie-hard/People_Like_Me` (or upload this folder).
3. Space settings: Python **3.10+**, app file **`app.py`**.
4. First build installs `requirements.txt` and runs Streamlit. Processed NHANES Parquet files are in `data/processed/` (~1 MB total).
5. Open the Space URL — no laptop required after deploy.

If the Space build fails on memory, reduce `K_NEIGHBORS_DEFAULT` in `src/config.py` for the demo tier.

## Local demo (judges at your table)

```powershell
cd People_Like_Me
$env:PYTHONPATH = (Get-Location)
python -m pip install -r requirements.txt
streamlit run app.py
```

Click **Maya demo** in the sidebar, then download the PDF brief.

## 2-minute pitch outline

1. **Problem:** Lab portals say “high” without context for *your* body (age, BMI, BP).
2. **Trap:** “Average among people like you” can still be unhealthy in the US.
3. **Solution:** **Lens A** (peer k-NN on NHANES) vs **Lens B** (optimal benchmark).
4. **Demo:** Maya, 45, BMI 31, HbA1c 5.8% — common among peers, extreme vs healthy cohort.
5. **Safety:** Descriptive epidemiology only — not diagnosis; US NHANES 2013–2018.
