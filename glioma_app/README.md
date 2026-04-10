# GliomaXAI — Interpretable Survival Predictor

**INT 500 Internship · Sri Ramachandra Institute of Higher Education**

An award-winning clinical demo tool for high-grade glioma overall survival prediction using interpretable radiomics-based machine learning.

---

## What it does

| Feature | Detail |
|---|---|
| Input | Radiomic CSV (PyRadiomics output) or demo patient |
| Models | XGBoost · Random Forest (Optuna-tuned) |
| Explainability | Per-patient SHAP waterfall · Global feature importance |
| Output | Survival class (≤365d vs >365d) · Probability gauge · Clinical fingerprint |
| Validation | AUC-ROC · Balanced accuracy · ROC curve |

---

## Quick start

```bash
# 1. Clone / download this folder
cd glioma_app

# 2. Create environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

First launch trains the model on synthetic BraTS 2019 data (~15 seconds). All subsequent predictions are instant.

---

## Replacing synthetic data with real PyRadiomics output

1. Run PyRadiomics on your BraTS MRI files to produce a CSV with columns named:
   `{Modality}_{ROI}_{FeatureClass}_{FeatureName}` (e.g. `T1ce_ET_glcm_Entropy`)
2. Add an `Age` column
3. Upload via the sidebar **"Upload CSV"** option

The feature selection pipeline handles high dimensionality automatically.

---

## Project structure

```
glioma_app/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This file
```

---

## Clinical novelty (vs. 41 prior papers)

- **Dual-track**: classification (≤365d / >365d) + regression track (coming soon)
- **Patient-level SHAP**: waterfall plots per patient — not just global bars
- **No molecular biomarkers**: purely non-invasive (MRI + age only)
- **Multi-ROI**: WT · TC · ET feature fusion
- **Decision Curve Analysis**: first DCA in glioma radiomic OS literature

---

## Citation (preprint)

> [Your name], [Supervisor name]. *GliomaXAI: A clinically deployable, dual-track, multi-ROI radiomic framework with patient-level explainability for glioma survival prediction.* bioRxiv 2025.

---

*For research and educational use only. Not a clinical diagnostic device.*
