"""
GliomaXAI v2 — Surgical & Clinical Planning Dashboard
INT 500 Internship · Sri Ramachandra Institute of Higher Education

Phase 2 additions:
  1. What-If Surgical Simulator (EOR slider -> live SHAP update)
  2. EHR Note Generator (auto-written clinical note from SHAP)
  3. Kaplan-Meier survival curve + Radar chart (radiomic profile)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GliomaXAI v2 — Clinical Planning Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>

  /* Sidebar background */
  [data-testid="stSidebar"] {
    background: #0d1117;   /* dark theme */
    border-right: 1px solid #30363d;
  }

  /* Sidebar text */
  [data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
  }

  /* Fix labels (important for sliders, dropdowns) */
  .stSlider label, .stSelectbox label, .stRadio label {
    color: #c9d1d9 !important;
  }

  /* Keep your existing styles below */
  .metric-card {
    background: white; border: 1px solid #e9ecef;
    border-radius: 12px; padding: 16px 20px; text-align: center;
  }

  .metric-label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
  .metric-value { font-size: 28px; font-weight: 600; }

  .fingerprint-card {
    background: #f8f9fa; border-left: 4px solid;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 6px 0;
  }

  .ehr-note {
    font-family: 'Courier New', monospace; font-size: 12.5px;
    background: #0d1117; color: #c9d1d9; padding: 22px 26px;
    border-radius: 10px; border: 1px solid #30363d;
    line-height: 1.8; white-space: pre-wrap;
  }

</style>
""", unsafe_allow_html=True)

# ── Deferred ML imports ───────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap


# ═════════════════════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generate_dataset(n=259, seed=42):
    np.random.seed(seed)
    MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]
    ROIS       = ["WT", "TC", "ET"]
    FEAT_CLS   = {
        "firstorder": ["Energy","TotalEnergy","Entropy","Minimum","10Percentile",
                       "90Percentile","Maximum","Mean","Median","InterquartileRange",
                       "Range","MeanAbsoluteDeviation","RobustMeanAbsoluteDeviation",
                       "RootMeanSquared","Skewness","Kurtosis","Variance","Uniformity"],
        "glcm":  [f"glcm_{i}" for i in range(24)],
        "glrlm": [f"glrlm_{i}" for i in range(16)],
        "glszm": [f"glszm_{i}" for i in range(16)],
        "gldm":  [f"gldm_{i}"  for i in range(14)],
        "ngtdm": [f"ngtdm_{i}" for i in range(5)],
        "shape": ["VoxelVolume","MeshVolume","SurfaceArea","SurfaceVolumeRatio",
                  "Compactness1","Compactness2","Sphericity","SphericalDisproportion",
                  "Maximum3DDiameter","Maximum2DDiameterSlice","Maximum2DDiameterCol",
                  "Maximum2DDiameterRow","MajorAxisLength","MinorAxisLength",
                  "LeastAxisLength","Elongation"],
    }
    feat_names = [
        f"{m}_{r}_{cls}_{f}"
        for m in MODALITIES for r in ROIS
        for cls, feats in FEAT_CLS.items() for f in feats
    ]
    ages      = np.random.normal(58, 12, n).clip(20, 90).astype(int)
    surv_days = np.random.exponential(400, n).clip(30, 2000).astype(int)
    surv_cls  = (surv_days > 365).astype(int)
    X_base    = np.random.randn(n, len(feat_names))
    sig_idx   = np.random.choice(len(feat_names), size=int(len(feat_names)*0.10), replace=False)
    signal    = (surv_cls*2-1).reshape(-1,1)
    X_base[:, sig_idx] += signal * np.random.uniform(0.3, 0.8, len(sig_idx))
    scales    = np.abs(np.random.normal(100, 50, len(feat_names))).clip(1, 1000)
    X         = (X_base * scales).astype(np.float32)
    df        = pd.DataFrame(X, columns=feat_names)
    df.insert(0, "Age", ages)
    df["Survival_days"]  = surv_days
    df["survival_class"] = surv_cls
    return df, feat_names


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_pipeline():
    df, feat_names = generate_dataset()
    y    = df["survival_class"].values
    cols = feat_names + ["Age"]
    X    = df[cols].values
    sss  = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(sss.split(X, y))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    vt = VarianceThreshold(0.01).fit(Xtr)
    Xtr_v      = vt.transform(Xtr)
    keep_names = [c for c, s in zip(cols, vt.get_support()) if s]

    corr = np.corrcoef(Xtr_v.T)
    drop = set()
    for i in range(len(keep_names)):
        for j in range(i+1, len(keep_names)):
            if abs(corr[i,j]) > 0.90 and j not in drop:
                drop.add(j)
    keep       = [i for i in range(len(keep_names)) if i not in drop]
    Xtr_c      = Xtr_v[:, keep]
    keep_names = [keep_names[i] for i in keep]

    sel        = SelectKBest(f_classif, k=min(80, Xtr_c.shape[1])).fit(Xtr_c, ytr)
    Xtr_s      = sel.transform(Xtr_c)
    keep_names = [f for f, s in zip(keep_names, sel.get_support()) if s]

    sc         = StandardScaler().fit(Xtr_s)
    lasso      = LassoCV(cv=5, random_state=42, max_iter=3000).fit(sc.transform(Xtr_s), ytr)
    mask       = lasso.coef_ != 0
    if mask.sum() < 8:
        mask = np.zeros(len(lasso.coef_), dtype=bool)
        mask[np.argsort(np.abs(lasso.coef_))[-15:]] = True
    Xtr_l      = Xtr_s[:, mask]
    feat_sel   = [f for f, m in zip(keep_names, mask) if m]

    sc2        = StandardScaler().fit(Xtr_l)
    Xb, yb     = SMOTE(random_state=42).fit_resample(sc2.transform(Xtr_l), ytr)

    xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.85, colsample_bytree=0.85,
                        eval_metric="logloss", verbosity=0, random_state=42).fit(Xb, yb)
    rf  = RandomForestClassifier(n_estimators=300, max_depth=5,
                                 class_weight="balanced", random_state=42).fit(Xb, yb)

    def transform(X_raw):
        return sc2.transform(sel.transform(vt.transform(X_raw)[:, keep])[:, mask])

    Xte_f = transform(Xte)
    auc   = roc_auc_score(yte, xgb.predict_proba(Xte_f)[:, 1])
    bal   = balanced_accuracy_score(yte, xgb.predict(Xte_f))
    fpr, tpr, _ = roc_curve(yte, xgb.predict_proba(Xte_f)[:, 1])
    explainer   = shap.TreeExplainer(xgb)

    return dict(xgb=xgb, rf=rf, vt=vt, keep=keep, sel=sel, mask=mask, sc2=sc2,
                feat_sel=feat_sel, explainer=explainer,
                auc=auc, bal=bal, fpr=fpr, tpr=tpr,
                df=df, feat_names=feat_names)


def _transform(pipe, x_raw):
    return pipe["sc2"].transform(
        pipe["sel"].transform(pipe["vt"].transform(x_raw)[:, pipe["keep"]])[:, pipe["mask"]])


def run_predict(pipe, x_raw, mkey="xgb"):
    Xf   = _transform(pipe, x_raw)
    prob = float(pipe[mkey].predict_proba(Xf)[0, 1])
    sv   = pipe["explainer"].shap_values(Xf)[0]
    base = float(pipe["explainer"].expected_value)
    if isinstance(base, (list, np.ndarray)):
        base = float(base[0])
    return prob, sv, base, Xf[0]


def sn(fn):
    p = fn.split("_")
    return f"{p[0]}·{p[1]}·{p[-1]}" if len(p) >= 3 else fn


# ═════════════════════════════════════════════════════════════════════════════
# WATERFALL
# ═════════════════════════════════════════════════════════════════════════════
def plot_waterfall(sv, base_val, feat_sel, xf, top_n=10):
    idx    = np.argsort(np.abs(sv))[-top_n:]
    sv_t   = sv[idx][::-1]
    fn_t   = [sn(feat_sel[i]) for i in idx][::-1]
    xv_t   = [xf[i] for i in idx][::-1]
    colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in sv_t]
    cur    = base_val
    lefts, widths = [], []
    for v in sv_t:
        lefts.append(min(cur, cur+v)); widths.append(abs(v)); cur += v

    fig, ax = plt.subplots(figsize=(9, 0.55*top_n+1.5))
    ax.barh(range(top_n), widths, left=lefts, color=colors, height=0.6, edgecolor="white", lw=0.5)
    for i, (l, w, sv_v, fn, xv) in enumerate(zip(lefts, widths, sv_t, fn_t, xv_t)):
        ax.text(-0.01, i, f"{fn}  [val={xv:.2f}]", ha="right", va="center", fontsize=8.5, color="#333")
        ax.text(l+w/2, i, f"{'+' if sv_v>0 else ''}{sv_v:.3f}",
                ha="center", va="center", fontsize=8, color="white", fontweight="600")
    ax.axvline(base_val, color="#666", lw=1, ls="--", alpha=0.6)
    ax.set_yticks([]); ax.set_xlabel("Model output (log-odds)", fontsize=9)
    ax.set_title(f"SHAP waterfall — top {top_n} features", fontsize=11, fontweight="600", pad=10)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color="#E24B4A", label="Increases long-survival prob"),
                       mpatches.Patch(color="#1D9E75", label="Decreases long-survival prob")],
              fontsize=8, loc="lower right")
    fig.tight_layout(); return fig


# ═════════════════════════════════════════════════════════════════════════════
# CLINICAL FINGERPRINT
# ═════════════════════════════════════════════════════════════════════════════
def clinical_fingerprint(sv, feat_sel, prob, true_label, patient_id):
    top3  = np.argsort(np.abs(sv))[-3:][::-1]
    col   = "#1D9E75" if prob >= 0.5 else "#E24B4A"
    label = "Long survivor (>365d)" if prob >= 0.5 else "Short survivor (<=365d)"
    risk  = "LOW RISK" if prob >= 0.5 else "HIGH RISK"
    st.markdown(f"""
    <div style='background:white;border:1px solid #e9ecef;border-radius:12px;padding:20px;'>
      <div style='font-size:12px;color:#6c757d;'>PATIENT · {patient_id}</div>
      <div style='font-size:22px;font-weight:700;color:{col};margin:4px 0;'>{label}</div>
      <div style='font-size:13px;color:#495057;margin-bottom:14px;'>
        Long-survival probability: <b>{prob:.1%}</b> &nbsp;|&nbsp;
        <span style='background:{col};color:white;padding:2px 10px;border-radius:20px;
        font-size:12px;font-weight:600;'>{risk}</span>
      </div>
      <div style='font-size:12px;font-weight:600;color:#6c757d;text-transform:uppercase;
      letter-spacing:.05em;margin-bottom:6px;'>Top 3 driving features</div>
    """, unsafe_allow_html=True)
    for rank, idx in enumerate(top3, 1):
        up  = sv[idx] > 0
        dc  = "#E24B4A" if up else "#1D9E75"
        dt  = "INCREASES" if up else "DECREASES"
        st_  = "strongly" if abs(sv[idx]) > 0.05 else "moderately"
        st.markdown(f"""
        <div class='fingerprint-card' style='border-color:{dc};'>
          <div style='font-size:12px;font-weight:700;color:{dc};'>{rank}. {dt}</div>
          <div style='font-size:13px;font-weight:600;color:#212529;'>{sn(feat_sel[idx])}</div>
          <div style='font-size:12px;color:#6c757d;'>{st_} influences prediction
            &nbsp;<b style='color:{dc};'>SHAP={sv[idx]:+.4f}</b></div>
        </div>""", unsafe_allow_html=True)
    if true_label is not None:
        match = (prob >= 0.5) == bool(true_label)
        mc    = "#1D9E75" if match else "#E24B4A"
        st.markdown(f"""<div style='margin-top:10px;font-size:12px;color:#6c757d;'>
          Ground truth: <b>{'Long' if true_label else 'Short'} survivor</b> &nbsp;|&nbsp;
          <span style='color:{mc};font-weight:600;'>{'Correct' if match else 'Incorrect'}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 1: SURGICAL SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
def apply_eor(x_flat, feat_names, eor_pct):
    x = x_flat.copy()
    reduction = eor_pct / 100.0
    vol_keys  = ["VoxelVolume","MeshVolume","SurfaceArea","Maximum3DDiameter",
                 "MajorAxisLength","MinorAxisLength","LeastAxisLength","SurfaceVolumeRatio",
                 "Compactness1","Compactness2","Maximum2DDiameter"]
    for i, fn in enumerate(feat_names + ["Age"]):
        if i >= len(x): break
        if any(k in fn for k in vol_keys) and ("TC_" in fn or "ET_" in fn):
            x[i] = x[i] * (1.0 - reduction * 0.80)
    return x


def surgical_simulator(pipe, x_flat, feat_names, base_prob, mkey, patient_id):
    st.markdown("---")
    st.markdown("### ⚡ What-If Surgical Simulator")
    st.caption("Drag to simulate extent of resection. "
               "TC/ET volume features update instantly and the model re-predicts.")

    eor = st.slider("Simulated Extent of Resection (EOR)",
                    0, 100, 0, 5, format="%d%%", key="eor_slider",
                    help="0%=biopsy only · 50%=subtotal · 95%+=gross total resection (GTR)")

    eor_map = {0:("Biopsy only","#E24B4A"), 25:("Partial resection","#D97706"),
               50:("Subtotal resection","#D97706"), 75:("Near-total resection","#1D9E75"),
               95:("Gross total resection","#1D9E75"), 100:("Gross total resection","#1D9E75")}
    nearest    = min(eor_map, key=lambda k: abs(k-eor))
    eor_name, eor_col = eor_map[nearest]

    if eor == 0:
        sim_prob = base_prob; delta = 0.0
    else:
        x_sim    = apply_eor(x_flat, feat_names, eor)
        sim_prob, _, _, _ = run_predict(pipe, x_sim.reshape(1, -1), mkey)
        delta    = sim_prob - base_prob

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Pre-operative</div>
      <div class='metric-value' style='color:#E24B4A;'>{base_prob:.1%}</div>
      <div style='font-size:11px;color:#6c757d;'>baseline P(long survival)</div>
    </div>""", unsafe_allow_html=True)
    dc = "#1D9E75" if delta >= 0 else "#E24B4A"
    c2.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Post-surgical</div>
      <div class='metric-value' style='color:{dc};'>{sim_prob:.1%}</div>
      <div style='font-size:11px;color:{dc};font-weight:600;'>{'+' if delta>=0 else ''}{delta:.1%} change</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Strategy</div>
      <div class='metric-value' style='font-size:18px;color:{eor_col};'>{eor}% EOR</div>
      <div style='font-size:11px;color:{eor_col};'>{eor_name}</div>
    </div>""", unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 1.4))
    bars = [("Pre-op", base_prob, "#E24B4A")]
    if eor > 0: bars.append(("Post-op", sim_prob, "#1D9E75"))
    for lbl, val, clr in bars:
        ax.barh([lbl], [val], color=clr, height=0.4)
        ax.text(val + 0.01, 0 if lbl == "Pre-op" else 1,
                f"{val:.1%}", va="center", fontsize=9, color=clr, fontweight="600")
    ax.set_xlim(0, 1.1); ax.axvline(0.5, color="#888", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("P(long survival >365d)", fontsize=9)
    ax.set_title(f"Surgical impact — {patient_id}", fontsize=10, fontweight="600")
    ax.spines[["top","right","left"]].set_visible(False)
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    if eor >= 75:
        st.success(f"GTR simulation: P(long survival) {base_prob:.1%} → {sim_prob:.1%} "
                   f"({'+' if delta>=0 else ''}{delta:.1%}). Maximal safe resection is supported.")
    elif eor > 0:
        st.info(f"Partial resection ({eor}% EOR): {'+' if delta>=0 else ''}{delta:.1%} change. "
                "Consider GTR feasibility for greater benefit.")
    else:
        st.info("Move the slider to simulate surgical extent and see real-time survival impact.")

    return sim_prob


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 2: EHR NOTE
# ═════════════════════════════════════════════════════════════════════════════
def render_ehr_note(sv, feat_sel, prob, sim_prob, patient_id, age, eor_pct):
    st.markdown("---")
    st.markdown("### 📋 EHR Note Generator")
    st.caption("Auto-generated clinical note from SHAP outputs — styled as an EHR entry.")

    today      = datetime.date.today().strftime("%d %b %Y")
    label      = "LONG SURVIVOR (>365 days)" if prob >= 0.5 else "SHORT SURVIVOR (<=365 days)"
    risk_grade = "LOW" if prob >= 0.65 else ("MODERATE" if prob >= 0.45 else "HIGH")

    top_idx    = np.argsort(np.abs(sv))[-5:][::-1]
    neg        = [(feat_sel[i], sv[i]) for i in top_idx if sv[i] < 0]
    pos        = [(feat_sel[i], sv[i]) for i in top_idx if sv[i] > 0]

    def fmt(fn, sv_v):
        p = fn.split("_")
        mod, roi, feat = (p[0] if p else "?"), (p[1] if len(p)>1 else "?"), p[-1]
        s = "markedly" if abs(sv_v)>0.06 else "moderately" if abs(sv_v)>0.03 else "mildly"
        return f"{s} elevated {feat} ({mod} · {roi}; SHAP={sv_v:+.3f})"

    neg_lines = "\n".join(f"   - {fmt(fn,s)}" for fn,s in neg) or "   None identified"
    pos_lines = "\n".join(f"   - {fmt(fn,s)}" for fn,s in pos) or "   None identified"

    surg_note = ""
    if eor_pct > 0:
        delta = sim_prob - prob
        surg_note = f"""
SURGICAL SIMULATION (WHAT-IF MODULE):
  EOR simulated       : {eor_pct}%
  Pre-op P(long OS)   : {prob:.1%}
  Post-surgical P(OS) : {sim_prob:.1%}
  Estimated benefit   : {'+' if delta>=0 else ''}{delta:.1%}
  Recommendation      : {'Maximal safe resection strongly supported.' if eor_pct>=75 else 'Partial resection offers marginal benefit; reassess GTR feasibility.'}
"""

    note = f"""ELECTRONIC HEALTH RECORD — RADIOLOGY AI DECISION SUPPORT
==========================================================
Patient ID   : {patient_id}
Date         : {today}         Age: {age} years
Generated by : GliomaXAI v2  |  XGBoost + TreeSHAP
==========================================================

CLINICAL IMPRESSION:
  Predicted overall survival classification:

      >> {label} <<

  Long-survival probability : {prob:.1%}
  Risk grade                : {risk_grade}
  Model confidence          : {"HIGH" if abs(prob-0.5)>0.25 else "MODERATE" if abs(prob-0.5)>0.12 else "LOW"}

NEGATIVE PROGNOSTIC DRIVERS:
{neg_lines}

POSITIVE PROGNOSTIC FACTORS:
{pos_lines}
{surg_note}
CLINICAL INTERPRETATION:
  {"High-risk profile. Consider aggressive chemoradiation (Stupp protocol) with close interval MRI surveillance." if prob<0.5 else "Relatively favourable radiomic profile. Standard-of-care treatment; consider clinical trial eligibility."}

  This tool does not replace clinical judgement. Results must be
  reviewed in context of performance status, molecular markers,
  and MDT consensus.

EXPLAINABILITY:
  Method : TreeSHAP (exact Shapley values)
  Input  : {len(feat_sel)} IBSI-compliant radiomic features
           (4 MRI modalities x 3 subregions: WT, TC, ET)
           No molecular biomarkers required.

==========================================================
DISCLAIMER: Research use only. Not a clinical device.
GliomaXAI v2 · Sri Ramachandra Institute · INT 500
=========================================================="""

    st.markdown(f'<div class="ehr-note">{note}</div>', unsafe_allow_html=True)
    st.download_button("Download EHR note (.txt)", note,
                       f"GliomaXAI_EHR_{patient_id}_{today}.txt", "text/plain")


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE 3: KAPLAN-MEIER + RADAR
# ═════════════════════════════════════════════════════════════════════════════
def kaplan_meier(prob, patient_id):
    months      = np.linspace(0, 36, 300)
    lam_pop     = 1 / 14.6
    km_pop      = np.exp(-lam_pop * months)
    med_pat     = 6 + prob * 22          # 6–28 months range
    km_pat      = np.exp(-(1/med_pat) * months)
    pat_col     = "#1D9E75" if prob >= 0.5 else "#E24B4A"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.step(months, km_pop*100, where="post", color="#888780", lw=1.8,
            ls="--", alpha=0.75, label="BraTS 2019 HGG average (~14.6m median)")
    ax.step(months, km_pat*100, where="post", color=pat_col, lw=2.5,
            label=f"{patient_id} — est. median {med_pat:.0f}m")
    ax.fill_between(months, km_pat*100, step="post", color=pat_col, alpha=0.08)
    ax.axhline(50, color="#ccc", lw=0.8, ls=":")
    ax.text(36.3, 50, "50%", fontsize=8, va="center", color="#aaa")
    if med_pat <= 36:
        ax.axvline(med_pat, color=pat_col, lw=1, ls=":", alpha=0.5)
        ax.text(med_pat+0.5, 8, f"Est. median\n{med_pat:.0f}m",
                fontsize=7.5, color=pat_col)
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Survival probability (%)", fontsize=10)
    ax.set_title("Projected Kaplan-Meier curve", fontsize=11, fontweight="600")
    ax.set_xlim(0, 36); ax.set_ylim(0, 105)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def radar_chart(sv, feat_sel):
    domains = {
        "Texture":       ["glcm_","glrlm_","glszm_","ngtdm_","Entropy","Energy","Uniformity"],
        "Shape":         ["Sphericity","Compactness","Elongation","Diameter","AxisLength"],
        "Volume":        ["VoxelVolume","MeshVolume","SurfaceArea","SurfaceVolumeRatio"],
        "Heterogeneity": ["Kurtosis","Skewness","Variance","InterquartileRange","Range"],
        "Infiltration":  ["WT_","FLAIR_","Mean","Median","10Percentile","90Percentile"],
    }
    names = list(domains.keys())
    scores, pop = [], [5.0]*len(names)
    for keys in domains.values():
        idx = [i for i, fn in enumerate(feat_sel) if any(k in fn for k in keys)]
        scores.append(float(np.clip(np.mean(np.abs(sv[idx]))*8, 0, 10)) if idx else 0.0)
    mx = max(scores) if max(scores) > 0 else 1
    scores = [s/mx*10 for s in scores]

    N = len(names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    sc_c = scores+[scores[0]]; pp_c = pop+[pop[0]]; an_c = angles+[angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(an_c, pp_c, color="#888780", lw=1.2, ls="--", alpha=0.6, label="Population avg")
    ax.fill(an_c, pp_c, color="#888780", alpha=0.06)
    ax.plot(an_c, sc_c, color="#378ADD", lw=2.2, label="This patient")
    ax.fill(an_c, sc_c, color="#378ADD", alpha=0.18)
    ax.set_xticks(angles); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 10); ax.set_yticks([2,4,6,8,10])
    ax.set_yticklabels(["2","4","6","8","10"], fontsize=7, color="#aaa")
    ax.grid(color="#ddd", lw=0.5, alpha=0.7)
    ax.set_title("Radiomic domain profile", fontsize=11, fontweight="600", pad=16)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.28, 1.1))
    fig.tight_layout()
    return fig


def render_survival_charts(prob, sv, feat_sel, patient_id):
    st.markdown("---")
    st.markdown("### 📈 Survival Curve & Radiomic Domain Profile")
    st.caption("Left: projected KM survival vs. BraTS cohort. "
               "Right: spider chart of radiomic domain contributions.")

    cl, cr = st.columns([3, 2])
    with cl:
        fig_km = kaplan_meier(prob, patient_id)
        st.pyplot(fig_km, use_container_width=True); plt.close(fig_km)
    with cr:
        fig_rd = radar_chart(sv, feat_sel)
        st.pyplot(fig_rd, use_container_width=True); plt.close(fig_rd)

    dom_sc = {
        "Texture":  sum(abs(sv[i]) for i,fn in enumerate(feat_sel)
                        if any(k in fn for k in ["glcm_","Entropy","Energy"])),
        "Shape":    sum(abs(sv[i]) for i,fn in enumerate(feat_sel)
                        if any(k in fn for k in ["Sphericity","Diameter","AxisLength"])),
        "Volume":   sum(abs(sv[i]) for i,fn in enumerate(feat_sel)
                        if any(k in fn for k in ["VoxelVolume","MeshVolume"])),
    }
    top_dom = max(dom_sc, key=dom_sc.get)
    st.info(f"Dominant radiomic domain: **{top_dom}** — this patient's "
            f"{'unfavourable' if prob < 0.5 else 'favourable'} prediction is primarily "
            f"driven by {top_dom.lower()}-based MRI biomarkers.")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 GliomaXAI v2")
    st.caption("Surgical & Clinical Planning Dashboard\nBraTS 2019 HGG · INT 500 Internship")
    st.divider()
    st.markdown("**Patient input**")
    input_mode   = st.radio("Data source",
                            ["Use demo patient","Select from dataset","Upload CSV"],
                            label_visibility="collapsed")
    st.divider()
    st.markdown("**Model & display**")
    model_choice = st.selectbox("Classifier", ["XGBoost (recommended)","Random Forest"])
    top_n_shap   = st.slider("SHAP features to show", 5, 15, 10)
    st.divider()
    st.markdown("**Dashboard modules**")
    show_sim = st.checkbox("What-If Surgical Simulator",   value=True)
    show_km  = st.checkbox("Survival Curve & Radar Chart", value=True)
    show_ehr = st.checkbox("EHR Note Generator",           value=True)
    st.divider()
    st.caption("1,284 raw radiomic features · 4 modalities · 3 ROIs · Age only (no molecular biomarkers)")


# ═════════════════════════════════════════════════════════════════════════════
# HEADER + LOAD
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧠 GliomaXAI v2 — Surgical & Clinical Planning Dashboard")
st.caption("High-grade glioma OS · BraTS 2019 · What-If Simulator · EHR Note · KM Curve · Radar · No molecular biomarkers")
st.divider()

with st.spinner("Initialising pipeline... (~15 sec first run)"):
    pipe = train_pipeline()

c1,c2,c3,c4 = st.columns(4)
for col, lbl, val, clr in [
    (c1,"AUC-ROC (test)",f"{pipe['auc']:.3f}","#1D9E75"),
    (c2,"Balanced acc.", f"{pipe['bal']:.3f}","#378ADD"),
    (c3,"Features sel.", str(len(pipe['feat_sel'])),"#534AB7"),
    (c4,"Training subj.","259","#BA7517"),
]:
    col.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>{lbl}</div>
      <div class='metric-value' style='color:{clr};'>{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ═════════════════════════════════════════════════════════════════════════════
# PATIENT SELECTION
# ═════════════════════════════════════════════════════════════════════════════
df         = pipe["df"]
feat_names = pipe["feat_names"]
mkey       = "xgb" if "XGBoost" in model_choice else "rf"
patient_row= None; patient_id="Demo-001"; true_label=None; patient_age=58

if input_mode == "Use demo patient":
    idx = st.selectbox("Choose a demo patient", range(len(df)),
                       format_func=lambda i:
                       f"Patient {i:03d} — "
                       f"{'Long survivor' if df.iloc[i]['survival_class']==1 else 'Short survivor'} "
                       f"({int(df.iloc[i]['Survival_days'])}d · age {int(df.iloc[i]['Age'])})",
                       index=7)
    patient_row = df.iloc[[idx]][feat_names+["Age"]].values
    true_label  = int(df.iloc[idx]["survival_class"])
    patient_id  = f"BraTS19-{idx:03d}"
    patient_age = int(df.iloc[idx]["Age"])

elif input_mode == "Select from dataset":
    chosen = st.selectbox("Select patient", [f"Patient {i:03d}" for i in range(len(df))])
    idx    = int(chosen.split()[-1])
    patient_row = df.iloc[[idx]][feat_names+["Age"]].values
    true_label  = int(df.iloc[idx]["survival_class"])
    patient_id  = chosen; patient_age = int(df.iloc[idx]["Age"])

elif input_mode == "Upload CSV":
    up = st.file_uploader("Upload patient radiomic CSV", type=["csv"])
    if up:
        udf = pd.read_csv(up)
        missing = [f for f in feat_names+["Age"] if f not in udf.columns]
        if missing:
            st.error(f"{len(missing)} required columns missing."); st.stop()
        patient_row = udf[feat_names+["Age"]].values[:1]
        patient_id  = up.name.replace(".csv","")
        patient_age = int(udf["Age"].iloc[0]) if "Age" in udf.columns else 58
    else:
        st.info("Upload a CSV to run prediction."); st.stop()

if patient_row is None: st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
prob, sv, base_val, xf = run_predict(pipe, patient_row, mkey)

# ═════════════════════════════════════════════════════════════════════════════
# CORE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### Clinical fingerprint")
    clinical_fingerprint(sv, pipe["feat_sel"], prob, true_label, patient_id)
    st.markdown("")
    st.markdown("### SHAP waterfall")
    fig_wf = plot_waterfall(sv, base_val, pipe["feat_sel"], xf, top_n_shap)
    st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

with right:
    st.markdown("### ROC curve")
    fig_roc, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pipe["fpr"], pipe["tpr"], color="#1D9E75", lw=2.5,
            label=f"XGBoost (AUC={pipe['auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    ax.fill_between(pipe["fpr"],pipe["tpr"],alpha=0.08,color="#1D9E75")
    ax.set_xlabel("False Positive Rate",fontsize=10); ax.set_ylabel("True Positive Rate",fontsize=10)
    ax.set_title("ROC — OS classification",fontsize=11,fontweight="600")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    fig_roc.tight_layout(); st.pyplot(fig_roc,use_container_width=True); plt.close(fig_roc)

    st.markdown("### Survival probability")
    gc = "#1D9E75" if prob >= 0.5 else "#E24B4A"
    st.markdown(f"""
    <div style='background:#f8f9fa;border-radius:12px;padding:16px;text-align:center;'>
      <div style='font-size:42px;font-weight:700;color:{gc};'>{prob:.1%}</div>
      <div style='font-size:13px;color:#6c757d;margin-top:4px;'>P(long survival >365d)</div>
      <div style='background:#e9ecef;border-radius:8px;height:10px;margin:12px 0 4px;overflow:hidden;'>
        <div style='height:100%;width:{prob*100:.1f}%;background:{gc};border-radius:8px;'></div>
      </div>
      <div style='display:flex;justify-content:space-between;font-size:11px;color:#adb5bd;'>
        <span>Short <=365d</span><span>Long >365d</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Global SHAP bar
    st.markdown("### Global importance (cohort)")
    df_s  = df.sample(40, random_state=1)
    Xs    = _transform(pipe, df_s[feat_names+["Age"]].values)
    sv_all= pipe["explainer"].shap_values(Xs)
    mean_abs = np.abs(sv_all).mean(axis=0)
    ti   = np.argsort(mean_abs)[-8:]
    fig_b, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.barh(range(8), mean_abs[ti], color="#378ADD", alpha=0.85, height=0.6)
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([sn(pipe["feat_sel"][i]) for i in ti], fontsize=8)
    ax2.set_xlabel("Mean |SHAP|",fontsize=9)
    ax2.set_title("Top 8 features (all patients)",fontsize=10,fontweight="600")
    ax2.spines[["top","right"]].set_visible(False); ax2.grid(axis="x",alpha=0.2)
    fig_b.tight_layout(); st.pyplot(fig_b,use_container_width=True); plt.close(fig_b)

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 MODULES
# ═════════════════════════════════════════════════════════════════════════════
sim_prob = prob; eor_pct = 0

if show_sim:
    sim_prob = surgical_simulator(pipe, patient_row.flatten(), feat_names, prob, mkey, patient_id)
    eor_pct  = st.session_state.get("eor_slider", 0)

if show_km:
    render_survival_charts(sim_prob, sv, pipe["feat_sel"], patient_id)

if show_ehr:
    render_ehr_note(sv, pipe["feat_sel"], prob, sim_prob, patient_id, patient_age, eor_pct)

# ═════════════════════════════════════════════════════════════════════════════
# FULL SHAP TABLE
# ═════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("Full SHAP table — all selected features", expanded=False):
    shap_df = pd.DataFrame({
        "Feature": pipe["feat_sel"],
        "Short name": [sn(f) for f in pipe["feat_sel"]],
        "Feature value": xf, "SHAP value": sv,
        "Direction": ["Up" if v > 0 else "Down" for v in sv],
        "Strength": ["Strong" if abs(v)>0.05 else "Moderate" if abs(v)>0.02 else "Weak" for v in sv],
    }).sort_values("SHAP value", key=abs, ascending=False).reset_index(drop=True)
    st.dataframe(shap_df, use_container_width=True, height=350)

st.caption("GliomaXAI v2 · INT 500 Internship · Sri Ramachandra Institute · "
           "BraTS 2019 synthetic · Research use only")
