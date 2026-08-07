"""
GlucoSense – Diabetes Risk Prediction System
=============================================
Multi-tab Streamlit UI.  This file contains ONLY the UI layer.
All ML logic lives in ``src/``.

Tabs
----
- 🩺 Predict      — real-time risk prediction + SHAP explanation + scatter plots
- 📊 Model Insights — saved test-set metrics, confusion matrix, ROC curve,
                     global SHAP importance, model comparison table
- ℹ️ About         — project description, dataset source, tech stack

Run the app
-----------
    # Train the model first (once):
    python src/train.py

    # Then launch the app:
    streamlit run app.py

Author : Pooja S (@POOJAS324)
License: MIT
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import json
import os
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# Make sure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))
from src.data_preprocessing import load_raw_data, clean_and_impute  # noqa: E402
from src.predict import load_bundle, predict  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for Streamlit

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model.pkl")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")
DATA_PATH = os.path.join(ROOT_DIR, "diabetes.csv")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoSense – Diabetes Risk Prediction",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached loaders ────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model bundle …")
def get_bundle():
    """Load the joblib model bundle once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return load_bundle(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics():
    """Load the saved test-set metrics once per session."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_raw_df():
    """Load and return the original (uncleaned) dataset for scatter plots."""
    return load_raw_data(DATA_PATH)


# ── Sidebar – patient input ───────────────────────────────────────────────────
st.sidebar.header("🩺 Patient Data")
st.sidebar.markdown("Adjust the sliders to enter your health metrics.")


def collect_user_inputs() -> pd.DataFrame:
    """
    Render sidebar sliders and return the user-entered health metrics
    as a single-row DataFrame matching the training feature order.
    """
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose (mg/dL)", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin (μU/mL)", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 20.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.47)
    age = st.sidebar.slider("Age", 21, 88, 33)

    return pd.DataFrame(
        {
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age],
        }
    )


user_data = collect_user_inputs()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩸 GlucoSense")
st.markdown("*AI-powered diabetes risk prediction from your health metrics*")

# ── Load assets ───────────────────────────────────────────────────────────────
bundle = get_bundle()
metrics = get_metrics()
df_raw = get_raw_df()

# ── Warning if model not yet trained ─────────────────────────────────────────
model_ready = bundle is not None

if not model_ready:
    st.warning(
        "⚠️ **Model not found.** Please run `python src/train.py` once "
        "before launching the app.  The Model Insights and Predict tabs "
        "will activate automatically once training is complete.",
        icon="⚠️",
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_insights, tab_about = st.tabs(
    ["🩺 Predict", "📊 Model Insights", "ℹ️ About"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Your Health Metrics")
    st.dataframe(user_data, use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if model_ready:
        label, proba, shap_vals = predict(user_data, bundle)
        risk_pct = proba * 100

        st.markdown("---")
        st.subheader("🔬 Diabetes Risk Estimate")

        # Color-coded risk level
        if risk_pct < 30:
            risk_label = "🟢 Low Risk"
            bar_color = "#22c55e"
        elif risk_pct < 60:
            risk_label = "🟡 Moderate Risk"
            bar_color = "#f59e0b"
        else:
            risk_label = "🔴 High Risk"
            bar_color = "#ef4444"

        col_gauge, col_detail = st.columns([1, 1])

        with col_gauge:
            st.markdown(
                f"""
                <div style="
                    background: #1E293B;
                    border-radius: 16px;
                    padding: 28px 32px;
                    text-align: center;
                    border: 1px solid #334155;
                ">
                    <div style="font-size: 56px; font-weight: 800;
                                color: {bar_color}; line-height: 1;">
                        {risk_pct:.1f}%
                    </div>
                    <div style="font-size: 20px; font-weight: 600;
                                color: #F1F5F9; margin-top: 8px;">
                        Estimated Diabetes Risk
                    </div>
                    <div style="font-size: 18px; margin-top: 12px;">
                        {risk_label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Progress bar
            st.markdown(
                f"""
                <div style="
                    background: #334155;
                    border-radius: 8px;
                    height: 18px;
                    margin-top: 16px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {risk_pct:.1f}%;
                        height: 100%;
                        background: linear-gradient(90deg, {bar_color}88, {bar_color});
                        border-radius: 8px;
                        transition: width 0.4s ease;
                    "></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_detail:
            st.markdown("#### Interpretation")
            if label == 0:
                st.success(
                    "✅ Based on the entered metrics, the model predicts a "
                    "**non-diabetic** outcome.  Risk probability is below 50 %."
                )
            else:
                st.error(
                    "⚠️ Based on the entered metrics, the model predicts a "
                    "**diabetic** outcome.  Risk probability exceeds 50 %."
                )

            st.info(
                "This tool is for educational purposes only and is **not** a "
                "substitute for professional medical advice.",
                icon="ℹ️",
            )

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 What Drove This Prediction? (SHAP)")
        st.caption(
            "Positive values (red) pushed the risk **up**; "
            "negative values (blue) pushed it **down**."
        )

        feature_names = bundle["feature_names"]

        try:
            sv = shap_vals[0] if shap_vals.ndim == 2 else shap_vals  # (n_features,)
            fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
            fig_shap.patch.set_facecolor("#0F172A")
            ax_shap.set_facecolor("#1E293B")

            colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sv]
            y_pos = np.arange(len(feature_names))
            bars = ax_shap.barh(y_pos, sv, color=colors, edgecolor="none", height=0.6)
            ax_shap.set_yticks(y_pos)
            ax_shap.set_yticklabels(feature_names, color="#F1F5F9", fontsize=10)
            ax_shap.set_xlabel("SHAP value (impact on model output)", color="#94a3b8")
            ax_shap.tick_params(colors="#94a3b8")
            ax_shap.axvline(0, color="#475569", linewidth=1)
            for spine in ax_shap.spines.values():
                spine.set_edgecolor("#334155")
            ax_shap.set_title(
                "Per-feature SHAP contribution", color="#F1F5F9", fontsize=11
            )
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close(fig_shap)
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")

    else:
        st.info("Run `python src/train.py` to enable predictions.")

    # ── Feature comparison scatter plots ──────────────────────────────────────
    st.markdown("---")
    st.title("📈 Visualised Patient Report")
    st.caption("Your data point is shown as a larger marker. Blue = healthy, Red = diabetic.")

    if model_ready:
        marker_color = "blue" if label == 0 else "red"
    else:
        marker_color = "gray"

    def plot_feature_comparison(
        feature: str,
        palette: str,
        y_ticks_range: tuple,
    ) -> None:
        """
        Scatter plot: dataset coloured by Outcome vs user's single point.
        Mirrors the original app's visualisation faithfully.
        """
        st.header(f"{feature} Graph (Others vs Yours)")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x="Age", y=feature, data=df_raw, hue="Outcome", palette=palette, ax=ax
        )
        sns.scatterplot(
            x=user_data["Age"],
            y=user_data[feature],
            s=150,
            color=marker_color,
            ax=ax,
        )
        ax.set_xticks(np.arange(10, 100, 5))
        ax.set_yticks(np.arange(*y_ticks_range))
        ax.set_title("0 - Healthy & 1 - Diabetic")
        st.pyplot(fig)
        plt.close(fig)

    plot_feature_comparison("Pregnancies", "Greens", (0, 20, 2))
    plot_feature_comparison("Glucose", "magma", (0, 220, 10))
    plot_feature_comparison("BloodPressure", "Reds", (0, 130, 10))
    plot_feature_comparison("SkinThickness", "Blues", (0, 110, 10))
    plot_feature_comparison("Insulin", "rocket", (0, 900, 50))
    plot_feature_comparison("BMI", "rainbow", (0, 70, 5))
    plot_feature_comparison("DiabetesPedigreeFunction", "YlOrBr", (0, 3, 0.2))

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_insights:
    st.subheader("📊 Model Insights")

    if not model_ready or metrics is None:
        st.info(
            "Run `python src/train.py` to generate model metrics.  "
            "This tab will populate automatically once training is complete."
        )
    else:
        model_name = metrics.get("best_model_name", "Unknown")
        st.markdown(f"**Best model selected:** `{model_name}`")
        st.caption(
            "The model was chosen by highest cross-validated ROC-AUC across "
            "LogisticRegression, RandomForestClassifier, and XGBClassifier."
        )

        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown("#### Test-Set Performance")
        ts = metrics["test_set"]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label_, value in zip(
            [c1, c2, c3, c4, c5],
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            [ts["accuracy"], ts["precision"], ts["recall"], ts["f1"], ts["roc_auc"]],
        ):
            with col:
                st.metric(label_, f"{value:.3f}")

        st.markdown("---")

        # ── Confusion matrix + ROC curve (side by side) ───────────────────────
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm_arr = np.array(ts["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            fig_cm.patch.set_facecolor("#0F172A")
            ax_cm.set_facecolor("#1E293B")
            sns.heatmap(
                cm_arr,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax_cm,
                xticklabels=["Not Diabetic", "Diabetic"],
                yticklabels=["Not Diabetic", "Diabetic"],
                linewidths=0.5,
                linecolor="#334155",
                cbar=False,
            )
            ax_cm.set_xlabel("Predicted", color="#F1F5F9")
            ax_cm.set_ylabel("Actual", color="#F1F5F9")
            ax_cm.tick_params(colors="#F1F5F9")
            ax_cm.set_title("Confusion Matrix", color="#F1F5F9")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_roc:
            st.markdown("#### ROC Curve")
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3.5))
            fig_roc.patch.set_facecolor("#0F172A")
            ax_roc.set_facecolor("#1E293B")
            ax_roc.plot(fpr, tpr, color="#0D9488", lw=2,
                        label=f"AUC = {ts['roc_auc']:.3f}")
            ax_roc.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--")
            ax_roc.set_xlabel("False Positive Rate", color="#94a3b8")
            ax_roc.set_ylabel("True Positive Rate", color="#94a3b8")
            ax_roc.tick_params(colors="#94a3b8")
            ax_roc.set_title("ROC Curve", color="#F1F5F9")
            ax_roc.legend(facecolor="#1E293B", labelcolor="#F1F5F9")
            for spine in ax_roc.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)

        st.markdown("---")

        # ── Global SHAP summary plot ──────────────────────────────────────────
        st.markdown("#### Global Feature Importance (SHAP)")
        st.caption(
            "Each bar shows the mean absolute SHAP value across the test set — "
            "a model-agnostic measure of how much each feature influences the "
            "prediction on average.  **Glucose** and **BMI** are typically the "
            "strongest predictors of diabetes risk in this dataset."
        )

        try:
            shap_vals_test = bundle["shap_values_test"]  # ideally (n_test, n_features)
            feature_names = bundle["feature_names"]

            # Defensive normalisation: in case the bundle was built with an
            # older train.py that saved a 3-D array (n_samples, n_features, n_classes)
            if isinstance(shap_vals_test, np.ndarray) and shap_vals_test.ndim == 3:
                shap_vals_test = shap_vals_test[:, :, 1]  # take positive class
            elif isinstance(shap_vals_test, list):
                shap_vals_test = shap_vals_test[1]        # older SHAP list format

            mean_abs_shap = np.abs(shap_vals_test).mean(axis=0)  # (n_features,)
            sorted_idx = np.argsort(mean_abs_shap)

            fig_gi, ax_gi = plt.subplots(figsize=(7, 4))
            fig_gi.patch.set_facecolor("#0F172A")
            ax_gi.set_facecolor("#1E293B")
            top2 = set(sorted_idx[-2:].tolist())
            bar_colors = [
                "#0D9488" if i in top2 else "#334155"
                for i in range(len(feature_names))
            ]
            ax_gi.barh(
                [feature_names[i] for i in sorted_idx],
                mean_abs_shap[sorted_idx],
                color=[bar_colors[i] for i in sorted_idx],
                edgecolor="none",
            )
            ax_gi.set_xlabel("Mean |SHAP value|", color="#94a3b8")
            ax_gi.tick_params(colors="#F1F5F9")
            ax_gi.set_title("Global Feature Importance (mean |SHAP|)", color="#F1F5F9")
            for spine in ax_gi.spines.values():
                spine.set_edgecolor("#334155")
            plt.tight_layout()
            st.pyplot(fig_gi)
            plt.close(fig_gi)
        except Exception as e:
            st.warning(f"SHAP summary plot unavailable: {e}")

        st.markdown("---")

        # ── Model comparison table ────────────────────────────────────────────
        st.markdown("#### Model Comparison — Cross-Validated ROC-AUC (5-fold)")
        st.caption(
            "All three models were evaluated with GridSearchCV before the best "
            "one was selected — no arbitrary choice was made."
        )
        comparison = metrics.get("model_comparison", [])
        if comparison:
            comp_df = pd.DataFrame(comparison).sort_values(
                "cv_roc_auc", ascending=False
            )
            comp_df.columns = ["Model", "CV ROC-AUC (5-fold)"]
            comp_df = comp_df.reset_index(drop=True)
            comp_df.index += 1  # 1-based ranking
            st.dataframe(comp_df, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About GlucoSense")

    st.markdown(
        """
**GlucoSense** is an end-to-end ML project that predicts the probability of
diabetes in female patients of Pima Indian heritage using eight clinical
features.  It is designed as a clean, reproducible, and readable codebase —
suitable as a portfolio or interview reference project.

---

### 📋 Dataset
- **Name:** Pima Indians Diabetes Dataset
- **Source:** [National Institute of Diabetes and Digestive and Kidney Diseases](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness,
  Insulin, BMI, Diabetes Pedigree Function, Age
- **Target:** Outcome — 0 (non-diabetic), 1 (diabetic)

> **Data quality note:** The dataset uses `0` as a sentinel for missing
> values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI —
> columns where a true zero is biologically impossible.  GlucoSense
> replaces these with `NaN` and imputes them using a **KNNImputer**,
> preserving the correlation structure between features.

---

### 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend / UI | Streamlit |
| ML Models | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Data Handling | pandas, NumPy |
| Imputation | scikit-learn KNNImputer |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Serialisation | joblib |
| Testing | pytest |
| Visualisation | Matplotlib, Seaborn |
| Language | Python 3.10+ |

---




### 👩‍💻 Author
**Pooja S** — [@POOJAS324](https://github.com/POOJAS324)
        """
    )

# ── Custom footer / badge (preserved exactly from original) ──────────────────
st.markdown(
    """
    <style>
    /* Hide Streamlit default footer and main menu */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .custom-badge-container {
        position: fixed;
        bottom: 0;
        right: 0;
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 999999;
        pointer-events: none;
    }
    .custom-avatar {
        pointer-events: auto;
    }
    .custom-avatar img {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        border: 2px solid #1E293B;
        display: block;
        margin-bottom: 8px;
    }
    .custom-badge {
        pointer-events: auto;
        background-color: #FF4B4B;
        color: white;
        padding: 12px 20px 12px 16px;
        border-top-left-radius: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: -2px -2px 10px rgba(0, 0, 0, 0.15);
    }
    .custom-badge-text {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 500;
        font-size: 15px;
        white-space: nowrap;
    }
    .custom-badge-logo {
        width: 20px;
        height: 20px;
        filter: brightness(0) invert(1);
    }
    </style>
    <div class="custom-badge-container">
        <div class="custom-avatar">
            <img src="https://github.com/POOJAS324.png" alt="Avatar">
        </div>
        <div class="custom-badge">
            <span class="custom-badge-text">Hosted with Streamlit</span>
            <img class="custom-badge-logo" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" alt="Streamlit Logo">
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
