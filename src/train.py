
import json
import os
import sys
import warnings

# Ensure the console outputs UTF-8 on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Make sure src/ is importable when run as `python src/train.py` ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import (  # noqa: E402
    FEATURE_COLS,
    clean_and_impute,
    load_raw_data,
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "diabetes.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

RANDOM_STATE = 42


def build_param_grids() -> list[tuple]:
    
    return [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0]},
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
            },
        ),
        (
            "XGBClassifier",
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        ),
    ]


def get_shap_explainer(model, X_train_scaled: np.ndarray):
    
    model_name = type(model).__name__
    if model_name in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        # LinearExplainer works for logistic regression
        explainer = shap.LinearExplainer(
            model, X_train_scaled, feature_perturbation="interventional"
        )
    return explainer


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # -- 1. Load & clean data -------------------------------------------------
    print("=" * 60)
    print(f"GlucoSense - Training Pipeline")
    print("=" * 60)
    print("\n[1/6] Loading and cleaning data...")

    raw_df = load_raw_data(DATA_PATH)
    cleaned_df, fitted_imputer = clean_and_impute(raw_df)

    X = cleaned_df[FEATURE_COLS].values
    y = cleaned_df["Outcome"].values

    # ── 2. Train / test split ────────────────────────────────────────────────
    print("[2/6] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ── 3. Scale features ────────────────────────────────────────────────────
    print("[3/6] Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. Model comparison via GridSearchCV ─────────────────────────────────
    print("[4/6] Running GridSearchCV for 3 models...")
    param_grids = build_param_grids()
    comparison_results = []
    best_score = -1.0
    best_name = ""
    best_model = None

    for name, estimator, param_grid in param_grids:
        print(f"      Searching {name}...", end=" ", flush=True)
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train_scaled, y_train)
        cv_score = gs.best_score_
        print(f"best CV ROC-AUC = {cv_score:.4f}  params = {gs.best_params_}")

        comparison_results.append(
            {"model": name, "cv_roc_auc": round(float(cv_score), 4)}
        )

        if cv_score > best_score:
            best_score = cv_score
            best_name = name
            best_model = gs.best_estimator_

    print(f"\n  [OK] Best model: {best_name}  (CV ROC-AUC = {best_score:.4f})")

    # ── 5. Evaluate on held-out test set ─────────────────────────────────────
    print("\n[5/6] Evaluating best model on test set...")
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Confusion Matrix: {cm}")

    # ── 6. SHAP explainer + save bundle ──────────────────────────────────────
    print("\n[6/6] Building SHAP explainer and saving artifacts...")

    explainer = get_shap_explainer(best_model, X_train_scaled)

    # Pre-compute SHAP values for the entire test set (used for global
    # feature importance plot in the Model Insights tab — no recomputation
    # at Streamlit startup time)
    shap_values_test = explainer.shap_values(X_test_scaled)

    # Normalise to 2-D array of shape (n_samples, n_features) for the
    # positive class.  Older SHAP returns a list [class0, class1];
    # newer SHAP (>= 0.46) returns an ndarray of shape
    # (n_samples, n_features, n_classes) for tree ensembles.
    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]          # list format
    elif isinstance(shap_values_test, np.ndarray) and shap_values_test.ndim == 3:
        shap_values_test = shap_values_test[:, :, 1]   # 3-D tensor format

    # Save joblib bundle
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "imputer": fitted_imputer,
        "scaler": scaler,
        "explainer": explainer,
        "feature_names": FEATURE_COLS,
        "X_test_scaled": X_test_scaled,
        "shap_values_test": shap_values_test,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"  [OK] Model bundle saved -> {MODEL_PATH}")

    # Save metrics JSON
    metrics = {
        "best_model_name": best_name,
        "test_set": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm,
        },
        "roc_curve": {
            "fpr": [round(float(v), 6) for v in fpr],
            "tpr": [round(float(v), 6) for v in tpr],
        },
        "model_comparison": comparison_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [OK] Metrics saved    -> {METRICS_PATH}")

    print("\n[DONE] Training complete! Run `streamlit run app.py` to launch the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
