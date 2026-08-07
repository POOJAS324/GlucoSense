"""
src/predict.py
==============
Prediction helpers for the GlucoSense Streamlit app.

This module loads the joblib bundle produced by ``src/train.py`` and exposes:
- :func:`load_bundle` — load the saved model bundle
- :func:`predict`     — apply imputation + scaling + predict + SHAP on user input

Author : Pooja S (@POOJAS324)
"""

import numpy as np
import pandas as pd
import joblib


def load_bundle(model_path: str) -> dict:
    """Load the joblib bundle produced by src/train.py."""
    return joblib.load(model_path)


def predict(user_df: pd.DataFrame, bundle: dict) -> tuple:
    """
    Apply training-time transformations to user_df and return the prediction.

    Returns
    -------
    label : int          — 0 (not diabetic) or 1 (diabetic)
    proba : float        — probability of positive class, in [0, 1]
    shap_vals : ndarray or None
        Per-feature SHAP values (1, n_features), or None if SHAP is unavailable
        on this platform (e.g. broken C extension on certain Python versions).
    """
    feature_names = bundle["feature_names"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    explainer = bundle["explainer"]

    # 1. Select and order features to match training order
    X_raw = user_df[feature_names].values  # shape (1, 8)

    # 2. Impute (handles zeros left at slider minimums)
    X_imputed = imputer.transform(X_raw)

    # 3. Scale
    X_scaled = scaler.transform(X_imputed)

    # 4. Predict label and probability
    label = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    # 5. SHAP values for this single observation
    # Wrapped in try/except so the whole app never crashes if the SHAP
    # C extension fails to load (e.g. broken ABI on Python 3.14+).
    shap_vals = None
    try:
        raw = explainer.shap_values(X_scaled)

        # Normalise to shape (1, n_features) for the positive class.
        # Older SHAP  → list [class0_vals, class1_vals]
        # Newer SHAP  → ndarray of shape (1, n_features, n_classes)
        if isinstance(raw, list):
            shap_vals = raw[1]                     # list format
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            shap_vals = raw[:, :, 1]               # 3-D tensor format
        else:
            shap_vals = raw                        # already 2-D
    except Exception:
        # SHAP explainer unavailable on this platform — prediction still works
        shap_vals = None

    return label, proba, shap_vals
