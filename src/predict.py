"""
src/predict.py
==============
Prediction helpers for the GlucoSense Streamlit app.

This module is the single point of contact between the UI and the trained
model.  It loads the joblib bundle produced by ``src/train.py`` and exposes
two public functions:

- :func:`load_bundle` — cached model loader (call once at app startup)
- :func:`predict`     — apply imputation + scaling + predict on user input

Author : Pooja S (@POOJAS324)
"""

import numpy as np
import pandas as pd
import joblib
import shap


def load_bundle(model_path: str) -> dict:
    """
    Load the joblib bundle produced by ``src/train.py``.

    The bundle is a plain Python dict with the following keys:

    - ``model``             — trained best estimator
    - ``model_name``        — string name of the model class
    - ``imputer``           — fitted ``KNNImputer``
    - ``scaler``            — fitted ``StandardScaler``
    - ``explainer``         — fitted SHAP explainer
    - ``feature_names``     — list of feature column names (order matters)
    - ``X_test_scaled``     — scaled test-set array (for global SHAP plot)
    - ``shap_values_test``  — pre-computed positive-class SHAP values for
                              the test set (for global SHAP plot)

    Parameters
    ----------
    model_path : str
        Path to ``models/model.pkl``.

    Returns
    -------
    dict
        The loaded bundle.
    """
    return joblib.load(model_path)


def predict(user_df: pd.DataFrame, bundle: dict) -> tuple[int, float, np.ndarray]:
    """
    Apply the training-time transformations to *user_df* and return the
    model's prediction for a single patient.

    Steps
    -----
    1. Reorder / select columns to match training feature order.
    2. Apply the pre-fitted ``KNNImputer`` (handles any zeros the user
       left at the slider minimum).
    3. Apply the pre-fitted ``StandardScaler``.
    4. Call ``predict`` and ``predict_proba`` on the scaled input.
    5. Compute per-prediction SHAP values using the saved explainer.

    Parameters
    ----------
    user_df : pd.DataFrame
        Single-row DataFrame with the 8 raw feature values entered via
        the Streamlit sidebar sliders.
    bundle : dict
        The loaded model bundle (see :func:`load_bundle`).

    Returns
    -------
    label : int
        Binary prediction — 0 (not diabetic) or 1 (diabetic).
    proba : float
        Probability of the positive class (diabetes), in [0, 1].
    shap_values : np.ndarray, shape (1, n_features)
        Per-feature SHAP values for this single prediction.  Positive
        values push the risk up; negative values push it down.
    """
    feature_names = bundle["feature_names"]
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    explainer = bundle["explainer"]

    # 1. Select and order features to match training
    X_raw = user_df[feature_names].values  # shape (1, 8)

    # 2. Impute (handles zeros left by slider minimums)
    X_imputed = imputer.transform(X_raw)

    # 3. Scale
    X_scaled = scaler.transform(X_imputed)

    # 4. Predict
    label = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    # 5. SHAP values for this single observation
    shap_vals = explainer.shap_values(X_scaled)

    # Normalise to shape (1, n_features) for the positive class.
    # Older SHAP → list [class0, class1]; newer SHAP → ndarray (1, n_features, n_classes)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]                              # list format
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]                        # 3-D tensor format

    return label, proba, shap_vals
