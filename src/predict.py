

import numpy as np
import pandas as pd
import joblib
import shap


def load_bundle(model_path: str) -> dict:
    
    return joblib.load(model_path)


def predict(user_df: pd.DataFrame, bundle: dict) -> tuple[int, float, np.ndarray]:
    
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
