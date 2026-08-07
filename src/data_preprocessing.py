

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


# Columns where 0 is medically impossible and must be treated as NaN
ZERO_AS_NAN_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# All 8 feature columns (no target)
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def load_raw_data(path: str = "diabetes.csv") -> pd.DataFrame:
    
    df = pd.read_csv(path)
    return df


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
   
    df_copy = df.copy()
    df_copy[ZERO_AS_NAN_COLS] = df_copy[ZERO_AS_NAN_COLS].replace(0, np.nan)
    return df_copy


def clean_and_impute(
    df: pd.DataFrame,
    imputer: KNNImputer | None = None,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, KNNImputer]:
    
    # Step 1 — replace impossible zeros with NaN
    df_nan = replace_zeros_with_nan(df)

    features = df_nan[FEATURE_COLS]

    if imputer is None:
        # Fit a new imputer on the training data
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_values = imputer.fit_transform(features)
    else:
        # Use the pre-fitted imputer (inference / test time)
        imputed_values = imputer.transform(features)

    # Rebuild as DataFrame to preserve column names
    cleaned_features = pd.DataFrame(
        imputed_values, columns=FEATURE_COLS, index=df_nan.index
    )

    # Re-attach the Outcome column (unchanged)
    cleaned_df = cleaned_features.copy()
    if "Outcome" in df_nan.columns:
        cleaned_df["Outcome"] = df_nan["Outcome"].values

    return cleaned_df, imputer
