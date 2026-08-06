"""
src/data_preprocessing.py
==========================
Data loading and cleaning functions for the GlucoSense ML pipeline.

WHY THIS CLEANING STEP EXISTS
------------------------------
The Pima Indians Diabetes Dataset uses ``0`` as a sentinel value for
*missing* measurements in several physiological columns where a true zero
is biologically impossible (e.g. a BMI of 0 or a glucose level of 0 would
be fatal). Simply leaving those zeros in the dataset would allow the model
to learn spurious patterns from them and would badly bias imputation-free
baselines. We therefore:

  1. Replace zeros with NaN **only** in the affected columns
     (Glucose, BloodPressure, SkinThickness, Insulin, BMI).
  2. Impute the missing values with a KNNImputer (k=5), which leverages
     the correlation structure between features for a more realistic
     estimate than a global mean/median.

Columns where 0 *is* a valid value — ``Pregnancies`` and ``Outcome`` —
are intentionally left untouched.

Author : Pooja S (@POOJAS324)
"""

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
    """
    Load the Pima Indians Diabetes Dataset from *path* and return the
    raw DataFrame (zeros still present, no cleaning applied).

    Parameters
    ----------
    path : str
        Relative or absolute path to ``diabetes.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 768 rows and 9 columns
        (8 features + ``Outcome``).
    """
    df = pd.read_csv(path)
    return df


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace biologically impossible zeros with NaN in the target columns.

    Only ``ZERO_AS_NAN_COLS`` are affected; ``Pregnancies`` and
    ``Outcome`` are left unchanged because 0 is a valid value for them.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by :func:`load_raw_data`.

    Returns
    -------
    pd.DataFrame
        A *copy* of ``df`` with zeros replaced by NaN in the relevant
        columns.
    """
    df_copy = df.copy()
    df_copy[ZERO_AS_NAN_COLS] = df_copy[ZERO_AS_NAN_COLS].replace(0, np.nan)
    return df_copy


def clean_and_impute(
    df: pd.DataFrame,
    imputer: KNNImputer | None = None,
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, KNNImputer]:
    """
    Replace impossible zeros with NaN, then impute them with KNNImputer.

    If *imputer* is ``None`` a new ``KNNImputer`` is **fitted** on the
    feature columns of *df* and returned alongside the cleaned data.
    Pass a pre-fitted imputer (loaded from disk) when transforming new
    single-row user input at prediction time, so the imputation is
    identical to what was done during training.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or already-NaN-ised) dataset.  Must contain all
        ``FEATURE_COLS`` plus ``Outcome``.
    imputer : KNNImputer or None
        Pre-fitted imputer to reuse.  Pass ``None`` to fit a fresh one.
    n_neighbors : int
        Number of neighbours for KNN imputation (ignored when a
        pre-fitted *imputer* is supplied).

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataset with no NaN values; same shape as input.
    imputer : KNNImputer
        Fitted imputer (either the one passed in or the newly fitted one).
    """
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
