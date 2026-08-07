

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Make sure src/ is importable when running `pytest` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_preprocessing import (
    FEATURE_COLS,
    ZERO_AS_NAN_COLS,
    clean_and_impute,
    load_raw_data,
    replace_zeros_with_nan,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
   
    np.random.seed(0)
    n = 10
    data = {
        "Pregnancies": np.random.randint(0, 5, n),           # 0 is valid here
        "Glucose": [0, 120, 0, 95, 115, 140, 0, 110, 80, 130],  # 0 = missing
        "BloodPressure": [70, 0, 68, 0, 72, 80, 65, 0, 75, 82],  # 0 = missing
        "SkinThickness": [20, 35, 0, 25, 0, 40, 30, 22, 0, 38],  # 0 = missing
        "Insulin": [0, 150, 80, 0, 200, 0, 90, 0, 120, 180],     # 0 = missing
        "BMI": [28.5, 0.0, 31.2, 24.7, 0.0, 35.0, 29.1, 27.8, 0.0, 33.4],
        "DiabetesPedigreeFunction": np.round(
            np.random.uniform(0.1, 1.5, n), 3
        ),
        "Age": np.random.randint(21, 60, n),
        "Outcome": np.random.randint(0, 2, n),
    }
    return pd.DataFrame(data)


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestReplaceZerosWithNaN:
    

    def test_zeros_replaced_in_target_columns(self, sample_df):
       
        result = replace_zeros_with_nan(sample_df)
        for col in ZERO_AS_NAN_COLS:
            original_zeros = (sample_df[col] == 0).sum()
            if original_zeros > 0:
                assert result[col].isna().sum() == original_zeros, (
                    f"Column '{col}': expected {original_zeros} NaN(s), "
                    f"got {result[col].isna().sum()}"
                )

    def test_pregnancies_zeros_untouched(self, sample_df):
      
        result = replace_zeros_with_nan(sample_df)
        # All positions that were 0 in Pregnancies should still be 0, not NaN
        preg_zeros_mask = sample_df["Pregnancies"] == 0
        assert not result.loc[preg_zeros_mask, "Pregnancies"].isna().any(), (
            "Pregnancies column should not have NaN where original value was 0"
        )

    def test_outcome_zeros_untouched(self, sample_df):
        
        result = replace_zeros_with_nan(sample_df)
        outcome_zeros_mask = sample_df["Outcome"] == 0
        assert not result.loc[outcome_zeros_mask, "Outcome"].isna().any(), (
            "Outcome column should not have NaN where original value was 0"
        )

    def test_original_df_not_mutated(self, sample_df):
        
        original_glucose = sample_df["Glucose"].copy()
        replace_zeros_with_nan(sample_df)
        pd.testing.assert_series_equal(sample_df["Glucose"], original_glucose)


class TestCleanAndImpute:
   

    def test_no_nans_after_imputation(self, sample_df):
       
        cleaned_df, _ = clean_and_impute(sample_df)
        nan_count = cleaned_df[FEATURE_COLS].isna().sum().sum()
        assert nan_count == 0, (
            f"Expected 0 NaNs after imputation, found {nan_count}"
        )

    def test_row_count_preserved(self, sample_df):
        
        cleaned_df, _ = clean_and_impute(sample_df)
        assert len(cleaned_df) == len(sample_df), (
            f"Row count changed: input={len(sample_df)}, output={len(cleaned_df)}"
        )

    def test_column_set_preserved(self, sample_df):
       
        cleaned_df, _ = clean_and_impute(sample_df)
        assert set(cleaned_df.columns) == set(sample_df.columns), (
            f"Column mismatch: {set(cleaned_df.columns)} != {set(sample_df.columns)}"
        )

    def test_imputer_returned(self, sample_df):
       
        from sklearn.impute import KNNImputer
        _, imputer = clean_and_impute(sample_df)
        assert isinstance(imputer, KNNImputer), (
            f"Expected KNNImputer, got {type(imputer)}"
        )

    def test_pre_fitted_imputer_accepted(self, sample_df):
       
        _, fitted_imputer = clean_and_impute(sample_df)
        cleaned_df2, _ = clean_and_impute(sample_df, imputer=fitted_imputer)
        assert cleaned_df2[FEATURE_COLS].isna().sum().sum() == 0
