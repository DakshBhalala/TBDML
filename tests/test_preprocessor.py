"""Unit tests for TabularPreprocessor and data schema."""

import numpy as np
import pandas as pd

from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.schema import DatasetSchema, TaskType


def test_numerical_and_categorical_preprocessing():
    df_train = pd.DataFrame(
        {
            "num_1": [1.0, 2.0, np.nan, 4.0, 5.0, 10.0],
            "num_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "cat_1": ["A", "B", "A", "C", np.nan, "B"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.BINARY,
        numerical_cols=["num_1", "num_2"],
        categorical_cols=["cat_1"],
    )

    preprocessor = TabularPreprocessor(schema=schema, num_bins=4)
    preprocessor.fit(df_train, df_train["target"])

    assert preprocessor.is_fitted_
    assert "num_1" in preprocessor.num_medians_
    assert "cat_1" in preprocessor.cat_vocabs_
    # OOV should be index 0
    assert preprocessor.cat_vocabs_["cat_1"]["__UNKNOWN__"] == 0

    num_vals, num_masks, cat_indices, y_trans = preprocessor.transform(df_train, df_train["target"])

    assert num_vals.shape == (6, 2)
    assert num_masks.shape == (6, 2)
    # Row 2 in num_1 was NaN -> should be imputed with median, mask=1.0
    assert num_masks[2, 0] == 1.0
    assert num_masks[0, 0] == 0.0
    assert cat_indices.shape == (6, 1)
    assert y_trans.shape == (6,)


def test_unseen_categories_and_missing_values_in_test():
    df_train = pd.DataFrame(
        {
            "num_1": [1.0, 2.0, 3.0, 4.0],
            "cat_1": ["X", "Y", "Z", "X"],
            "target": [10.0, 20.0, 30.0, 40.0],
        }
    )

    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.REGRESSION,
        numerical_cols=["num_1"],
        categorical_cols=["cat_1"],
    )

    preprocessor = TabularPreprocessor(schema=schema, num_bins=4)
    preprocessor.fit(df_train, df_train["target"])

    df_test = pd.DataFrame(
        {
            "num_1": [np.nan, 100.0],
            "cat_1": ["UNKNOWN_CAT", "Y"],
        }
    )

    num_vals, num_masks, cat_indices, _ = preprocessor.transform(df_test)

    # UNKNOWN_CAT should be mapped to 0 (OOV)
    assert cat_indices[0, 0] == 0
    # "Y" should be mapped to non-zero valid ID
    assert cat_indices[1, 0] > 0
    # Missing num_1 in test row 0 should be flagged in mask
    assert num_masks[0, 0] == 1.0
    assert num_masks[1, 0] == 0.0


def test_regression_target_inverse_transform():
    y_raw = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5], "target": y_raw})
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.REGRESSION,
        numerical_cols=["num"],
    )

    preprocessor = TabularPreprocessor(schema=schema)
    preprocessor.fit(df, y_raw)

    _, _, _, y_scaled = preprocessor.transform(df, y_raw)
    assert np.isclose(np.mean(y_scaled), 0.0, atol=1e-5)

    y_recovered = preprocessor.inverse_transform_target(y_scaled)
    assert np.allclose(y_raw, y_recovered, atol=1e-5)


def test_data_leakage_isolation():
    """Verify that transforming test data does NOT alter training parameters."""
    df_train = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    schema = DatasetSchema(target_col="target", task_type=TaskType.BINARY, numerical_cols=["num"])
    preprocessor = TabularPreprocessor(schema=schema, num_bins=4)
    preprocessor.fit(df_train, df_train["target"])

    orig_median = preprocessor.num_medians_["num"]
    orig_edges = preprocessor.num_bin_edges_["num"].copy()

    # Extreme outlier in test data
    df_test = pd.DataFrame({"num": [999999.0, -999999.0]})
    preprocessor.transform(df_test)

    # Ensure preprocessor statistics remain pristine
    assert preprocessor.num_medians_["num"] == orig_median
    assert np.array_equal(preprocessor.num_bin_edges_["num"], orig_edges)
