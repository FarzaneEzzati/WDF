"""
test_split.py

Unit tests for src/data_prep/split.py. 
This test file checks if src/data_prep/split.py returns the correct 
split data without creating any deformation/leakage/lossing data points.
"""

import pandas as pd
import numpy as np
import pytest

from src.data_prep.split import time_based_split, random_split


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Date": pd.date_range("2010-01-01", periods=20, freq="W"),
        "feature1": range(20),
        "feature2": np.random.choice(a=[0, 1], size=20),
        "Weekly_Sales": [x * 10 for x in range(20)],
    })


# ---------------------------------------------------------------------------
# time_based_split
# ---------------------------------------------------------------------------

def test_no_date_overlap_between_train_and_test(sample_df):
    train_df, test_df = time_based_split(sample_df, date_column="Date", test_size=0.2)
    assert train_df["Date"].max() <= test_df["Date"].min() , (
        "Chronological split produced overlapping dates -- check for "
        "duplicate timestamps or an unsorted date column."
        )


def test_split_sizes_respect_test_size(sample_df):
    expected_test_size = 0.25
    train_df, test_df = time_based_split(sample_df, date_column="Date", test_size=expected_test_size)
    returned_test_size = round(len(sample_df) * 0.25)
    assert len(test_df) == returned_test_size, (
        f"Returned test size does not match the desired size. \
        Expected {expected_test_size} but got {returned_test_size}."
        )
    assert len(train_df) + len(test_df) == len(sample_df), (
        "Returned train and test data do not represent all data. Data is lost!"
    )


def test_returns_two_dataframes_when_no_target_given(sample_df):
    result = time_based_split(sample_df, date_column="Date", test_size=0.2)
    assert len(result) == 2, "Not the correct output when the target column is not given."
    train_df, test_df = result
    assert isinstance(train_df, pd.DataFrame), "Train data must be a pd.DataFrame instance."
    assert isinstance(test_df, pd.DataFrame), "Test data must be a pd.DataFrame instance."


def test_returns_xy_when_target_given(sample_df):
    X_train, X_test, y_train, y_test = time_based_split(
        sample_df, date_column="Date", test_size=0.2, target_column="Weekly_Sales"
    )
    assert "Weekly_Sales" not in X_train.columns, "Target column cannot be in the train data."
    assert "Weekly_Sales" not in X_test.columns, "Target column cannot be in the test data."
    assert len(X_train) == len(y_train), "X_train and y_train must be of the same size."
    assert len(X_test) == len(y_test), "X_test and y_test must be of the same size."


def test_invalid_test_size_raises(sample_df):
    with pytest.raises(ValueError):
        time_based_split(sample_df, test_size=1.5)
    with pytest.raises(ValueError):
        time_based_split(sample_df, test_size=0)


def test_missing_date_column_raises(sample_df):
    df_no_date = sample_df.drop(columns=["Date"])
    with pytest.raises(KeyError):
        time_based_split(df_no_date)


def test_missing_target_column_raises(sample_df):
    with pytest.raises(KeyError):
        time_based_split(sample_df, target_column="does_not_exist")


def test_split_is_sorted_by_date_even_if_input_is_shuffled(sample_df):
    shuffled_df = sample_df.sample(frac=1, random_state=1).reset_index(drop=True)
    train_df, test_df = time_based_split(shuffled_df, date_column="Date", test_size=0.2)
    assert train_df["Date"].max() <= test_df["Date"].min()
    assert train_df["Date"].is_monotonic_increasing
    assert test_df["Date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# random_split
# ---------------------------------------------------------------------------

def test_random_split_shapes(sample_df):
    X_train, X_test, y_train, y_test = random_split(sample_df, target_column="Weekly_Sales", test_size=0.2)
    assert len(X_train) + len(X_test) == len(sample_df), \
        "Returned train and test data do not represent all data. Data is lost!"
    assert "Weekly_Sales" not in X_train.columns, "Target column not in the input data."


def test_random_split_missing_target_raises(sample_df):
    with pytest.raises(KeyError):
        random_split(sample_df, target_column="does_not_exist")


def test_random_split_is_reproducible(sample_df):
    result1 = random_split(sample_df, target_column="Weekly_Sales", test_size=0.2, random_state=7)
    result2 = random_split(sample_df, target_column="Weekly_Sales", test_size=0.2, random_state=7)
    pd.testing.assert_frame_equal(result1[0], result2[0]), "random_split is not producible."