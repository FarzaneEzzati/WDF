"""
split.py

Train/test split logic for the pipeline.

This is time-series data. A random train/test split would let the model 
train on rows *after* the test period and then be evaluated on rows 
*before* it (classic data leakage). Hence chronological split it used (time_based_split).

A random split function is also included for cases where it's genuinely
appropriate (e.g. quick experimentation with a non-temporal subset of
features), but it is not used for the real model comparison.
"""

import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split
import logging

logger = logging.getLogger(__name__)

def time_based_split(
    df: pd.DataFrame,
    date_column: str = "Date",
    test_size: float = 0.2,
    target_column: str = None,
):
    """
    Split a DataFrame chronologically: the earliest (1 - test_size) of
    rows (by date) go to train, the most recent test_size go to test.
    This mirrors how the model will actually be used -- trained on the
    past, evaluated on the future.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `date_column`.
    date_column : str
        Name of the datetime column to sort/split on.
    test_size : float
        Fraction of rows (by time) to hold out for testing. Must be in (0, 1).
    target_column : str, optional
        If provided, the function also splits off X/y and returns
        X_train, X_test, y_train, y_test. If omitted, returns two full
        DataFrames: train_df, test_df.

    Returns
    -------
    Either (train_df, test_df) or (X_train, X_test, y_train, y_test),
    depending on whether target_column is given.
    """
    logger.info(f"Splitting dataset with test_size={test_size}...")
    # Initial tests
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    if date_column not in df.columns:
        raise KeyError(f"'{date_column}' not found in DataFrame columns: {list(df.columns)}")
    if target_column not in df.columns:
        raise KeyError(f"target_column '{target_column}' not found in DataFrame columns.")
    
    # Sort by date and split using test size
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    # Guard against an accidental leak: every train date should be <= every test date
    if len(train_df) and len(test_df):
        assert train_df[date_column].max() <= test_df[date_column].min(), (
            "Chronological split produced overlapping dates -- check for "
            "duplicate timestamps or an unsorted date column."
        )

    if target_column is None:
        return train_df, test_df

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return X_train, X_test, y_train, y_test


def random_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Standard random train/test split (via sklearn). Use ONLY when rows are
    truly independent of each other (no time-ordering to respect). 

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise KeyError(f"target_column '{target_column}' not found in DataFrame columns.")
    if not 0 < test_size < 1:
        raise KeyError(f"test_size must be between 0 and 1, got {test_size}")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)