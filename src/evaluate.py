"""
evaluate.py

Evaluation of the prediction models in the pipeline.
As the logic behind model evaluation is the same across models ...
The evaluation is based on the following criteria:
* 
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """
    Computes standard regression metrics.
    
    Returns
    dict: dict with keys: rmse, mae, r2, mape
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch for predictions: expected {y_true.shape}, got {y_pred.shape}"
        )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    nnz_mask = y_true != 0
    if nnz_mask.sum() == 0:
        mape = np.nan
    else:
        diff = y_pred[nnz_mask] - y_true[nnz_mask]
        relative_diff = diff / y_true[nnz_mask]
        mape = np.mean(np.abs(relative_diff)) * 100

    return {'rmse': rmse, 'mae': mae,'r2': r2,'mape': mape}


def evaluate(model_name: str, y_true, y_pred) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    return dict({'model': model_name, **metrics})

def evaluate_baseline(y_true) -> dict:
    y_true = np.asarray(y_true)
    mean_pred = np.full_like(y_true, fill_value=y_true.mean(), dtype=float)
    return evaluate("Baseline (mean)", y_true, mean_pred)

def evaluations_to_dataframe(evals: list) -> pd.DataFrame:
    """
    Converts a list of evaluate()/baseline_metrics() into a data frame
    for comparison purpose.
    """
    df = pd.DataFrame(evals)
    return df.sort_values("rmse").reset_index(drop=True)
