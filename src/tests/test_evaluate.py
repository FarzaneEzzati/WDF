"""
test_evaluate.py
 
Unit tests for src/evaluate.py. 
"""

import numpy as np
import pytest

from src.evaluate import compute_metrics, evaluate_baseline, evaluate, evaluations_to_dataframe

def test_perfect_predictions_give_zero_error():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([10, 20, 30, 40])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1
    assert metrics["mape"] == 0

def test_known_mae_value():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 22, 28, 38])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mape"] == pytest.approx(2.0)

def test_metrics_return_expected_keys():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 22, 28, 38])
    metrics = compute_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {"rmse", "mae", "r2", "mape"}

def test_shape_mismatch():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1, 2, 3]), np.array([1, 2]))

def test_mape_handles_zero_true_values():
    y_true = np.array([0, 20, 30, 40])
    y_pred = np.array([12, 22, 28, 38])
    metrics = compute_metrics(y_true, y_pred)
    assert np.isfinite(metrics["mape"])

def test_evaluate_does_not_alter_model_name():
    result = evaluate("TestModel", np.array([1, 2]), np.array([1, 2]))
    assert result["model"] == "TestModel"

def test_evaluate_baseline_r2_is_zero():
    y_true = np.array([10, 20, 30, 40])
    result = evaluate_baseline(y_true)
    assert result["model"] == "Baseline (mean)"
    assert result["r2"] == 0.0

def test_evals_to_dataframe_sroted_by_rmse_ascending():
    evals = [
        {"model": "Worse", "rmse": 50, "mae": 40, "r2": 0.5, "mape": 20},
        {"model": "Better", "rmse": 10, "mae": 8, "r2": 0.9, "mape": 5},
    ]
    df = evaluations_to_dataframe(evals)
    assert list(df["rmse"]) == sorted(df["rmse"])