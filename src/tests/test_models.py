"""
test_models.py

Unit test for src/models/models.py.

Each model is tested for the following purposes:
- every model conforms to the BaseModel interface
- fit/predict run without error on synthetic data
- predictions have the right shape and size given the input
- a model that fails to implement a method is identified

"""


import pandas as pd
import numpy as np
import pytest

from sklearn.datasets import make_regression
from src.models.base import BaseModel
from src.models.models import (
    LinearRegressionModel,
    LassoModel,
    ElasticNetModel,
    RandomForestRegressorModel,
    XGBoostModel,
)

# Synthetic regression data generator
@pytest.fixture
def synthetic_data():
    X, y = make_regression(
        n_features=3, 
        n_samples=50,
        n_targets=1,
        noise=0.1,
        random_state=42
    )
    return X, y

# Specify models for which the whole test runs 
@pytest.fixture(params=[
    LinearRegressionModel, LassoModel, ElasticNetModel,
    RandomForestRegressorModel, XGBoostModel,
])
def model(request):
    return request.param()


# ---------------------------------------------------------------------------
# BaseModel Interface Tests
# ---------------------------------------------------------------------------
def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()
    # The logic
    # if BaseModel() raises a TypeError, pytest catches it, so test passes.
    # if BaseModel() raises a different error or none, pytests raises its error and test fails.


def test_incomplete_subclass_cannot_be_instantiated():
    # An incomplete subclass is a model without required methods defined.
    class IncompleteModel(BaseModel):
        def fit(self, X_train, y_train):
            pass
        # predict() and get_params() are missing.

    with pytest.raises(TypeError):
        IncompleteModel()


def test_model_has_required_callable_methods(model):
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
    assert callable(model.fit)
    assert callable(model.predict)
    assert callable(model.get_params)

# ---------------------------------------------------------------------------
# Model Behavioral Tests
# ---------------------------------------------------------------------------

def test_model_runs_on_synthetic_data(model, synthetic_data):
    X, y = synthetic_data
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred is not None, f"Model {model} fails on synthetic data"

def test_predictions_have_correct_shape(model, synthetic_data):
    X, y = synthetic_data
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape, (
        f"Expected {y.shape} for predictions, got {y_pred.shape} instead."
    )

def test_predictions_are_numric_and_finite(model, synthetic_data):
    X, y = synthetic_data
    model.fit(X, y)
    y_pred = model.predict(X)
    assert np.issubdtype(y_pred.dtype, np.number)
    assert np.all(np.isfinite(y_pred))

def test_get_params_returns_dict(model):
    parameters = model.get_params()
    assert isinstance(parameters, dict)

def test_model_beats_mean_baseline(model, synthetic_data):
    X, y = synthetic_data
    model.fit(X, y)
    y_pred = model.predict(X)

    y_mean = np.full_like(y, fill_value=y.mean())

    model_mse = np.mean((y - y_pred)**2)
    mean_mse  = np.mean((y - y_mean)**2)
    
    assert model_mse < mean_mse


