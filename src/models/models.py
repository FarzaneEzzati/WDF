"""
models.py

All model wrappers used in the project. Every wrapper follows the BaseModel
interface defined in base.py (fit / predict / get_params), so the pipeline
can train and evaluate them interchangeably.

Linear family : LinearRegressionModel, LassoModel, ElasticNetModel
    Categorical features (default: Store) are one-hot encoded. Lasso and
    ElasticNet also standardise the remaining features so the penalty
    treats every feature equally regardless of its units.
Tree family   : RandomForestRegressorModel, XGBoostModel
    Trees split on raw values, so they need no encoding or scaling.
"""

from functools import partial

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.models.base import BaseModel

DEFAULT_CATEGORICAL_FEATURES = ("Store",)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _present_columns(X, columns):
    """Return the subset of `columns` that exist in X (none for plain arrays)."""
    available = getattr(X, "columns", [])
    return [c for c in columns if c in available]


def _linear_preprocessor(categorical_features, scale_numeric):
    """
    One-hot encode the categorical features and pass the rest through,
    optionally standardising them. Categorical features missing from the
    input are skipped, so the model also works on plain numeric arrays.
    """
    categorical_features = tuple(categorical_features)
    return ColumnTransformer(
        [("onehot",
          OneHotEncoder(handle_unknown="ignore"),
          partial(_present_columns, columns=categorical_features))],
        remainder=StandardScaler() if scale_numeric else "passthrough",
    )


class _SklearnWrapper(BaseModel):
    """Delegates fit / predict / get_params to the estimator in self.model."""

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return np.asarray(self.model.predict(X_test))

    def get_params(self) -> dict:
        return self.model.get_params()


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------
class LinearRegressionModel(_SklearnWrapper):

    def __init__(self, categorical_features=DEFAULT_CATEGORICAL_FEATURES, **kwargs):
        self.model = make_pipeline(
            _linear_preprocessor(categorical_features, scale_numeric=False),
            LinearRegression(**kwargs),
        )


class LassoModel(_SklearnWrapper):
    """L1-regularised linear regression."""

    def __init__(self, categorical_features=DEFAULT_CATEGORICAL_FEATURES, **kwargs):
        defaults = dict(alpha=1.0, max_iter=10_000, random_state=42)
        defaults.update(kwargs)
        self.model = make_pipeline(
            _linear_preprocessor(categorical_features, scale_numeric=True),
            Lasso(**defaults),
        )


class ElasticNetModel(_SklearnWrapper):
    """
    Linear regression with a mix of L1 and L2 penalties
    (l1_ratio=1 is pure Lasso, l1_ratio=0 is pure Ridge).
    """

    def __init__(self, categorical_features=DEFAULT_CATEGORICAL_FEATURES, **kwargs):
        defaults = dict(alpha=1.0, l1_ratio=0.5, max_iter=10_000, random_state=42)
        defaults.update(kwargs)
        self.model = make_pipeline(
            _linear_preprocessor(categorical_features, scale_numeric=True),
            ElasticNet(**defaults),
        )


# ---------------------------------------------------------------------------
# Tree models
# ---------------------------------------------------------------------------
class RandomForestRegressorModel(_SklearnWrapper):

    def __init__(self, **kwargs):
        defaults = dict(n_estimators=10, max_depth=6, random_state=42)
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**defaults)


class XGBoostModel(_SklearnWrapper):

    def __init__(self, **kwargs):
        defaults = dict(n_estimators=200, max_depth=6, random_state=42)
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)
