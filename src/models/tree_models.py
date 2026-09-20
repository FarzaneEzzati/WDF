from src.models.base import BaseModel
import numpy as np


class XGBoostModel(BaseModel):

    def __init__(self, **kwargs):
        from xgboost import XGBRegressor
        defaults = dict(n_estimators=200, max_depth=6, random_state=42)
        defaults.update(kwargs)
        self.model = XGBRegressor(**defaults)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return np.asarray(self.model.predict(X_test))

    def get_params(self) -> dict:
        return self.model.get_params()


class RandomForestRegressorModel(BaseModel):
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        defaults = dict(n_estimators=10, max_depth=6, random_state=42)
        defaults.update(kwargs)
        self.model = RandomForestRegressor(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
       return self.model.predict(X_test)

    def get_params(self) -> dict:
        return self.model.get_params()