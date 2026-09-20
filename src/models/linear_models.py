
from src.models.base import BaseModel
import numpy as np


class LinearRegressionModel(BaseModel):

    def __init__(self, **kwargs):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return np.asarray(self.model.predict(X_test))

    def get_params(self) -> dict:
        return self.model.get_params()





