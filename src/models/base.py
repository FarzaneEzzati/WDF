"""
base.py

Defines the common interface every model wrapper in this project must follow.
All model wrappers in models.py implement this interface.

"""

from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the model in place.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
        y_train : array-like, shape (n_samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        """
        Generate predictions for new data.

        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict:
        """Return the hyperparameters used to configure the model."""
        raise NotImplementedError

    def name(self) -> str:
        
        return self.__class__.__name__

