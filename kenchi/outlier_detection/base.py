import time
from abc import abstractmethod, ABC
from functools import wraps
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

OneDimArray = Union[np.ndarray, pd.Series]
TwoDimArray = Union[np.ndarray, pd.DataFrame]
RandomState = Union[int, np.random.RandomState]


def timeit(func):
    """Return the wrapper function that measures the elapsed time.

    Parameters
    ----------
    func : callable
        Wrapped function.

    Returns
    -------
    wrapper : callable
        Wrapper function.
    """

    @wraps(func)
    def wrapper(estimator, *args, **kwargs):
        def short_format_time(t: float) -> str:
            if t > 60.:
                return f'{t / 60.:5.1f} min'
            else:
                return f'{t:5.1f} sec'

        start   = time.time()
        result  = func(estimator, *args, **kwargs)
        elapsed = time.time() - start

        if estimator.verbose:
            print(f'elaplsed: {short_format_time(elapsed)}')

        return result

    return wrapper


class BaseDetector(BaseEstimator, ABC):
    """Base class for all outlier detectors."""

    # TODO: Implement plot_anomaly_score method
    # TODO: Implement plot_roc_curve method

    @abstractmethod
    def __init__(self, **params) -> None:
        """Initialize parameters."""

    @abstractmethod
    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

    @abstractmethod
    def fit(
        self, X: TwoDimArray, y: OneDimArray = None, **fit_params
    ) -> 'BaseDetector':
        """Fit the model according to the given training data."""

    @abstractmethod
    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
        """Compute the anomaly score for each sample."""

    @abstractmethod
    def score(self, X: TwoDimArray, y: OneDimArray = None) -> float:
        """Compute the mean log-likelihood of the given data."""

    def predict(
        self,
        X:         TwoDimArray = None,
        threshold: float       = None
    ) -> OneDimArray:
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 1 for inliers and -1 for outliers.
        """

        check_is_fitted(self, 'threshold_')

        if threshold is None:
            threshold = self.threshold_

        return np.where(self.anomaly_score(X) <= threshold, 1, -1)

    def fit_predict(
        self,
        X:         TwoDimArray,
        y:         OneDimArray = None,
        threshold: float       = None,
        **fit_params
    ) -> OneDimArray:
        """Fit the model according to the given training data and predict if a
        particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : ignored

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 1 for inliers and -1 for outliers.
        """

        return self.fit(X, **fit_params).predict(threshold=threshold)
