from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .utils import construct_pandas_object, plot_anomaly_score


class DetectorMixin(metaclass=ABCMeta):
    """Mixin class for all detectors."""

    _estimator_type    = 'detector'

    plot_anomaly_score = plot_anomaly_score

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data."""

        pass

    @abstractmethod
    @construct_pandas_object
    def anomaly_score(self, X, y=None):
        """Compute anomaly scores for test samples."""

        pass

    @construct_pandas_object
    def detect(self, X, y=None):
        """Detect if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        is_outlier : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        check_is_fitted(self, 'threshold_')

        return (self.anomaly_score(X, y) > self.threshold_).astype(np.int32)

    def fit_detect(self, X, y=None, **fit_params):
        """Fit the model according to the given training data and detect if a
        particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        is_outlier : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        return self.fit(X, y, **fit_params).detect(X, y)
