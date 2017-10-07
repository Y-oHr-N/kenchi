from abc import abstractmethod, ABC

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .utils import construct_pandas_obj, plot_anomaly_score


class DetectorMixin(ABC):
    """Mixin class for all detectors."""

    plot_anomaly_score = plot_anomaly_score

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data."""

    @abstractmethod
    def anomaly_score(self, X, y=None):
        """Compute anomaly scores for test samples."""

    @construct_pandas_obj
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


class AnalyzerMixin(ABC):
    """Mixin class for all analyzers."""

    @abstractmethod
    def feature_wise_anomaly_score(self, X, y=None):
        """Compute feature-wise anomaly scores for test samples."""

    @construct_pandas_obj
    def analyze(self, X, y=None):
        """Analyze which features contribute to anomalies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        is_outlier : array-like, shape = (n_samples, n_features)
        """

        check_is_fitted(self, 'feature_wise_threshold_')

        return (
            self.feature_wise_anomaly_score(X, y) \
            > self.feature_wise_threshold_
        ).astype(np.int32)

    def fit_analyze(self, X, y=None, **fit_params):
        """Fit the model according to the given training data and analyze which
        features contribute to anomalies.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        is_outlier : array-like, shape = (n_samples, n_features)
        """

        return self.fit(X, y, **fit_params).analyze(X, y)
