from abc import abstractmethod, ABC

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .utils import construct_pandas_obj, plot_anomaly_score


class DetectorMixin(ABC):
    """Mixin class for all detectors."""

    # TODO: Implement score method
    # TODO: Implement plot_roc_curve method

    plot_anomaly_score = plot_anomaly_score

    @abstractmethod
    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data."""

    @abstractmethod
    def anomaly_score(self, X):
        """Compute anomaly scores for test samples."""

    @construct_pandas_obj
    def detect(self, X, threshold=None):
        """Detect if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        check_is_fitted(self, 'threshold_')

        if threshold is None:
            threshold = self.threshold_

        return (self.anomaly_score(X) > threshold).astype(np.int32)

    def fit_detect(self, X, y=None, **fit_params):
        """Fit the model according to the given training data and detect if a
        particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        if hasattr(self, '_fit_X'):
            return self.fit(X, **fit_params).detect(None)
        else:
            return self.fit(X, **fit_params).detect(X)


class AnalyzerMixin(ABC):
    """Mixin class for all analyzers."""

    @abstractmethod
    def feature_wise_anomaly_score(self, X):
        """Compute feature-wise anomaly scores for test samples."""

    @construct_pandas_obj
    def analyze(self, X, feature_wise_threshold=None):
        """Analyze which features contribute to anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        feature_wise_threshold : ndarray of shape (n_features,), default None
            User-provided feature-wise threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_features)
        """

        check_is_fitted(self, 'feature_wise_threshold_')

        if feature_wise_threshold is None:
            feature_wise_threshold = self.feature_wise_threshold_

        return (
            self.feature_wise_anomaly_score(X) > feature_wise_threshold
        ).astype(np.int32)

    def fit_analyze(self, X, y=None, **fit_params):
        """Fit the model according to the given training data and analyze which
        features contribute to anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_features)
        """

        if hasattr(self, '_fit_X'):
            return self.fit(X, **fit_params).analyze(None)
        else:
            return self.fit(X, **fit_params).analyze(X)
