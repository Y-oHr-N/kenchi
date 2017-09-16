from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted

from .utils import construct_pandas_object


class DetectorMixin(metaclass=ABCMeta):
    """Mixin class for all detectors."""

    _estimator_type = 'detector'

    @abstractmethod
    def fit(self, X, y=None, **fit_param):
        """Fit the model according to the given training data."""

        pass

    def fit_predict(self, X, y=None, **fit_param):
        """Fit the model to the training set X and return labels (0
        inlier, 1 outlier) on the training set.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        is_outlier : array-like
            Return 0 for inliers and 1 for outliers.
        """

        return self.fit(X, y, **fit_param).predict(X)

    @construct_pandas_object
    @abstractmethod
    def anomaly_score(self, X):
        """Compute the anomaly score."""

        pass

    @construct_pandas_object
    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        is_outlier : array-like
            Return 0 for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['threshold_'])

        X = check_array(X)

        return (self.anomaly_score(X) > self.threshold_).astype(np.int32)
