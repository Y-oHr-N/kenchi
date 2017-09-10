import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted


class DetectorMixin:
    """Mixin class for all detectors."""

    _estimator_type = 'detector'

    def fit_predict(self, X, y=None, **fit_param):
        """Fit the model to the training set X and return the labels (0
        inlier, 1 outlier) on the training set.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        is_outlier : ndarray, shape = (n_samples) or (n_windows)
            Return 0 for inliers and 1 for outliers.
        """

        return self.fit(X, y, **fit_param).predict(X)

    def predict(self, X):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        is_outlier : ndarray, shape = (n_samples) or (n_windows)
            Return 0 for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['threshold_'])

        X = check_array(X)

        if isinstance(self.threshold_, float):
            return (
                self.decision_function(X) > self.threshold_
            ).astype(np.int32)

        else:
            return np.any(
                self.decision_function(X) > self.threshold_, axis=1
            ).astype(np.int32)


def window_generator(X, window=5, shift=1):
    """Generator that yields windows from given data.

    parameters
    ----------
    X : array-like, shpae = (n_samples, n_features)
        Samples.

    window : integer
        Window size.

    shift : integer
        Shift size.

    Returns
    -------
    gen : generator
        Generator.
    """

    if window < shift:
        raise ValueError('window must be greater than or equal to shift.')

    X            = check_array(X)
    n_samples, _ = X.shape

    for i in range((n_samples - window + shift) // shift):
        yield X[i * shift:i * shift + window]
