from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y

from .visualization import plot_anomaly_score, plot_roc_curve

__all__ = ['is_outlier_detector', 'BaseOutlierDetector']


def is_outlier_detector(estimator):
    """Return True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    """

    return getattr(estimator, '_estimator_type', None) == 'outlier_detector'


class BaseOutlierDetector(BaseEstimator, ABC):
    """Base class for all outlier detectors in kenchi."""

    # TODO: Update anomaly_score method so that the normalized score can be
    # computed

    _estimator_type = 'outlier_detector'

    @abstractmethod
    def __init__(self, contamination=0.01, verbose=False):
        self.contamination = contamination
        self.verbose       = verbose

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : object
            Return self.
        """

    @abstractmethod
    def anomaly_score(self, X=None):
        """Compute the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If not provided, the anomaly score for each training sample
            is returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if not 0. <= self.contamination <= 0.5:
            raise ValueError(
                f'contamination must be between 0.0 and 0.5 inclusive '
                f'but was {self.contamination}'
            )

    def _get_threshold(self):
        """Define the threshold according to the given training data."""

        return np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

    def predict(self, X=None, threshold=None):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, Labels on the given training data are returned.

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

    def fit_predict(self, X, y=None, threshold=None, **fit_params):
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

    def plot_anomaly_score(self, X=None, **kwargs):
        """Plot the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, plot the anomaly score for each training samples.

        ax : matplotlib Axes, default None
            Target axes instance.

        bins : int, str or array-like, default 'auto'
            Number of hist bins.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        grid : boolean, default True
            If True, turn the axes grids on.

        hist : bool, default True
            If True, plot a histogram of anomaly scores.

        title : string, default None
            Axes title. To disable, pass None.

        xlabel : string, default 'Samples'
            X axis title label. To disable, pass None.

        xlim : tuple, default None
            Tuple passed to `ax.xlim`.

        ylabel : string, default 'Anomaly score'
            Y axis title label. To disable, pass None.

        ylim : tuple, default None
            Tuple passed to `ax.ylim`.

        **kwargs : dict
            Other keywords passed to `ax.plot`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        kwargs['threshold'] = self.threshold_

        return plot_anomaly_score(self.anomaly_score(X), **kwargs)

    def plot_roc_curve(self, X, y, **kwargs):
        """Plot the Receiver Operating Characteristic (ROC) curve.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Labels.

        ax : matplotlib Axes, default None
            Target axes instance.

        figsize: tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        grid : boolean, default True
            If True, turn the axes grids on.

        label : str, default None
            Legend label.

        title : string, default None
            Axes title. To disable, pass None.

        xlabel : string, default 'FPR'
            X axis title label. To disable, pass None.

        ylabel : string, default 'TPR'
            Y axis title label. To disable, pass None.

        **kwargs : dict
            Other keywords passed to `ax.plot`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        X, y = check_X_y(X, y)

        return plot_roc_curve(y, self.anomaly_score(X), **kwargs)
