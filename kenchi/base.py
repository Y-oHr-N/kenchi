from abc import abstractmethod, ABC

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .visualization import plot_anomaly_score, plot_roc_curve

__all__ = ['is_detector', 'BaseDetector']


def is_detector(estimator):
    return getattr(estimator, '_estimator_type', None) == 'detector'


class BaseDetector(BaseEstimator, ABC):
    """Base class for all outlier detectors."""

    _estimator_type = 'detector'

    @abstractmethod
    def __init__(self, contamination=0.01, verbose=False):
        self.contamination = contamination
        self.verbose       = verbose

    @abstractmethod
    def check_params(self, X, y=None):
        """Check validity of parameters and raise ValueError if not valid."""

        if not 0. <= self.contamination <= 0.5:
            raise ValueError(
                f'contamination must be between 0.0 and 0.5 inclusive '
                f'but was {self.contamination}'
            )

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data."""

    @abstractmethod
    def anomaly_score(self, X=None):
        """Compute the anomaly score for each sample."""

    def predict(self, X=None, threshold=None):
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
            Data. If not provided, plot the anomaly score for each training
            samples.

        ax : matplotlib Axes, default None
            Target axes instance.

        figsize: tuple, default None
            Tuple denoting figure size of the plot.

        title : string, default None
            Axes title. To disable, pass None.

        xlim : tuple, default None
            Tuple passed to `ax.xlim`.

        ylim : tuple, default None
            Tuple passed to `ax.ylim`.

        xlabel : string, default 'Samples'
            X axis title label. To disable, pass None.

        ylabel : string, default 'Anomaly score'
            Y axis title label. To disable, pass None.

        grid : boolean, default True
            If True, turn the axes grids on.

        filepath : str, default None
            If not None, save the current figure.

        **kwargs : dict
            Other keywords passed to `ax.plot`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        return plot_anomaly_score(
            self.anomaly_score(X), self.threshold_, **kwargs
        )

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

        label : str, default None
            Legend label.

        title : string, default None
            Axes title. To disable, pass None.

        grid : boolean, default True
            If True, turn the axes grids on.

        filepath : str, default None
            If not None, save the current figure.

        **kwargs : dict
            Other keywords passed to `ax.plot`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        return plot_roc_curve(y, self.anomaly_score(X), **kwargs)
