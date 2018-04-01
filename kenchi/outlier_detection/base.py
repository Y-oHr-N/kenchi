from abc import abstractmethod, ABC

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..visualization import plot_anomaly_score, plot_roc_curve

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
    """Base class for all outlier detectors in kenchi.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM'11, pp. 13-24, 2011.
    """

    _estimator_type = 'outlier_detector'

    @abstractmethod
    def __init__(self, contamination=0.1):
        self.contamination = contamination

    def _check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if not 0. < self.contamination <= 0.5:
            raise ValueError(
                f'contamination must be in (0.0, 0.5] '
                f'but was {self.contamination}'
            )

    def _check_array(self, X, n_features=None, **kwargs):
        """Check validity of the array and raise ValueError if not valid."""

        X              = check_array(X, **kwargs)
        _, _n_features = X.shape

        if n_features is not None and _n_features != n_features:
            raise ValueError(
                f'X is expected to have {n_features} features '
                f'but had {_n_features} features'
            )

        return X

    def _get_threshold(self):
        """Get the threshold according to the derived anomaly scores."""

        return np.percentile(
            self.anomaly_score_, 100. * (1. - self.contamination)
        )

    def _get_rv(self):
        """Get the RV object according to the derived anomaly scores."""

        loc, scale = norm.fit(self.anomaly_score_)

        return norm(loc=loc, scale=scale)

    @abstractmethod
    def _fit(self, X):
        pass

    @abstractmethod
    def _anomaly_score(self, X):
        pass

    def fit_predict(self, X, y=None):
        """Fit the model according to the given training data and predict if a
        particular training sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : ignored

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return -1 for outliers and +1 for inliers.
        """

        if getattr(self, 'novelty', False):
            raise ValueError(
                'fit_predict is not available when novelty=True, use '
                'novelty=False if you want to predict on the training data'
            )

        return self.fit(X).predict()

    def fit(self, X, y=None):
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

        self._check_params()

        X                   = self._check_array(X, estimator=self)
        _, self._n_features = X.shape

        self._fit(X)

        self.anomaly_score_ = self._anomaly_score(X)
        self.threshold_     = self._get_threshold()
        self._rv            = self._get_rv()

        return self

    def predict(self, X=None, threshold=None):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, predict if a particular training sample is an
            outlier or not.

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return -1 for outliers and +1 for inliers.
        """

        return np.where(
            self.decision_function(X, threshold=threshold) >= 0., 1, -1
        )

    def decision_function(self, X=None, threshold=None):
        """Compute the decision function of the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, compute the decision function of the given training
            samples.

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_score : array-like of shape (n_samples,)
            Shifted opposite of the anomaly score for each sample. Negative
            scores represent outliers and positive scores represent inliers.
        """

        anomaly_score = self.anomaly_score(X)

        if threshold is None:
            threshold = self.threshold_

        return threshold - anomaly_score

    def anomaly_score(self, X=None, normalize=False):
        """Compute the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, compute the anomaly score for each training sample.

        normalize : bool, default False
            If True, return the normalized anomaly score.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        check_is_fitted(self, 'anomaly_score_')

        if X is None:
            anomaly_score = self.anomaly_score_

            if normalize:
                return np.maximum(0., 2. * self._rv.cdf(anomaly_score) - 1.)
            else:
                return anomaly_score

        if getattr(self, 'novelty', True):
            X             = self._check_array(
                X, n_features=self._n_features, estimator=self
            )
            anomaly_score = self._anomaly_score(X)

            if normalize:
                return np.maximum(0., 2. * self._rv.cdf(anomaly_score) - 1.)
            else:
                return anomaly_score

        raise ValueError(
            'anomaly_score is not available when novelty=False, use '
            'novelty=True if you want to predict on new unseen data'
        )

    def plot_anomaly_score(self, X=None, normalize=False, **kwargs):
        """Plot the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, plot the anomaly score for each training samples.

        normalize : bool, default False
            If True, return the normalized anomaly score.

        ax : matplotlib Axes, default None
            Target axes instance.

        bins : int, str or array-like, default 'auto'
            Number of hist bins.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        hist : bool, default True
            If True, plot a histogram of anomaly scores.

        kde : bool, default True
            If True, plot a gaussian kernel density estimate.

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

        kwargs['anomaly_score'] = self.anomaly_score(X, normalize=normalize)

        kwargs.setdefault('label', self.__class__.__name__)

        if normalize:
            kwargs['threshold'] = np.maximum(
                0., 2. * self._rv.cdf(self.threshold_) - 1.
            )

            kwargs.setdefault('ylim', (0., 1.05))

        else:
            kwargs['threshold'] = self.threshold_

            kwargs.setdefault('ylim', (0., 2. * self.threshold_))

        return plot_anomaly_score(**kwargs)

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

        title : string, default 'ROC curve'
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

        kwargs['y_true']  = y
        kwargs['y_score'] = self.decision_function(X)

        kwargs.setdefault('label', self.__class__.__name__)

        return plot_roc_curve(**kwargs)
