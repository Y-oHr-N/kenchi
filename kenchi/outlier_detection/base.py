from abc import abstractmethod, ABC

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import dump
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..plotting import plot_anomaly_score, plot_roc_curve
from ..utils import check_contamination, check_novelty

__all__   = ['BaseOutlierDetector']

NEG_LABEL = -1
POS_LABEL = 1


class BaseOutlierDetector(BaseEstimator, ABC):
    """Base class for all outlier detectors in kenchi.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert, E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.
    """

    _estimator_type = 'outlier_detector'

    def _check_params(self):
        """Raise ValueError if parameters are not valid."""

        if hasattr(self, 'contamination'):
            check_contamination(self.contamination)

    def _check_array(self, X, **kwargs):
        """Raise ValueError if the array is not valid."""

        X             = check_array(X, **kwargs)
        _, n_features = X.shape
        n_features_   = getattr(self, 'n_features_', n_features)

        if n_features != n_features_:
            raise ValueError(
                f'X is expected to have {n_features_} features '
                f'but had {n_features} features'
            )

        return X

    def _check_is_fitted(self):
        """Raise NotFittedError if the estimator is not fitted."""

        check_is_fitted(
            self, [
                'anomaly_score_', 'classes_', 'contamination_', 'n_features_',
                'random_variable_', 'threshold_'
            ]
        )

    def _get_contamination(self):
        """Get the contamination according to the derived anomaly scores."""

        if hasattr(self, 'contamination'):
            return self.contamination

        is_outlier = self.anomaly_score_ > self.threshold_
        n_samples, = is_outlier.shape
        n_outliers = np.sum(is_outlier)

        return n_outliers / n_samples

    def _get_threshold(self):
        """Get the threshold according to the derived anomaly scores."""

        return np.percentile(
            self.anomaly_score_,
            100. * (1. - self.contamination),
            interpolation = 'lower'
        )

    def _get_random_variable(self):
        """Get the RV object according to the derived anomaly scores."""

        loc, scale = norm.fit(self.anomaly_score_)

        return norm(loc=loc, scale=scale)

    @abstractmethod
    def _fit(self, X):
        pass

    @abstractmethod
    def _anomaly_score(self, X):
        pass

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

        X                     = self._check_array(X, estimator=self)

        self._fit(X)

        self.classes_         = np.array([NEG_LABEL, POS_LABEL])
        _, self.n_features_   = X.shape
        self.anomaly_score_   = self._anomaly_score(X)
        self.threshold_       = self._get_threshold()
        self.contamination_   = self._get_contamination()
        self.random_variable_ = self._get_random_variable()

        return self

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

        if hasattr(self, 'novelty'):
            check_novelty(self.novelty, 'fit_predict')

        return self.fit(X).predict()

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
            self.decision_function(X, threshold=threshold) >= 0.,
            POS_LABEL,
            NEG_LABEL
        )

    def predict_proba(self, X=None):
        """Predict class probabilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, predict if a particular training sample is an
            outlier or not.

        Returns
        -------
        y_score : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """

        anomaly_score = self.anomaly_score(X, normalize=True)

        return np.concatenate([
            anomaly_score[:, np.newaxis], 1. - anomaly_score[:, np.newaxis]
        ], axis=1)

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
        shiftted_score_samples : array-like of shape (n_samples,)
            Shifted opposite of the anomaly score for each sample. Negative
            scores represent outliers and positive scores represent inliers.
        """

        score_samples = self.score_samples(X)

        if threshold is None:
            threshold = self.threshold_

        return score_samples + threshold

    def score_samples(self, X=None):
        """Compute the opposite of the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, compute the opposite of the anomaly score for each
            training sample.

        Returns
        -------
        score_samples : array-like of shape (n_samples,)
            Opposite of the anomaly score for each sample.
        """

        return -self.anomaly_score(X)

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

        self._check_is_fitted()

        if X is None:
            anomaly_score = self.anomaly_score_

            if normalize:
                return np.maximum(
                    0., 2. * self.random_variable_.cdf(anomaly_score) - 1.
                )
            else:
                return anomaly_score

        if hasattr(self, 'novelty'):
            check_novelty(self.novelty, 'anomaly_score')

        X                 = self._check_array(X, estimator=self)
        anomaly_score     = self._anomaly_score(X)

        if normalize:
            return np.maximum(
                0., 2. * self.random_variable_.cdf(anomaly_score) - 1.
            )
        else:
            return anomaly_score

    def to_pickle(self, filename, **kwargs):
        """Persist an outlier detector object.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path of the file in which it is to be stored.

        kwargs : dict
            Other keywords passed to ``sklearn.externals.joblib.dump``.

        Returns
        -------
        filenames : list
            List of file names in which the data is stored.
        """

        return dump(self, filename, **kwargs)

    def plot_anomaly_score(self, X=None, normalize=False, **kwargs):
        """Plot the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, plot the anomaly score for each training samples.

        normalize : bool, default False
            If True, plot the normalized anomaly score.

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
            Tuple passed to ``ax.xlim``.

        ylabel : string, default 'Anomaly score'
            Y axis title label. To disable, pass None.

        ylim : tuple, default None
            Tuple passed to ``ax.ylim``.

        **kwargs : dict
            Other keywords passed to ``ax.plot``.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        kwargs['anomaly_score'] = self.anomaly_score(X, normalize=normalize)

        kwargs.setdefault('label', self.__class__.__name__)

        if normalize:
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
            Other keywords passed to ``ax.plot``.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        kwargs['y_true']  = y
        kwargs['y_score'] = self.score_samples(X)

        kwargs.setdefault('label', self.__class__.__name__)

        return plot_roc_curve(**kwargs)
