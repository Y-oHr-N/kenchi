from abc import abstractmethod, ABCMeta

import numpy as np
import matplotlib.pyplot as plt
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
        """Fit the model according to the given training data and predict
        labels (0 inlier, 1 outlier) on the training set.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        return self.fit(X, y, **fit_param).predict(X)

    @construct_pandas_object
    @abstractmethod
    def anomaly_score(self, X):
        """Compute anomaly scores."""

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
        y_pred : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        check_is_fitted(self, ['threshold_'])

        return (self.anomaly_score(X) > self.threshold_).astype(np.int32)

    def plot_anomaly_score(
        self,             X,
        ax=None,          title=None,
        xlim=None,        ylim=None,
        xlabel='Samples', ylabel='Anomaly score',
        grid=True,        **kwargs
    ):
        """Plot anomaly scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        ax : matplotlib Axes, default None
            Target axes instance.

        title : string, default None
            Axes title. To disable, pass None.

        xlim : tuple, default None
            Tuple passed to axes.xlim().

        ylim : tuple, default None
            Tuple passed to axes.ylim().

        xlabel : string, default "Samples"
            X axis title label. To disable, pass None.

        ylabel : string, default "Anomaly score"
            Y axis title label. To disable, pass None.

        grid : boolean, default True
            If True, turn the axes grids on.

        **kwargs : dictionary
            Other keywords passed to ax.bar().

        Returns
        -------
        ax : matplotlib Axes
        """

        check_is_fitted(self, ['threshold_'])

        X            = check_array(X)
        n_samples, _ = X.shape

        xlocs        = np.arange(n_samples)
        scores       = self.anomaly_score(X)

        if ax is None:
            _, ax    = plt.subplots(1, 1)

        if xlim is None:
            xlim     = (-1, n_samples)

        if ylim is None:
            ylim     = (0, 1.1 * max(max(scores), self.threshold_))

        if title is not None:
            ax.set_title(title)

        if xlabel is not None:
            ax.set_xlabel(xlabel)

        if ylabel is not None:
            ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(grid)

        ax.bar(
            left     = xlocs,
            height   = scores,
            align    = 'center',
            **kwargs
        )

        ax.hlines(self.threshold_, *xlim)

        return ax
