from typing import Union

import numpy as np
import scipy as sp
from sklearn import covariance, cluster, mixture, neighbors
from sklearn.utils.validation import check_is_fitted

from .base import ArrayLike, BaseDetector

__all__ = ['GMM', 'KDE', 'SparseStructureLearning']


class GMM(BaseDetector):
    """Outlier detector using Gaussian mixture models.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kwargs : dict
        All other keyword arguments are passed to mixture.GaussianMixture.

    Attributes
    ----------
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    covariances_ : ndarray
        Covariance of each mixture component.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    means_ : ndarray of shape (n_components, n_features)
        Mean of each mixture component.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    precisions_ : ndarray
        Precision matrices for each component in the mixture.

    precisions_cholesky_ : ndarray
        Cholesky decomposition of the precision matrices of each mixture
        component.

    weights_ : ndarray of shape (n_components,)
        Weights of each mixture components.

    anomaly_score_ : ndarray of shape (n_samples,)
        Anomaly score for each training sample.

    threshold_ : float
        Threshold.
    """

    @property
    def converged_(self) -> bool:
        return self._gmm.converged_

    @property
    def covariances_(self) -> np.ndarray:
        return self._gmm.covariances_

    @property
    def lower_bound_(self) -> float:
        return self._gmm.lower_bound_

    @property
    def means_(self) -> np.ndarray:
        return self._gmm.means_

    @property
    def n_iter_(self) -> int:
        return self._gmm.n_iter_

    @property
    def precisions_(self) -> np.ndarray:
        return self._gmm.precisions_

    @property
    def precisions_cholesky_(self) -> np.ndarray:
        return self._gmm.precisions_cholesky_

    @property
    def weights_(self) -> np.ndarray:
        return self._gmm.weights_

    def __init__(self, fpr=0.01, **kwargs) -> None:
        self.fpr  = fpr
        self._gmm = mixture.GaussianMixture(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'GMM':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored.

        Returns
        -------
        self : GMM
            Return self.
        """

        self._gmm.fit(X)

        self.anomaly_score_ = self.anomaly_score(X)
        self.threshold_     = np.percentile(
            self.anomaly_score_, 100. * (1. - self.fpr)
        )

        return self

    def anomaly_score(self, X: ArrayLike=None) -> np.ndarray:
        """Compute the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        if X is None:
            return self.anomaly_score_
        else:
            return -self._gmm.score_samples(X)

    def score(self, X: ArrayLike, y: None=None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : None
            Ignored.

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        return self._gmm.score(X)


class KDE(BaseDetector):
    """Outlier detector using kernel density estimation.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kwargs : dict
        All other keyword arguments are passed to neighbors.KernelDensity.

    Attributes
    ----------
    anomaly_score_ : ndarray of shape (n_samples,)
        Anomaly score for each training sample.

    threshold_ : float
        Threshold.
    """

    @property
    def tree_(self) -> Union[neighbors.BallTree, neighbors.KDTree]:
        self._kde.tree_

    def __init__(self, fpr: float=0.01, **kwargs) -> None:
        self.fpr  = fpr
        self._kde = neighbors.KernelDensity(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'KDE':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored.

        Returns
        -------
        self : KDE
            Return self.
        """

        self._kde.fit(X)

        self.anomaly_score_ = self.anomaly_score(X)
        self.threshold_     = np.percentile(
            self.anomaly_score_, 100. * (1. - self.fpr)
        )

        return self

    def anomaly_score(self, X: ArrayLike=None) -> np.ndarray:
        """Compute the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        anomaly_score : ndarray of shape (n_samples,)
            Anomaly score for each sample.
        """

        check_is_fitted(self, 'tree_')

        if X is None:
            return self.anomaly_score_
        else:
            return -self._kde.score_samples(X)

    def score(self, X: ArrayLike, y: None=None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : None
            Ignored.

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        return np.mean(self._kde.score_samples(X))


class SparseStructureLearning(BaseDetector):
    """Outlier detector using sparse structure learning.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kwargs : dict
        All other keyword arguments are passed to covariance.GraphLasso.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    n_iter_ : int
        Number of iterations run.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    partial_corrcoef_ : ndarray of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    labels_ : ndarray of shape (n_features,)
        Label of each feature.

    anomaly_score_ : ndarray of shape (n_samples,)
        Anomaly score for each training sample.

    feature_wise_anomaly_score_ : ndarray of shape (n_samples, n_features)
        Feature-wise anomaly score for each training sample.

    threshold_ : float
        Threshold.

    References
    ----------
    T. Ide, C. Lozano, N. Abe and Y. Liu,
    "Proximity-based anomaly detection using sparse structure learning,"
    In Proceedings of SDM'09, pp. 97-108, 2009.
    """

    # TODO: Implement plot_partial_corrcoef method
    # TODO: Implement plot_graphical_model method

    @property
    def covariance_(self) -> np.ndarray:
        return self._glasso.covariance_

    @property
    def location_(self) -> np.ndarray:
        return self._glasso.location_

    @property
    def n_iter_(self) -> int:
        return self._glasso.n_iter_

    @property
    def precision_(self) -> np.ndarray:
        return self._glasso.precision_

    @property
    def partial_corrcoef_(self) -> np.ndarray:
        """Return the partial correlation coefficient matrix."""

        _, n_features    = self.precision_.shape
        diag             = np.diag(self.precision_)[np.newaxis]
        partial_corrcoef = - self.precision_ / np.sqrt(diag.T @ diag)
        partial_corrcoef.flat[::n_features + 1] = 1.

        return partial_corrcoef

    @property
    def labels_(self) -> np.ndarray:
        """Return the label of each feature."""

        # cluster using affinity propagation
        _, labels = cluster.affinity_propagation(self.partial_corrcoef_)

        return labels

    def __init__(self, fpr: float=0.01, **kwargs) -> None:
        self.fpr     = fpr
        self._glasso = covariance.GraphLasso(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'SparseStructureLearning':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : None
            Ignored.

        Returns
        -------
        self : SparseStructureLearning
            Return self.
        """

        self._glasso.fit(X)

        self.anomaly_score_ = self.anomaly_score(X)
        df, loc, scale      = sp.stats.chi2.fit(self.anomaly_score_)
        self.threshold_     = sp.stats.chi2.ppf(1.0 - self.fpr, df, loc, scale)

        self.feature_wise_anomaly_score_ = self.feature_wise_anomaly_score(X)

        return self

    def anomaly_score(self, X: ArrayLike=None) -> np.ndarray:
        """Compute thre anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        anomaly_score : ndarray of shape (n_samples,)
            Anomaly score for each sample.
        """

        check_is_fitted(self, ['covariance_', 'n_iter_', 'precision_'])

        if X is None:
            return self.anomaly_score_
        else:
            return self._glasso.mahalanobis(X)

    def feature_wise_anomaly_score(self, X: ArrayLike=None) -> np.ndarray:
        """Compute the feature-wise anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        feature_wise_anomaly_score : ndarray of shape (n_samples, n_features)
            Feature-wise anomaly score for each sample.
        """

        check_is_fitted(self, ['covariance_', 'n_iter_', 'precision_'])

        if X is None:
            return self.feature_wise_anomaly_score_
        else:
            return .5 * np.log(
                2. * np.pi / np.diag(self.precision_)
            ) + .5 / np.diag(
                self.precision_
            ) * ((X - self.location_) @ self.precision_) ** 2

    def score(self, X: ArrayLike, y: None=None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : None
            Ignored.

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        return self._glasso.score(X)
