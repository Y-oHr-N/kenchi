import numpy as np
import scipy as sp
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphLasso
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import timeit, BaseDetector, OneDimArray, TwoDimArray
from ..utils import plot_partial_corrcoef

__all__ = ['GMM', 'KDE', 'SparseStructureLearning']


class GMM(BaseDetector):
    """Outlier detector using Gaussian Mixture Models (GMMs).

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        All other keyword arguments are passed to
        sklearn.mixture.GaussianMixture().

    Attributes
    ----------
    converged_ : bool
        True when convergence was reached in fit(), False otherwise.

    covariances_ : array-like
        Covariance of each mixture component.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    means_ : array-like of shape (n_components, n_features)
        Mean of each mixture component.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    precisions_ : array-like
        Precision matrices for each component in the mixture.

    precisions_cholesky_ : array-like
        Cholesky decomposition of the precision matrices of each mixture
        component.

    threshold_ : float
        Threshold.

    weights_ : array-like of shape (n_components,)
        Weights of each mixture components.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
    """

    @property
    def converged_(self) -> bool:
        return self._gmm.converged_

    @property
    def covariances_(self) -> OneDimArray:
        return self._gmm.covariances_

    @property
    def lower_bound_(self) -> float:
        return self._gmm.lower_bound_

    @property
    def means_(self) -> OneDimArray:
        return self._gmm.means_

    @property
    def n_iter_(self) -> int:
        return self._gmm.n_iter_

    @property
    def precisions_(self) -> OneDimArray:
        return self._gmm.precisions_

    @property
    def precisions_cholesky_(self) -> OneDimArray:
        return self._gmm.precisions_cholesky_

    @property
    def weights_(self) -> OneDimArray:
        return self._gmm.weights_

    def __init__(
        self,
        fpr:     float = 0.01,
        verbose: bool  = False,
        **kwargs
    ) -> None:
        self.fpr     = fpr
        self.verbose = verbose
        self._gmm    = GaussianMixture(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'GMM':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : GMM
            Return self.
        """

        self._gmm.fit(X)

        self.X_         = check_array(X)
        anomaly_score   = self.anomaly_score()
        self.threshold_ = np.percentile(anomaly_score, 100. * (1. - self.fpr))

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
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
            X = self.X_

        return -self._gmm.score_samples(X)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        """Compute the feature-wise anomaly score for each sample."""

        raise NotImplementedError()

    def score(self, X: TwoDimArray, y: OneDimArray = None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored.

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        return self._gmm.score(X)


class KDE(BaseDetector):
    """Outlier detector using Kernel Density Estimation (KDE).

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        All other keyword arguments are passed to
        sklearn.neighbors.KernelDensity().

    Attributes
    ----------
    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
    """

    @property
    def X_(self) -> TwoDimArray:
        return self._kde.tree_.data

    def __init__(
        self,
        fpr:     float = 0.01,
        verbose: bool  = False,
        **kwargs
    ) -> None:
        self.fpr     = fpr
        self.verbose = verbose
        self._kde    = KernelDensity(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'KDE':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : KDE
            Return self.
        """

        self._kde.fit(X)

        anomaly_score   = self.anomaly_score()
        self.threshold_ = np.percentile(anomaly_score, 100. * (1. - self.fpr))

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
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

        check_is_fitted(self, 'X_')

        if X is None:
            X = self.X_

        return -self._kde.score_samples(X)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        """Compute the feature-wise anomaly score for each sample."""

        raise NotImplementedError()

    def score(self, X: TwoDimArray, y: OneDimArray = None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

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

    cluster_params : dict, default None
        Additional keyword arguments for
        sklearn.cluster.affinity_propagation().

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        All other keyword arguments are passed to
        sklearn.covariance.GraphLasso().

    Attributes
    ----------
    covariance_ : array-like of shape (n_features, n_features)
        Estimated covariance matrix.

    labels_ : array-like of shape (n_features,)
        Label of each feature.

    location_ : array-like of shape (n_features,)
        Estimated location.

    n_iter_ : int
        Number of iterations run.

    partial_corrcoef_ : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    precision_ : array-like of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    T. Ide, C. Lozano, N. Abe and Y. Liu,
    "Proximity-based anomaly detection using sparse structure learning,"
    In Proceedings of SDM'09, pp. 97-108, 2009.
    """

    # TODO: Implement plot_graphical_model method

    plot_partial_corrcoef = plot_partial_corrcoef

    @property
    def covariance_(self) -> TwoDimArray:
        return self._glasso.covariance_

    @property
    def labels_(self) -> OneDimArray:
        """Return the label of each feature."""

        # cluster using affinity propagation
        _, labels = affinity_propagation(
            self.partial_corrcoef_, **self.cluster_params
        )

        return labels

    @property
    def location_(self) -> OneDimArray:
        return self._glasso.location_

    @property
    def n_iter_(self) -> int:
        return self._glasso.n_iter_

    @property
    def partial_corrcoef_(self) -> TwoDimArray:
        """Return the partial correlation coefficient matrix."""

        _, n_features    = self.precision_.shape
        diag             = np.diag(self.precision_)[np.newaxis]
        partial_corrcoef = - self.precision_ / np.sqrt(diag.T @ diag)
        partial_corrcoef.flat[::n_features + 1] = 1.

        return partial_corrcoef

    @property
    def precision_(self) -> TwoDimArray:
        return self._glasso.precision_

    def __init__(
        self,
        fpr:            float = 0.01,
        cluster_params: dict  = None,
        verbose:        bool  = False,
        **kwargs
    ) -> None:
        self.fpr            = fpr
        self.cluster_params = {} if cluster_params is None else cluster_params
        self.verbose        = verbose
        self._glasso        = GraphLasso(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    @timeit
    def fit(
        self,
        X: TwoDimArray,
        y: OneDimArray = None
    ) -> 'SparseStructureLearning':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : ignored

        Returns
        -------
        self : SparseStructureLearning
            Return self.
        """

        self._glasso.fit(X)

        self.X_         = check_array(X)
        anomaly_score   = self.anomaly_score()
        df, loc, scale  = sp.stats.chi2.fit(anomaly_score)
        self.threshold_ = sp.stats.chi2.ppf(1.0 - self.fpr, df, loc, scale)

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
        """Compute thre anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        check_is_fitted(self, 'X_')

        if X is None:
            X = self.X_

        return self._glasso.mahalanobis(X)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        """Compute the feature-wise anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly score for each sample.
        """

        check_is_fitted(self, 'X_')

        if X is None:
            X = self.X_

        return 0.5 * np.log(
            2. * np.pi / np.diag(self.precision_)
        ) + 0.5 / np.diag(
            self.precision_
        ) * ((X - self.location_) @ self.precision_) ** 2

    def score(self, X: TwoDimArray, y: OneDimArray = None) -> float:
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        return self._glasso.score(X)
