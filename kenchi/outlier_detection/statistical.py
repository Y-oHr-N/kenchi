import numpy as np
import scipy as sp
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphLasso
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, Axes, OneDimArray, TwoDimArray
from ..visualization import plot_graphical_model, plot_partial_corrcoef

__all__ = ['GMM', 'KDE', 'SparseStructureLearning']


class GMM(BaseDetector):
    """Outlier detector using Gaussian Mixture Models (GMMs).

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    gmm_params : dict, default None
        Other keywords passed to sklearn.mixture.GaussianMixture().

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
        contamination: float = 0.01,
        verbose:       bool  = False,
        gmm_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.gmm_params = gmm_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

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

        self.check_params(X)

        self.X_         = check_array(X)

        if self.gmm_params is None:
            gmm_params  = {}
        else:
            gmm_params  = self.gmm_params

        self._gmm       = GaussianMixture(**gmm_params).fit(X)

        self.threshold_ = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
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

        check_is_fitted(self, '_gmm')

        if X is None:
            X = self.X_

        return -self._gmm.score_samples(X)

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

        check_is_fitted(self, '_gmm')

        return self._gmm.score(X)


class KDE(BaseDetector):
    """Outlier detector using Kernel Density Estimation (KDE).

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    kde_params : dict, default None
        Other keywords passed to sklearn.neighbors.KernelDensity().

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
        contamination: float = 0.01,
        verbose:       bool  = False,
        kde_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.kde_params = kde_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

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

        self.check_params(X)

        if self.kde_params is None:
            kde_params  = {}
        else:
            kde_params  = self.kde_params

        self._kde       = KernelDensity(**kde_params).fit(X)

        self.threshold_ = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
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

        check_is_fitted(self, '_kde')

        if X is None:
            X = self.X_

        return -self._kde.score_samples(X)

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

        check_is_fitted(self, '_kde')

        return np.mean(self._kde.score_samples(X))


class SparseStructureLearning(BaseDetector):
    """Outlier detector using sparse structure learning.

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    apcluster_params : dict, default None
        Other keywords passed to sklearn.cluster.affinity_propagation().

    glasso_params : dict, default None
        Other keywords passed to sklearn.covariance.GraphLasso().

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

    @property
    def covariance_(self) -> TwoDimArray:
        return self._glasso.covariance_

    @property
    def labels_(self) -> OneDimArray:
        if self.apcluster_params is None:
            apcluster_params = {}
        else:
            apcluster_params = self.apcluster_params

        # cluster using affinity propagation
        _, labels = affinity_propagation(
            self.partial_corrcoef_, **apcluster_params
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
        n_features, _    = self.precision_.shape
        diag             = np.diag(self.precision_)[np.newaxis]
        partial_corrcoef = - self.precision_ / np.sqrt(diag.T @ diag)
        partial_corrcoef.flat[::n_features + 1] = 1.

        return partial_corrcoef

    @property
    def precision_(self) -> TwoDimArray:
        return self._glasso.precision_

    def __init__(
        self,
        contamination:    float = 0.01,
        verbose:          bool  = False,
        apcluster_params: dict  = None,
        glasso_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.apcluster_params = apcluster_params
        self.glasso_params    = glasso_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

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

        self.check_params(X)

        self.X_           = check_array(X)

        if self.glasso_params is None:
            glasso_params = {}
        else:
            glasso_params = self.glasso_params

        self._glasso      = GraphLasso(**glasso_params).fit(X)

        df, loc, scale    = sp.stats.chi2.fit(self.anomaly_score())
        self.threshold_   = sp.stats.chi2.ppf(
            1.0 - self.contamination, df, loc, scale
        )

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
        """Compute thre anomaly score for each sample.

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

        check_is_fitted(self, '_glasso')

        if X is None:
            X = self.X_
        else:
            X = check_array(X)

        return self._glasso.mahalanobis(X)

    def featurewise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        """Compute the feature-wise anomaly scores for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If not provided, the feature-wise anomaly scores for each
            training sample are returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for each sample.
        """

        check_is_fitted(self, '_glasso')

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

        check_is_fitted(self, '_glasso')

        return self._glasso.score(X)

    def plot_graphical_model(self, **kwargs) -> Axes:
        """Plot the Gaussian Graphical Model (GGM).

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        title : string, default 'Graphical model'
            Axes title. To disable, pass None.

        filepath : str, default None
            If not None, save the current figure.

        **kwargs : dict
            Other keywords passed to nx.draw_networkx().

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        kwargs['node_color'] = self.labels_

        return plot_graphical_model(self.partial_corrcoef_, **kwargs)

    def plot_partial_corrcoef(self, **kwargs) -> Axes:
        """Plot the partial correlation coefficient matrix.

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        cmap : matplotlib Colormap, default None
            If None, plt.cm.RdYlBu is used.

        vmin : float, default -1.0
            Used in conjunction with norm to normalize luminance data.

        vmax : float, default 1.0
            Used in conjunction with norm to normalize luminance data.

        cbar : bool, default True.
            Whether to draw a colorbar.

        title : string, default 'Partial correlation'
            Axes title. To disable, pass None.

        filepath : str, default None
            If not None, save the current figure.

        **kwargs : dict
            Other keywords passed to ax.imshow().

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        return plot_partial_corrcoef(self.partial_corrcoef_, **kwargs)
