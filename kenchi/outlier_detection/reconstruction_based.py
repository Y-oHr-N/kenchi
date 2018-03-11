import numpy as np
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector

__all__ = ['PCA']


class PCA(BaseOutlierDetector):
    """Outlier detector using Principal Component Analysis (PCA).

    Parameters
    ----------
    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    iterated_power : int, default 'auto'
        Number of iterations for the power method computed by svd_solver ==
        'randomized'.

    n_components : int, float, or string, default None
        Number of components to keep.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    svd_solver : string, default 'auto'
        SVD solver to use. Valid solvers are
        ['auto'|'full'|'arpack'|'randomized'].

    tol : float, default 0.0
        Tolerance to declare convergence for singular values computed by
        svd_solver == 'arpack'.

    verbose : bool, default False
        Enable verbose output.

    whiten : bool, default False
        When True the `components_` vectors are multiplied by the square root
        of n_samples and then divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    fit_time_ : float
        Time spent for fitting in seconds.

    threshold_ : float
        Threshold.

    components_ : array-like of shape (n_components, n_features)
        Principal axes in feature space, representing the directions of maximum
        variance in the data.

    explained_variance_ : array-like of shape (n_components,)
        Amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array-like of shape (n_components,)
        Percentage of variance explained by each of the selected components.

    mean_ : array-like of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.

    noise_variance_ : float
        Estimated noise covariance following the Probabilistic PCA model from
        Tipping and Bishop 1999.

    n_components_ : int
        Estimated number of components.

    singular_values_ : array-like of shape (n_components,)
        Singular values corresponding to each of the selected components.
    """

    @property
    def components_(self):
        return self._pca.components_

    @property
    def explained_variance_(self):
        return self._pca.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self._pca.explained_variance_ratio_

    @property
    def mean_(self):
        return self._pca.mean_

    @property
    def noise_variance_(self):
        return self._pca.noise_variance_

    @property
    def n_components_(self):
        return self._pca.n_components_

    @property
    def singular_values_(self):
        return self._pca.singular_values_

    def __init__(
        self, contamination=0.1, iterated_power='auto', n_components=None,
        random_state=None, svd_solver='auto', tol=0., verbose=False,
        whiten=False
    ):
        super().__init__(contamination=contamination, verbose=verbose)

        self.iterated_power = iterated_power
        self.n_components   = n_components
        self.random_state   = random_state
        self.svd_solver     = svd_solver
        self.tol            = tol
        self.whiten         = whiten

    def _fit(self, X):
        self._pca          = SKLearnPCA(
            iterated_power = self.iterated_power,
            n_components   = self.n_components,
            random_state   = self.random_state,
            svd_solver     = self.svd_solver,
            tol            = self.tol,
            whiten         = self.whiten
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return np.sum((X - self._reconstruct(X)) ** 2, axis=1)

    def _reconstruct(self, X):
        """Apply dimensionality reduction to the given data, and transform the
        data back to its original space.
        """

        return self._pca.inverse_transform(self._pca.transform(X))

    def score(self, X, y=None):
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

        check_is_fitted(self, '_pca')

        return self._pca.score(X)
