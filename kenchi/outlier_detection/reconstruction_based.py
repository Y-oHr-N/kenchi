import numpy as np
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit

__all__ = ['PCA']


class PCA(BaseDetector):
    """Outlier detector using Principal Component Analysis (PCA).

    Parameters
    ----------
    contamination : float, default 0.01
        Proportion of outliers in the data set. Used to define the threshold.

    copy : bool, default False
        If False, data passed to `fit` are overwritten.

    iterated_power : int, default 'auto'
        Number of iterations for the power method computed by svd_solver ==
        'randomized'.

    n_components : int, float, or string, default None
        Number of components to keep.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    svd_solver : string, default 'auto'
        auto :
            the solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized' method
            is enabled. Otherwise the exact full SVD is computed and optionally
            truncated afterwards.
        full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`.
        randomized :
            run randomized SVD by the method of Halko et al.

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

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
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
        self,              contamination=0.01,
        copy=True,         iterated_power='auto',
        n_components=None, random_state=None,
        svd_solver='auto', tol=0.,
        verbose=False,     whiten=False
    ):
        super().__init__(contamination=contamination, verbose=verbose)

        self.copy           = copy
        self.iterated_power = iterated_power
        self.n_components   = n_components
        self.random_state   = random_state
        self.svd_solver     = svd_solver
        self.tol            = tol
        self.whiten         = whiten

    def check_params(self, X, y=None):
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

    @timeit
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : PCA
            Return self.
        """

        self.check_params(X)

        self.X_            = check_array(X, estimator=self)
        self._pca          = SKLearnPCA(
            copy           = self.copy,
            iterated_power = self.iterated_power,
            n_components   = self.n_components,
            random_state   = self.random_state,
            svd_solver     = self.svd_solver,
            tol            = self.tol,
            whiten         = self.whiten
        ).fit(X)
        self.threshold_    = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

    def reconstruct(self, X):
        """Apply dimensionality reduction to the given data, and transform the
        data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        Returns
        -------
        X_rec : array-like of shape (n_samples, n_features)
        """

        check_is_fitted(self, '_pca')

        return self._pca.inverse_transform(self._pca.transform(X))

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

        return np.sqrt(np.sum(self.featurewise_anomaly_score(X), axis=1))

    def featurewise_anomaly_score(self, X=None):
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

        check_is_fitted(self, '_pca')

        if X is None:
            X = self.X_
        else:
            X = check_array(X, estimator=self)

        return (X - self.reconstruct(X)) ** 2

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
