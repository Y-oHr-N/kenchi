import numpy as np
from sklearn.decomposition import PCA as SKLearnPCA
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, OneDimArray, TwoDimArray

__all__ = ['PCA']


class PCA(BaseDetector):
    """Outlier detector using Principal Component Analysis (PCA).

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    pca_params : dict, default None
        Other keywords passed to sklearn.decomposition.PCA().

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
    def components_(self) -> TwoDimArray:
        return self._pca.components_

    @property
    def explained_variance_(self) -> OneDimArray:
        return self._pca.explained_variance_

    @property
    def explained_variance_ratio_(self) -> OneDimArray:
        return self._pca.explained_variance_ratio_

    @property
    def mean_(self) -> OneDimArray:
        return self._pca.mean_

    @property
    def noise_variance_(self) -> float:
        return self._pca.noise_variance_

    @property
    def n_components_(self) -> int:
        return self._pca.n_components_

    @property
    def singular_values_(self) -> OneDimArray:
        return self._pca.singular_values_

    def __init__(
        self,
        contamination: float = 0.01,
        verbose:       bool  = False,
        pca_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.pca_params = pca_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'PCA':
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

        self.X_         = check_array(X)

        if self.pca_params is None:
            pca_params  = {}
        else:
            pca_params  = self.pca_params

        self._pca       = SKLearnPCA(**pca_params).fit(X)

        self.threshold_ = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

    def reconstruct(self, X: TwoDimArray) -> OneDimArray:
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

        return np.sqrt(np.sum(self.featurewise_anomaly_score(X), axis=1))

    def featurewise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        """Compute the feature-wise anomaly scores for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If not provided, the feature-wise anomaly scores for each
            training sample art returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for each sample.
        """

        check_is_fitted(self, '_pca')

        if X is None:
            X = self.X_
        else:
            X = check_array(X)

        return (X - self.reconstruct(X)) ** 2

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

        check_is_fitted(self, '_pca')

        return self._pca.score(X)
