import numpy as np
from sklearn.cluster import MiniBatchKMeans as SKLearnMiniBatchKMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, OneDimArray, TwoDimArray

__all__ = ['MiniBatchKMeans']


class MiniBatchKMeans(BaseDetector):
    """Outlier detector using K-means clustering.

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    kmeans_params : dict, default None
        Other keywords passed to sklearn.cluster.MiniBatchKMeans().

    Attributes
    ----------
    cluster_centers_ : array-like of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    inertia_ : float
        Value of the inertia criterion associated with the chosen partition.

    labels_ : array-like of shape (n_samples,)
        Label of each point.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
    """

    @property
    def cluster_centers_(self) -> OneDimArray:
        return self._kmeans.cluster_centers_

    @property
    def inertia_(self) -> float:
        return self._kmeans.inertia_

    @property
    def labels_(self) -> OneDimArray:
        return self._kmeans.labels_

    def __init__(
        self,
        contamination: float = 0.01,
        verbose:       bool  = False,
        kmeans_params: dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.kmeans_params = kmeans_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'MiniBatchKMeans':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : MiniBatchKMeans
            Return self.
        """

        self.check_params(X)

        self.X_           = check_array(X)

        if self.kmeans_params is None:
            kmeans_params = {}
        else:
            kmeans_params = self.kmeans_params

        self._kmeans      = SKLearnMiniBatchKMeans(**kmeans_params).fit(X)

        self.threshold_   = np.percentile(
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

        check_is_fitted(self, '_kmeans')

        if X is None:
            X = self.X_

        return np.min(self._kmeans.transform(X), axis=1)

    def score(self, X: TwoDimArray, y: OneDimArray = None) -> float:
        """Compute the opposite value of the given data on the K-means
        objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Opposite value of the given data on the K-means objective.
        """

        check_is_fitted(self, '_kmeans')

        return self._kmeans.score(X)
