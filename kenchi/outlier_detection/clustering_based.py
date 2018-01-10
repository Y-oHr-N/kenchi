import numpy as np
from sklearn.cluster import MiniBatchKMeans as SKLearnMiniBatchKMeans
from sklearn.utils import check_array

from .base import timeit, BaseDetector, OneDimArray, TwoDimArray

__all__ = ['MiniBatchKMeans']


class MiniBatchKMeans(BaseDetector):
    """Outlier detector using K-means clustering.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        All other keyword arguments are passed to
        sklearn.cluster.MiniBatchKMeans().

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
        fpr:     float = 0.01,
        verbose: bool  = False,
        **kwargs
    ) -> None:
        self.fpr     = fpr
        self.verbose = verbose
        self._kmeans = SKLearnMiniBatchKMeans(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

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

        self._kmeans.fit(X)

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

        return self._kmeans.score(X)
