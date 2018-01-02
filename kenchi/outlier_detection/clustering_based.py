import numpy as np
from sklearn import cluster

from .base import ArrayLike, BaseDetector

__all__ = ['MiniBatchKMeans']


class MiniBatchKMeans(BaseDetector):
    """Outlier detector using k-means clustering.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kwargs : dict
        All other keyword arguments are passed to cluster.MiniBatchKMeans.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point.

    inertia_ : float
        Value of the inertia criterion associated with the chosen partition.

    anomaly_score_ : ndarray of shape (n_samples,)
        Anomaly score for each training sample.

    threshold_ : float
        Threshold.
    """

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self._kmeans.cluster_centers_

    @property
    def labels_(self) -> np.ndarray:
        return self._kmeans.labels_

    @property
    def inertia_(self) -> float:
        return self._kmeans.inertia_

    def __init__(self, fpr: float=0.01, **kwargs) -> None:
        self.fpr     = fpr
        self._kmeans = cluster.MiniBatchKMeans(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'MiniBatchKMeans':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored.

        Returns
        -------
        self : MiniBatchKMeans
            Return self.
        """

        self._kmeans.fit(X)

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

        if X is None:
            return self.anomaly_score_
        else:
            return np.min(self._kmeans.transform(X), axis=1)

    def score(self, X: ArrayLike, y: None=None) -> float:
        """Compute the opposite value of the given data on the K-means
        objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : None
            Ignored.

        Returns
        -------
        score : float
            Opposite value of the given data on the K-means objective.
        """

        return self._kmeans.score(X)
