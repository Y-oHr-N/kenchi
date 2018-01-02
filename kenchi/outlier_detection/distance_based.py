import numpy as np
from sklearn import neighbors

from .base import ArrayLike, BaseDetector

__all__ = ['KNN']


class KNN(BaseDetector):
    """Outlier detector using k-nearest neighbors algorithm.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    weight : bool, default False
        If True, anomaly score is the sum of the distances from k nearest
        neighbors.

    kwargs : dict
        All other keyword arguments are passed to neighbors.NearestNeighbors.

    Attributes
    ----------
    anomaly_score_ : ndarray of shape (n_samples,)
        Anoamly score for each training sample.

    threshold_ : float
        Threshold.

    References
    ----------
    S. Ramaswamy, R. Rastogi and K. Shim,
    "Efficient algorithms for mining outliers from large data sets,"
    In Proceedings of SIGMOD'00, pp. 427-438, 2000.

    F. Angiulli and C. Pizzuti,
    "Fast outlier detection in high dimensional spaces,"
    In Proceedings of PKDD'02, pp. 15-27, 2002.
    """

    def __init__(self, fpr: float=0.01, weight: bool=True, **kwargs) -> None:
        self.fpr    = fpr
        self.weight = weight
        self._knn   = neighbors.NearestNeighbors(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'KNN':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : None
            Ignored.

        Returns
        -------
        self : KNN
            Return self.
        """

        self._knn.fit(X)

        self.anomaly_score_ = self.anomaly_score()
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
            dist, _ = self._knn.kneighbors()
        else:
            dist, _ = self._knn.kneighbors(X)

        if self.weight:
            return np.sum(dist, axis=1)
        else:
            return np.max(dist, axis=1)

    def score(X: ArrayLike, y: None=None) -> float:
        """Compute the mean log-likelihood of the given data."""

        raise NotImplementedError()
