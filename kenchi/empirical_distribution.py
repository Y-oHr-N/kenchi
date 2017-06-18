import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector, DetectorMixin


class EmpiricalDetector(BaseDetector, DetectorMixin):
    """Detector using the k-nearest neighbors algorithm.

    Parameters
    ----------
    fpr : float
        False positive rate. Used to compute the threshold.

    n_jobs : integer
        Number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. Doesn't affect fit method.

    n_neighbors : integer
        Number of neighbors.

    p : integer
        Power parameter for the Minkowski metric.

    threshold : float
        Threshold. If None, it is computed automatically.
    """

    def __init__(self, fpr=0.01, n_jobs=1, n_neighbors=5, p=2, threshold=None):
        self.fpr         = fpr
        self.n_jobs      = n_jobs
        self.n_neighbors = n_neighbors
        self.p           = p
        self.threshold   = threshold

    def fit(self, X, y=None):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : object
            Returns self.
        """

        X                   = check_array(X)

        self._neigh         = NearestNeighbors(
            metric          = 'minkowski',
            n_jobs          = self.n_jobs,
            n_neighbors     = self.n_neighbors,
            p               = self.p
        ).fit(X)

        if self.threshold is None:
            scores          = self.compute_anomaly_score(X)
            self._threshold = np.percentile(scores, 100.0 * (1.0 - self.fpr))

        else:
            self._threshold = self.threshold

        return self

    def compute_anomaly_score(self, X):
        """Computes the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples)
            The anomaly score for test samples.
        """

        n_samples, n_features = X.shape

        dist, ind             = self._neigh.kneighbors(X)
        radius                = np.max(dist, axis=1)

        return -np.log(self.n_neighbors) + n_features * np.log(radius)
