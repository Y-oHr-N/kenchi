import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array

from .base import DetectorMixin


class EmpiricalOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector using the k-nearest neighbors algorithm.

    Parameters
    ----------
    fpr : float
        False positive rate. Used to compute the threshold.

    n_jobs : integer
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores. Doesn't affect fit method.

    n_neighbors : integer
        Number of neighbors.

    p : integer
        Power parameter for the Minkowski metric.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(self, fpr=0.01, n_jobs=1, n_neighbors=5, p=2):
        self.fpr         = fpr
        self.n_jobs      = n_jobs
        self.n_neighbors = n_neighbors
        self.p           = p

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : object
            Return self.
        """

        X               = check_array(X)

        self._neigh     = NearestNeighbors(
            metric      = 'minkowski',
            n_jobs      = self.n_jobs,
            n_neighbors = self.n_neighbors,
            p           = self.p
        ).fit(X)

        scores          = self.decision_function(X)
        self.threshold_ = np.percentile(scores, 100.0 * (1.0 - self.fpr))

        return self

    def decision_function(self, X):
        """Compute the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples)
            The anomaly score for test samples.
        """

        _, n_features = X.shape

        dist, _       = self._neigh.kneighbors(X)
        radius        = np.max(dist, axis=1)

        return -np.log(self.n_neighbors) + n_features * np.log(radius)
