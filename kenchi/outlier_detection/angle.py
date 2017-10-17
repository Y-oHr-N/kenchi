from itertools import combinations

import numpy as np
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


def _minkowski(x, y, order):
    dist = minkowski(x, y, order)

    return dist if dist != 0 else np.inf


class FastABOD(NearestNeighbors, DetectorMixin):
    """Fast angle-based outlier detector.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores. Doesn't affect fit method.

    n_neighbors : int, default 5
        Number of neighbors.

    p : int, default 2
        Power parameter for the Minkowski metric.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(self, fpr=0.01, n_jobs=1, n_neighbors=5, p=2):
        super().__init__(
            metric        = _minkowski,
            metric_params = {'order': p},
            n_jobs        = n_jobs,
            n_neighbors   = n_neighbors
        )

        self.fpr          = fpr

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0 or 1 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.n_neighbors <= 1:
            raise ValueError(
                'n_neighbors must be greator than 1 but was {0}'.format(
                    self.n_neighbors
                )
            )

        if self.p < 1:
            raise ValueError(
                'p must be greater than or equal to 1 but was {0}'.format(
                    self.p
                )
            )

    @assign_info_on_pandas_obj
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : detector
            Return self.
        """

        X               = check_array(X)

        super().fit(X)

        scores          = self.anomaly_score(X)
        self.threshold_ = np.percentile(scores, 100.0 * (1.0 - self.fpr))

        return self

    @construct_pandas_obj
    def abof(self, X, y=None):
        """Compute angle-based outlier factors for test samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        foctors : array-like, shape = (n_samples,)
            Angle-based outlier factors for test samples.
        """

        check_is_fitted(self, '_fit_method')

        X   = check_array(X)
        ind = self.kneighbors(X, return_distance=False)

        return np.var([
            [
                (ab @ ac) / (ab @ ab) / (ac @ ac) for ab, ac in combinations(
                    self._fit_X[ind_a] - a, 2
                )
            ] for a, ind_a in zip(X, ind)
        ], axis=1)

    @construct_pandas_obj
    def anomaly_score(self, X, y=None):
        """Compute anomaly scores for test samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : array-like, shape = (n_samples,)
            anomaly scores for test samples.
        """

        return -self.abof(X)
