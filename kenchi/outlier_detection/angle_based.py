from itertools import combinations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseOutlierDetector

__all__ = ['FastABOD']


class FastABOD(BaseOutlierDetector):
    """Fast Angle-Based Outlier Detector (FastABOD).

    Parameters
    ----------
    algorithm : str, default 'auto'
        Tree algorithm to use. Valid algorithms are
        ['kd_tree'|'ball_tree'|'auto'].

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    leaf_size : int, default 30
        Leaf size of the underlying tree.

    metric : str or callable, default 'minkowski'
        Distance metric to use.

    novelty : bool, default False
        If True, you can use predict, decision_function and anomaly_score on
        new unseen data and not on the training data.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    n_neighbors : int, default 20
        Number of neighbors.

    p : int, default 2
        Power parameter for the Minkowski metric.

    metric_params : dict, default None
        Additioal parameters passed to the requested metric.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    n_neighbors_ : int
        Actual number of neighbors used for `kneighbors` queries.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM'11, pp. 13-24, 2011.

    .. [#kriegel08] Kriegel, H.-P., Schubert M., and Zimek, A.,
        "Angle-based outlier detection in high-dimensional data,"
        In Proceedings of SIGKDD'08, pp. 444-452, 2008.
    """

    @property
    def X_(self):
        return self._estimator._fit_X

    def __init__(
        self, algorithm='auto', contamination=0.1, leaf_size=30,
        metric='minkowski', novelty=False, n_jobs=1, n_neighbors=20,
        p=2, metric_params=None
    ):
        super().__init__(contamination=contamination)

        self.algorithm     = algorithm
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.novelty       = novelty
        self.n_jobs        = n_jobs
        self.n_neighbors   = n_neighbors
        self.p             = p
        self.metric_params = metric_params

    def _fit(self, X):
        n_samples, _            = X.shape
        self.n_neighbors_       = np.maximum(
            1, np.minimum(self.n_neighbors, n_samples - 1)
        )
        self._estimator         = NearestNeighbors(
            algorithm           = self.algorithm,
            leaf_size           = self.leaf_size,
            metric              = self.metric,
            n_jobs              = self.n_jobs,
            n_neighbors         = self.n_neighbors_,
            p                   = self.p,
            metric_params       = self.metric_params
        ).fit(X)
        self._anomaly_score_min = np.max(
            self._anomaly_score(X, regularize=False)
        )

        return self

    def _anomaly_score(self, X, regularize=True):
        abof = self._abof(X)

        if regularize:
            return -np.log(abof / self._anomaly_score_min)
        else:
            return abof

    def _abof(self, X):
        """Compute the Angle-Based Outlier Factor (ABOF) for each sample."""

        if X is self.X_:
            neigh_ind = self._estimator.kneighbors(return_distance=False)
        else:
            neigh_ind = self._estimator.kneighbors(X, return_distance=False)

        return np.var([
            [
                (pa @ pb) / (pa @ pa) / (pb @ pb) for pa, pb in combinations(
                    X_neigh - query_point, 2
                )
            ] for query_point, X_neigh in zip(X, self.X_[neigh_ind])
        ], axis=1)
