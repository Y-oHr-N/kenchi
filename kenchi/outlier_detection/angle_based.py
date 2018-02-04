import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals.joblib import delayed, Parallel
from sklearn.utils import check_array, gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, OneDimArray, TwoDimArray

__all__ = ['FastABOD']


def _approximate_abof(
    X:         TwoDimArray,
    X_train:   TwoDimArray,
    neigh_ind: TwoDimArray
) -> OneDimArray:
    """Compute the approximate Angle-Based Outlier Factor (ABOF) for each
    sample.
    """

    with np.errstate(invalid='raise'):
        return np.var([
            [
                (diff_a @ diff_b) / (diff_a @ diff_a) / (diff_b @ diff_b)
                for diff_a, diff_b in itertools.combinations(
                    X_neigh - query_point, 2
                )
            ] for query_point, X_neigh in zip(X, X_train[neigh_ind])
        ], axis=1)


class FastABOD(BaseDetector):
    """Fast Angle-Based Outlier Detector (FastABOD).

    Parameters
    ----------
    algorithm : str, default 'auto'
        Tree algorithm to use. Valid algorithms are
        ['kd_tree'|'ball_tree'|'auto'].

    contamination : float, default 0.01
        Proportion of outliers in the data set. Used to define the threshold.

    leaf_size : int, default 30
        Leaf size of the underlying tree.

    metric : str or callable, default 'minkowski'
        Distance metric to use.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    n_neighbors : int, default 5
        Number of neighbors.

    p : int, default 2
        Power parameter for the Minkowski metric.

    verbose : bool, default False
        Enable verbose output.

    metric_params : dict, default None
        Additioal parameters passed to the requested metric.

    Attributes
    ----------
    abof_max_ : float
        Maximum possible ABOF.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    H.-P. Kriegel, M. Schubert and A. Zimek,
    "Angle-based outlier detection in high-dimensional data,"
    In Proceedings of SIGKDD'08, pp. 444-452, 2008.

    H.-P. Kriegel, P. Kroger, E. Schubert and A. Zimek,
    "Interpreting and unifying outlier scores,"
    In Proceedings of SDM'11, pp. 13-24, 2011.
    """

    @property
    def X_(self):
        return self._knn._fit_X

    def __init__(
        self,               algorithm='auto',
        contamination=0.01, leaf_size=30,
        metric='minkowski', n_jobs=1,
        n_neighbors=5,      p=2,
        verbose=False,      metric_params=None
    ):
        super().__init__(contamination=contamination, verbose=verbose)

        self.algorithm     = algorithm
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.n_jobs        = n_jobs
        self.n_neighbors   = n_neighbors
        self.p             = p
        self.metric_params = metric_params

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
        self : FastABOD
            Return self.
        """

        self.check_params(X)

        self._knn         = NearestNeighbors(
            algorithm     = self.algorithm,
            leaf_size     = self.leaf_size,
            metric        = self.metric,
            n_jobs        = self.n_jobs,
            n_neighbors   = self.n_neighbors,
            p             = self.p,
            metric_params = self.metric_params
        ).fit(X)
        self.threshold_   = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

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

        check_is_fitted(self, '_knn')

        if X is None:
            query_is_train = True
            X              = self.X_
            neigh_ind      = self._knn.kneighbors(None, return_distance=False)
        else:
            query_is_train = False
            X              = check_array(X, estimator=self)
            neigh_ind      = self._knn.kneighbors(X, return_distance=False)

        n_samples, _       = X.shape

        result             = Parallel(n_jobs=self.n_jobs)(
            delayed(_approximate_abof)(
                X[s], self.X_, neigh_ind[s]
            ) for s in gen_even_slices(n_samples, self.n_jobs)
        )

        approximate_abof   = np.concatenate(result)

        if query_is_train:
            self.abof_max_ = np.max(approximate_abof)

        # transform raw scores into regular scores
        return np.maximum(0., -np.log(approximate_abof / self.abof_max_))
