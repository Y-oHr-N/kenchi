import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals.joblib import delayed, Parallel
from sklearn.utils import check_array, gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, OneDimArray, TwoDimArray

__all__ = ['FastABOD']


def approximate_abof(
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
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    verbose : bool, default False
        Enable verbose output.

    knn_params : dict, default None
        Other keywords passed to sklearn.neighbors.NearestNeighbors().

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
    def X_(self) -> TwoDimArray:
        return self._knn._fit_X

    def __init__(
        self,
        contamination: float = 0.01,
        n_jobs:        int   = 1,
        verbose:       bool  = False,
        knn_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.n_jobs     = n_jobs
        self.knn_params = knn_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'FastABOD':
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

        if self.knn_params is None:
            knn_params  = {}
        else:
            knn_params  = self.knn_params

        self._knn       = NearestNeighbors(**knn_params).fit(X)

        self.threshold_ = np.percentile(
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

        check_is_fitted(self, '_knn')

        if X is None:
            query_is_train = True
            X              = self.X_
            neigh_ind      = self._knn.kneighbors(None, return_distance=False)
        else:
            query_is_train = False
            X              = check_array(X)
            neigh_ind      = self._knn.kneighbors(X, return_distance=False)

        n_samples, _       = X.shape

        result             = Parallel(n_jobs=self.n_jobs)(
            delayed(approximate_abof)(
                X[s], self.X_, neigh_ind[s]
            ) for s in gen_even_slices(n_samples, self.n_jobs)
        )

        abof               = np.concatenate(result)

        if query_is_train:
            self.abof_max_ = np.max(abof)

        # transform raw scores into regular scores
        return np.maximum(0., -np.log(abof / self.abof_max_))
