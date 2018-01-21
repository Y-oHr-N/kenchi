import itertools

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.externals.joblib import delayed, Parallel
from sklearn.utils import gen_even_slices

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
            ]
            for query_point, X_neigh in zip(X, X_train[neigh_ind])
        ], axis=1)


class FastABOD(BaseDetector):
    """Fast Angle-Based Outlier Detector (FastABOD).

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        Other keywords passed to sklearn.neighbors.NearestNeighbors().

    Attributes
    ----------
    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    H.-P. Kriegel, M. Schubert and A. Zimek,
    "Angle-based outlier detection in high-dimensional data,"
    In Proceedings of SIGKDD'08, pp. 444-452, 2008.
    """

    @property
    def X_(self) -> TwoDimArray:
        return self._knn._fit_X

    def __init__(
        self,
        fpr:     float = 0.01,
        n_jobs:  int   = 1,
        verbose: bool  = False,
        **kwargs
    ) -> None:
        super().__init__(fpr=fpr, verbose=verbose)

        self.n_jobs = n_jobs
        self._knn   = NearestNeighbors(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params()

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

        self._knn.fit(X)

        anomaly_score   = self.anomaly_score()
        self.threshold_ = np.percentile(anomaly_score, 100. * (1. - self.fpr))

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

        neigh_ind    = self._knn.kneighbors(X, return_distance=False)

        if X is None:
            X        = self.X_

        n_samples, _ = X.shape

        try:
            result   = Parallel(n_jobs=self.n_jobs)(
                delayed(approximate_abof)(
                    X[s], self.X_, neigh_ind[s]
                ) for s in gen_even_slices(n_samples, self.n_jobs)
            )
        except FloatingPointError as e:
            raise ValueError('X must not contain training samples') from e

        return -np.concatenate(result)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        raise NotImplementedError()

    def score(X: TwoDimArray, y: OneDimArray = None) -> float:
        raise NotImplementedError()
