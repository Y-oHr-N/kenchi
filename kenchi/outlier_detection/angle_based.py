from itertools import combinations

import numpy as np
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import gen_even_slices

from .base import ArrayLike, BaseDetector

__all__ = ['FastABOD']


def _abof(X: np.ndarray, ind: np.ndarray, fit_X: np.ndarray) -> np.ndarray:
    """Compute the angle-based outlier factor for each sample."""

    with np.errstate(invalid='raise'):
        return np.var([[
            (pa @ pb) / (pa @ pa) / (pb @ pb) for pa, pb in combinations(
                fit_X[ind_p] - p, 2
            )
        ] for p, ind_p in zip(X, ind)], axis=1)


class FastABOD(BaseDetector):
    """Fast angle-based outlier detector.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    Attributes
    ----------
    anomaly_score_ : np.ndarray of shape (n_samples,)
        Anomaly score for each training sample.

    threshold_ : float
        Threshold.

    References
    ----------
    H.-P. Kriegel, M. Schubert and A. Zimek,
    "Angle-based outlier detection in high-dimensional data,"
    In Proceedings of SIGKDD'08, pp. 444-452, 2008.
    """

    def __init__(self, fpr: float=0.01, n_jobs: int=1, **kwargs) -> None:
        self.fpr    = fpr
        self.n_jobs = n_jobs
        self._knn   = neighbors.NearestNeighbors(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0. or self.fpr > 1.:
            raise ValueError(
                f'fpr must be between 0.0 and 1.0 inclusive but was {self.fpr}'
            )

    def fit(self, X: ArrayLike, y: None=None) -> 'FastABOD':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : None
            Ignored.

        Returns
        -------
        self : FastABOD
            Return self.
        """

        self._knn.fit(X)

        self.anomaly_score_ = self.anomaly_score()
        self.threshold_     = np.percentile(
            self.anomaly_score_, 100.0 * (1.0 - self.fpr)
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
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        ind          = self._knn.kneighbors(X, return_distance=False)

        if X is None:
            X        = self._knn._fit_X

        n_samples, _ = X.shape

        try:
            result   = Parallel(self.n_jobs)(
                delayed(_abof)(
                    X[s], ind[s], self._knn._fit_X
                ) for s in gen_even_slices(n_samples, self.n_jobs)
            )
        except FloatingPointError as e:
            raise ValueError('X must not contain training samples') from e

        return -np.concatenate(result)

    def score(X: ArrayLike, y: None=None) -> float:
        """Compute the mean log-likelihood of the given data."""

        raise NotImplementedError()
