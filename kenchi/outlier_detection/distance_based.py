import numpy as np
from sklearn.neighbors import DistanceMetric, NearestNeighbors
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit, OneDimArray, RandomState, TwoDimArray

__all__ = ['KNN', 'OneTimeSampling']


class KNN(BaseDetector):
    """Outlier detector using k-nearest neighbors algorithm.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    verbose : bool, default False
        Enable verbose output.

    weight : bool, default False
        If True, anomaly score is the sum of the distances from k nearest
        neighbors.

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
    S. Ramaswamy, R. Rastogi and K. Shim,
    "Efficient algorithms for mining outliers from large data sets,"
    In Proceedings of SIGMOD'00, pp. 427-438, 2000.

    F. Angiulli and C. Pizzuti,
    "Fast outlier detection in high dimensional spaces,"
    In Proceedings of PKDD'02, pp. 15-27, 2002.
    """

    @property
    def X_(self) -> TwoDimArray:
        return self._knn._fit_X

    def __init__(
        self,
        fpr:     float = 0.01,
        verbose: bool  = False,
        weight:  bool  = False,
        **kwargs
    ) -> None:
        super().__init__(fpr=fpr, verbose=verbose)

        self.weight = weight
        self._knn   = NearestNeighbors(**kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params()

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'KNN':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : ignored

        Returns
        -------
        self : KNN
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
            Data.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        dist, _ = self._knn.kneighbors(X)

        if self.weight:
            return np.sum(dist, axis=1)
        else:
            return np.max(dist, axis=1)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        raise NotImplementedError()

    def score(X: TwoDimArray, y: OneDimArray = None) -> float:
        raise NotImplementedError()


class OneTimeSampling(BaseDetector):
    """One-time sampling.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    metric : str, default 'euclidean'
        Distance metric to use.

    n_subsamples : int, default 20
        Number of subsamples.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    verbose : bool, default False
        Enable verbose output.

    kwargs : dict
        Other keywords passed to the requested metric.

    Attributes
    ----------
    sub_ : array-like of shape (n_subsamples,)
        Indices of subsamples.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    X_sub_ : array-like of shape (n_subsamples, n_features)
        Subset of the given training data.

    References
    ----------
    M. Sugiyama, and K. Borgwardt,
    "Rapid distance-based outlier detection via sampling,"
    Advances in NIPS'13, pp. 467-475, 2013.
    """

    @property
    def X_sub_(self) -> TwoDimArray:
        return self.X_[self.sub_]

    def __init__(
        self,
        fpr:          float       = 0.01,
        metric:       str         = 'euclidean',
        n_subsamples: int         = 20,
        random_state: RandomState = None,
        verbose:      bool        = False,
        **kwargs
    ) -> None:
        super().__init__(fpr=fpr, verbose=verbose)

        self.n_subsamples = n_subsamples
        self.random_state = check_random_state(random_state)
        self._metric      = DistanceMetric.get_metric(metric, **kwargs)

        self.check_params()

    def check_params(self) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params()

        if self.n_subsamples <= 0:
            raise ValueError(
                f'n_subsamples must be positive but was {self.n_subsamples}'
            )

    @timeit
    def fit(self, X: TwoDimArray, y: OneDimArray = None) -> 'OneTimeSampling':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training Data.

        y : ignored

        Returns
        -------
        self : OneTimeSampling
            Return self.
        """

        self.X_         = check_array(X)
        n_samples, _    = self.X_.shape

        if n_samples <= self.n_subsamples:
            raise ValueError(
                f'n_samples must be larger than {self.n_subsamples} ' \
                + f'but was {n_samples}'
            )

        self.sub_       = np.sort(
            self.random_state.choice(
                n_samples, size=self.n_subsamples, replace=False
            )
        )

        anomaly_score   = self.anomaly_score()
        self.threshold_ = np.percentile(anomaly_score, 100. * (1. - self.fpr))

        return self

    def anomaly_score(self, X: TwoDimArray = None) -> OneDimArray:
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

        check_is_fitted(self, 'X_')

        if X is None:
            X = self.X_

        dist  = self._metric.pairwise(X, self.X_sub_)

        return np.min(dist, axis=1)

    def feature_wise_anomaly_score(self, X: TwoDimArray = None) -> TwoDimArray:
        raise NotImplementedError()

    def score(X: TwoDimArray, y: OneDimArray = None) -> float:
        raise NotImplementedError()
