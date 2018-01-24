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
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    verbose : bool, default False
        Enable verbose output.

    weight : bool, default False
        If True, anomaly score is the sum of the distances from k nearest
        neighbors.

    knn_params : dict, default None
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
        contamination: float = 0.01,
        verbose:       bool  = False,
        weight:        bool  = False,
        knn_params:    dict  = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.weight     = weight
        self.knn_params = knn_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

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

        dist, _ = self._knn.kneighbors(X)

        if self.weight:
            return np.sum(dist, axis=1)
        else:
            return np.max(dist, axis=1)


class OneTimeSampling(BaseDetector):
    """One-time sampling.

    Parameters
    ----------
    contamination : float, default 0.01
        Amount of contamination of the data set, i.e. the proportion of
        outliers in the data set. Used to define the threshold.

    n_samples : int, default 20
        Number of random samples to be used.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    verbose : bool, default False
        Enable verbose output.

    metric : str, default 'euclidean'
        Distance metric to use.

    metric_params : dict, default None
        Other keywords passed to the requested metric.

    Attributes
    ----------
    sampled_ : array-like of shape (n_samples,)
        Indices of subsamples.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    X_sampled_ : array-like of shape (n_samples, n_features)
        Subset of the given training data.

    References
    ----------
    M. Sugiyama, and K. Borgwardt,
    "Rapid distance-based outlier detection via sampling,"
    Advances in NIPS'13, pp. 467-475, 2013.
    """

    @property
    def X_sampled_(self) -> TwoDimArray:
        return self.X_[self.sampled_]

    def __init__(
        self,
        contamination: float       = 0.01,
        n_samples:     int         = 20,
        random_state:  RandomState = None,
        verbose:       bool        = False,
        metric:        str         = 'euclidean',
        metric_params: dict        = None
    ) -> None:
        super().__init__(contamination=contamination, verbose=verbose)

        self.n_samples     = n_samples
        self.random_state  = random_state
        self.metric        = metric
        self.metric_params = metric_params

    def check_params(self, X: TwoDimArray, y: OneDimArray = None) -> None:
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

        n_samples, _ = X.shape

        if self.n_samples <= 0:
            raise ValueError(
                f'n_samples must be positive but was {self.n_samples}'
            )

        if self.n_samples >= n_samples:
            raise ValueError(
                f'n_samples must be smaller than {n_samples} ' \
                + f'but was {self.n_samples}'
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

        self.check_params(X)

        self.X_           = check_array(X)
        n_samples, _      = self.X_.shape

        rnd               = check_random_state(self.random_state)
        self.sampled_     = rnd.choice(
            n_samples, size=self.n_samples, replace=False
        )

        # sort again as choice does not guarantee sorted order
        self.sampled_     = np.sort(self.sampled_)

        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params

        self._metric      = DistanceMetric.get_metric(
            self.metric, **metric_params
        )

        self.threshold_   = np.percentile(
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

        check_is_fitted(self, '_metric')

        if X is None:
            X = self.X_
        else:
            X = check_array(X)

        dist  = self._metric.pairwise(X, self.X_sampled_)

        return np.min(dist, axis=1)
