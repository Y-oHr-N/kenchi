import numpy as np
from sklearn.neighbors import DistanceMetric, NearestNeighbors
from sklearn.utils import check_random_state

from .base import BaseOutlierDetector

__all__ = ['KNN', 'OneTimeSampling']


class KNN(BaseOutlierDetector):
    """Outlier detector using k-nearest neighbors algorithm.

    Parameters
    ----------
    aggregate : bool, default False
        If True, return the sum of the distances from k nearest neighbors as
        the anomaly score.

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
    .. [#angiulli02] Angiulli, F., and Pizzuti, C.,
        "Fast outlier detection in high dimensional spaces,"
        In Proceedings of PKDD'02, pp. 15-27, 2002.

    .. [#ramaswamy00] Ramaswamy, S., Rastogi R., and Shim, K.,
        "Efficient algorithms for mining outliers from large data sets,"
        In Proceedings of SIGMOD'00, pp. 427-438, 2000.
    """

    @property
    def X_(self):
        return self._estimator._fit_X

    def __init__(
        self, aggregate=False, algorithm='auto', contamination=0.1,
        leaf_size=30, metric='minkowski', novelty=False, n_jobs=1,
        n_neighbors=20, p=2, metric_params=None
    ):
        super().__init__(contamination=contamination)

        self.aggregate     = aggregate
        self.algorithm     = algorithm
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.novelty       = novelty
        self.n_jobs        = n_jobs
        self.n_neighbors   = n_neighbors
        self.p             = p
        self.metric_params = metric_params

    def _fit(self, X):
        n_samples, _      = X.shape
        self.n_neighbors_ = np.maximum(
            1, np.minimum(self.n_neighbors, n_samples - 1)
        )
        self._estimator   = NearestNeighbors(
            algorithm     = self.algorithm,
            leaf_size     = self.leaf_size,
            metric        = self.metric,
            n_jobs        = self.n_jobs,
            n_neighbors   = self.n_neighbors_,
            p             = self.p,
            metric_params = self.metric_params
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        if X is self.X_:
            dist, _ = self._estimator.kneighbors()
        else:
            dist, _ = self._estimator.kneighbors(X)

        if self.aggregate:
            return np.sum(dist, axis=1)
        else:
            return np.max(dist, axis=1)


class OneTimeSampling(BaseOutlierDetector):
    """One-time sampling.

    Parameters
    ----------
    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    metric : str, default 'euclidean'
        Distance metric to use.

    novelty : bool, default False
        If True, you can use predict, decision_function and anomaly_score on
        new unseen data and not on the training data.

    n_subsamples : int, default 20
        Number of random samples to be used.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    metric_params : dict, default None
        Additional parameters passed to the requested metric.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    subsamples_ : array-like of shape (n_subsamples,)
        Indices of subsamples.

    S_ : array-like of shape (n_subsamples, n_features)
        Subset of the given training data.

    References
    ----------
    .. [#sugiyama13] Sugiyama M., and Borgwardt, K.,
        "Rapid distance-based outlier detection via sampling,"
        Advances in NIPS'13, pp. 467-475, 2013.
    """

    @property
    def _metric_params(self):
        if self.metric_params is None:
            return dict()
        else:
            return self.metric_params

    def __init__(
        self, contamination=0.1, metric='euclidean', novelty=False,
        n_subsamples=20, random_state=None, metric_params=None
    ):
        super().__init__(contamination=contamination)

        self.metric        = metric
        self.novelty       = novelty
        self.n_subsamples  = n_subsamples
        self.random_state  = random_state
        self.metric_params = metric_params

    def _check_params(self):
        super()._check_params()

        if self.n_subsamples <= 0:
            raise ValueError(
                f'n_subsamples must be positive but was {self.n_subsamples}'
            )

    def _check_array(self, X, **kwargs):
        X            = super()._check_array(X, **kwargs)
        n_samples, _ = X.shape

        if self.n_subsamples >= n_samples:
            raise ValueError(
                f'n_subsamples must be smaller than {n_samples} '
                f'but was {self.n_subsamples}'
            )

        return X

    def _fit(self, X):
        n_samples, _     = X.shape
        rnd              = check_random_state(self.random_state)

        subsamples       = rnd.choice(
            n_samples, size=self.n_subsamples, replace=False
        )

        # sort again as choice does not guarantee sorted order
        self.subsamples_ = np.sort(subsamples)
        self.S_          = X[self.subsamples_]

        self._metric     = DistanceMetric.get_metric(
            self.metric, **self._metric_params
        )

        return self

    def _anomaly_score(self, X):
        return np.min(self._metric.pairwise(X, self.S_), axis=1)
