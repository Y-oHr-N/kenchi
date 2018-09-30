import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector

__all__ = ['LOF']


class LOF(BaseOutlierDetector):
    """Local Outlier Factor.

    Parameters
    ----------
    algorithm : str, default 'auto'
        Tree algorithm to use. Valid algorithms are
        ['kd_tree'|'ball_tree'|'auto'].

    contamination : float, default 'auto'
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

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    References
    ----------
    .. [#breunig00] Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J.,
        "LOF: identifying density-based local outliers,"
        In Proceedings of SIGMOD, pp. 93-104, 2000.

    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert, E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import LOF
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = LOF(n_neighbors=3)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def negative_outlier_factor_(self):
        """array-like of shape (n_samples,): Opposite LOF of the training
        samples.
        """

        return self.estimator_.negative_outlier_factor_

    @property
    def n_neighbors_(self):
        """int: Actual number of neighbors used for ``kneighbors`` queries.
        """

        return self.estimator_.n_neighbors_

    @property
    def X_(self):
        """array-like of shape (n_samples, n_features): Training data.
        """

        return self.estimator_._fit_X

    def __init__(
        self, algorithm='auto', contamination='auto', leaf_size=30,
        metric='minkowski', novelty=False, n_jobs=1, n_neighbors=20,
        p=2, metric_params=None
    ):
        self.algorithm     = algorithm
        self.contamination = contamination
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.novelty       = novelty
        self.n_jobs        = n_jobs
        self.n_neighbors   = n_neighbors
        self.p             = p
        self.metric_params = metric_params

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, ['negative_outlier_factor_', 'n_neighbors_', 'X_']
        )

    def _get_threshold(self):
        return - self.estimator_.offset_ - 1.

    def _fit(self, X):
        self.estimator_   = LocalOutlierFactor(
            algorithm     = self.algorithm,
            contamination = self.contamination,
            leaf_size     = self.leaf_size,
            metric        = self.metric,
            novelty       = self.novelty,
            n_jobs        = self.n_jobs,
            n_neighbors   = self.n_neighbors,
            p             = self.p,
            metric_params = self.metric_params
        ).fit(X)

        return self

    def _anomaly_score(self, X, regularize=True):
        lof = self._lof(X)

        if regularize:
            return np.maximum(0., lof - 1.)
        else:
            return lof

    def _lof(self, X):
        """Compute the Local Outlier Factor (LOF) for each sample."""

        if X is self.X_:
            return -self.negative_outlier_factor_
        else:
            return -self.estimator_.score_samples(X)
