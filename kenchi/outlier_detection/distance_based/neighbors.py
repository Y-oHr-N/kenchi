import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import DetectorMixin
from ...utils import assign_info_on_pandas_obj, construct_pandas_obj


class KNNOutlierDetector(NearestNeighbors, DetectorMixin):
    """Outlier detector using k-nearest neighbors algorithm.

    Parameters
    ----------
    aggregate : bool, default False
        If True, anomaly score is the sum of the distances from k nearest
        neighbors.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    metric : str or callable, default 'minkowski'
        Metric to use for distance computation.

    metric_params : dict, default None
        Additional keyword arguments for the metric function.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    n_neighbors : int, default 5
        Number of neighbors.

    p : int, default 2
        Power parameter for the Minkowski metric.

    Attributes
    ----------
    threshold_ : float
        Threshold.

    References
    ----------
    S. Ramaswamy, R. Rastogi and K. Shim,
    "Efficient algorithms for mining outliers from large data sets,"
    In Proceedings of SIGMOD'00, pp. 427-438, 2000.

    F. Angiulli and C. Pizzuti,
    "Fast outlier detection in high dimensional spaces,"
    In Proceedings of PKDD'02, pp. 15-27, 2002.
    """

    def __init__(
        self,               aggregate=True,
        fpr=0.01,           metric='minkowski',
        metric_params=None, n_jobs=1,
        n_neighbors=5,      p=2
    ):
        super().__init__(
            metric        = metric,
            metric_params = metric_params,
            n_jobs        = n_jobs,
            n_neighbors   = n_neighbors,
            p             = p
        )

        self.aggregate    = aggregate
        self.fpr          = fpr

    def check_params(self, X):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0.0 or 1.0 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.n_neighbors <= 0:
            raise ValueError(
                'n_neighbors must be positive but was {0}'.format(
                    self.n_neighbors
                )
            )

        if self.p < 1:
            raise ValueError(
                'p must be greater than or equeal to 1 but was {0}'.format(
                    self.p
                )
            )

    @assign_info_on_pandas_obj
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        self : detector
            Return self.
        """

        X               = check_array(X)

        self.check_params(X)

        super().fit(X)

        self.y_score_   = self.anomaly_score()
        self.threshold_ = np.percentile(
            self.y_score_, 100.0 * (1.0 - self.fpr)
        )

        return self

    @construct_pandas_obj
    def anomaly_score(self, X=None):
        """Compute anomaly scores for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples,)
            Anomaly scores for test samples.
        """

        check_is_fitted(self, '_fit_method')

        if X is None:
            X       = self._fit_X
            dist, _ = self.kneighbors(None)
        else:
            X       = check_array(X)
            dist, _ = self.kneighbors(X)

        if np.any(dist == 0.0):
            raise ValueError('X must not contain training samples')

        if self.aggregate:
            return np.sum(dist, axis=1)
        else:
            return np.max(dist, axis=1)
