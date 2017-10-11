import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class EmpiricalOutlierDetector(NearestNeighbors, DetectorMixin):
    """Outlier detector using k-nearest neighbors algorithm.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    metric : string or callable, default ‘minkowski’
        Metric to use for distance computation.

    metric_params : dict, default None
        Additional keyword arguments for the metric function.

    n_jobs : integer, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores. Doesn't affect fit method.

    n_neighbors : integer, default 5
        Number of neighbors.

    p : integer, default 2
        Power parameter for the Minkowski metric.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(
        self,               fpr=0.01,
        metric='minkowski', metric_params=None,
        n_jobs=1,           n_neighbors=5,
        p=2
    ):
        super().__init__(
            metric        = metric,
            metric_params = metric_params,
            n_jobs        = n_jobs,
            n_neighbors   = n_neighbors,
            p             = p
        )

        self.fpr          = fpr

    @assign_info_on_pandas_obj
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : detector
            Return self.
        """

        X               = check_array(X)

        super().fit(X)

        scores          = self.anomaly_score(X)
        self.threshold_ = np.percentile(scores, 100.0 * (1.0 - self.fpr))

        return self

    @construct_pandas_obj
    def anomaly_score(self, X, y=None):
        """Compute anomaly scores for test samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : array-like, shape = (n_samples,)
            anomaly scores for test samples.
        """

        check_is_fitted(self, '_fit_method')

        X             = check_array(X)
        _, n_features = X.shape

        dist, _       = self.kneighbors(X)
        radius        = np.max(dist, axis=1)

        return -np.log(self.n_neighbors) + n_features * np.log(radius)
