import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class KernelDensityOutlierDetector(KernelDensity, DetectorMixin):
    """Outlier detector using kernel density estimation.

    Parameters
    ----------
    bandwidth : float, default 1.0
        Bandwidth of the kernel.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kernel : string, default 'gaussian'
        Kernel to use.

    metric : string or callable, default ‘minkowski’
        Metric to use for distance computation.

    metric_params : dict, default None
        Additional keyword arguments for the metric function.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(
        self,               bandwidth=1.0,
        fpr=0.01,           kernel='gaussian',
        metric='euclidean', metric_params=None
    ):
        super().__init__(
            bandwidth     = bandwidth,
            kernel        = kernel,
            metric        = metric,
            metric_params = metric_params
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
            Anomaly scores for test samples.
        """

        check_is_fitted(self, 'tree_')

        X = check_array(X)

        return -self.score_samples(X)
