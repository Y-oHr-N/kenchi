import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj

VALID_KERNELS = [
    'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'
]


class KernelDensityOutlierDetector(KernelDensity, DetectorMixin):
    """Outlier detector using kernel density estimation.

    Parameters
    ----------
    bandwidth : float, default 1.0
        Bandwidth of the kernel.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    kernel : str, default 'gaussian'
        Kernel to use.

    metric : str or callable, default 'minkowski'
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

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.bandwidth <= 0:
            raise ValueError(
                'bandwidth must be positive but was {0}'.format(
                    self.bandwidth
                )
            )

        if self.fpr < 0 or 1 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.kernel not in VALID_KERNELS:
            raise ValueError('invalid kernel: {0}'.format(self.kernel))

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

        super().fit(X)

        y_score         = self.anomaly_score(X)
        self.threshold_ = np.percentile(y_score, 100.0 * (1.0 - self.fpr))

        return self

    @construct_pandas_obj
    def anomaly_score(self, X):
        """Compute anomaly scores for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples,)
            Anomaly scores for test samples.
        """

        check_is_fitted(self, 'tree_')

        X = check_array(X)

        return -self.score_samples(X)
