from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted

from ..base import BaseOutlierDetector

__all__ = ['KDE']


class KDE(BaseOutlierDetector):
    """Outlier detector using Kernel Density Estimation (KDE).

    Parameters
    ----------
    algorithm : str, default 'auto'
        Tree algorithm to use. Valid algorithms are
        ['kd_tree'|'ball_tree'|'auto'].

    atol : float, default 0.0
        Desired absolute tolerance of the result.

    bandwidth : float, default 1.0
        Bandwidth of the kernel.

    breadth_first : bool, default True
        If true, use a breadth-first approach to the problem. Otherwise use a
        depth-first approach.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    kernel : str, default 'gaussian'
        Kernel to use. Valid kernels are
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine'].

    leaf_size : int, default 40
        Leaf size of the underlying tree.

    metric : str, default 'euclidean'
        Distance metric to use.

    rtol : float, default 0.0
        Desired relative tolerance of the result.

    metric_params : dict, default None
        Additional parameters to be passed to the requested metric.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    .. [#parzen62] Parzen, E.,
        "On estimation of a probability density function and mode,"
        Ann. Math. Statist., 33(3), pp. 1065-1076, 1962.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.novelty_detection import KDE
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = KDE()
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def X_(self):
        return self.estimator_.tree_.data

    def __init__(
        self, algorithm='auto', atol=0., bandwidth=1.,
        breadth_first=True, contamination=0.1, kernel='gaussian', leaf_size=40,
        metric='euclidean', rtol=0., metric_params=None
    ):
        self.algorithm     = algorithm
        self.atol          = atol
        self.bandwidth     = bandwidth
        self.breadth_first = breadth_first
        self.contamination = contamination
        self.kernel        = kernel
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.rtol          = rtol
        self.metric_params = metric_params

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(self, 'X_')

    def _fit(self, X):
        self.estimator_   = KernelDensity(
            algorithm     = self.algorithm,
            atol          = self.atol,
            bandwidth     = self.bandwidth,
            breadth_first = self.breadth_first,
            kernel        = self.kernel,
            leaf_size     = self.leaf_size,
            metric        = self.metric,
            rtol          = self.rtol,
            metric_params = self.metric_params
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return -self.estimator_.score_samples(X)
