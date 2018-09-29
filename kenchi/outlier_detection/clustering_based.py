import numpy as np
from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector

__all__ = ['MiniBatchKMeans']


class MiniBatchKMeans(BaseOutlierDetector):
    """Outlier detector using K-means clustering.

    Parameters
    ----------
    batch_size : int, optional, default 100
        Size of the mini batches.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    init : str or array-like, default 'k-means++'
        Method for initialization. Valid options are ['k-means++'|'random'].

    init_size : int, default: 3 * batch_size
        Number of samples to randomly sample for speeding up the
        initialization.

    max_iter : int, default 100
        Maximum number of iterations.

    max_no_improvement : int, default 10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed inertia. To disable
        convergence detection based on inertia, set max_no_improvement to None.

    n_clusters : int, default 8
        Number of clusters.

    n_init : int, default 3
        Number of initializations to perform.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    reassignment_ratio : float, default 0.01
        Control the fraction of the maximum number of counts for a center to be
        reassigned.

    tol : float, default 0.0
        Tolerance to declare convergence.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import MiniBatchKMeans
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = MiniBatchKMeans(n_clusters=1, random_state=0)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def cluster_centers_(self):
        """array-like of shape (n_clusters, n_features): Coordinates of cluster
        centers.
        """

        return self.estimator_.cluster_centers_

    @property
    def inertia_(self):
        """float: Value of the inertia criterion associated with the chosen
        partition.
        """

        return self.estimator_.inertia_

    @property
    def labels_(self):
        """array-like of shape (n_samples,): Label of each point.
        """

        return self.estimator_.labels_

    def __init__(
        self, batch_size=100, contamination=0.1, init='k-means++',
        init_size=None, max_iter=100, max_no_improvement=10, n_clusters=8,
        n_init=3, random_state=None, reassignment_ratio=0.01, tol=0.0
    ):
        self.batch_size         = batch_size
        self.contamination      = contamination
        self.init               = init
        self.init_size          = init_size
        self.max_iter           = max_iter
        self.max_no_improvement = max_no_improvement
        self.n_clusters         = n_clusters
        self.n_init             = n_init
        self.random_state       = random_state
        self.reassignment_ratio = reassignment_ratio
        self.tol                = tol

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(self, ['cluster_centers_', 'inertia_', 'labels_'])

    def _fit(self, X):
        self.estimator_        = _MiniBatchKMeans(
            batch_size         = self.batch_size,
            init               = self.init,
            init_size          = self.init_size,
            max_iter           = self.max_iter,
            max_no_improvement = self.max_no_improvement,
            n_clusters         = self.n_clusters,
            n_init             = self.n_init,
            random_state       = self.random_state,
            reassignment_ratio = self.reassignment_ratio,
            tol                = self.tol
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return np.min(self.estimator_.transform(X), axis=1)
