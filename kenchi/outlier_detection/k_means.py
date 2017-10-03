import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class KMeansOutlierDetector(KMeans, DetectorMixin):
    """Outlier detector using k-means clustering.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    max_iter : integer, default 300
        Maximum number of iterations.

    n_clusters : integer, default 8
        Number of clusters to form as well as the number of centroids to
        generate.

    n_jobs : integer, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores. Doesn't affect fit method.

    random_state : integer, RandomState instance, default None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    tol : float, default 1e-04
        Convergence threshold.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(
        self,         fpr=0.01,
        max_iter=300, n_clusters=8,
        n_jobs=1,     random_state=None,
        tol=1e-04
    ):
        super().__init__(
            max_iter     = max_iter,
            n_clusters   = n_clusters,
            n_jobs       = n_jobs,
            random_state = random_state,
            tol          = tol
        )

        self.fpr         = fpr

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

        check_is_fitted(self, 'cluster_centers_')

        X  = check_array(X)

        return np.min(self.transform(X), axis=1)
