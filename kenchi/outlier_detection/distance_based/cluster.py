import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ...utils import assign_info_on_pandas_obj, construct_pandas_obj


class KMeansOutlierDetector(BaseDetector):
    """Outlier detector using k-means clustering.

    Parameters
    ----------
    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    max_iter : int, default 300
        Maximum number of iterations.

    n_clusters : int, default 8
        Number of clusters to form as well as the number of centroids to
        generate.

    n_jobs : int, default 1
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores. Doesn't affect fit method.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    tol : float, default 1e-04
        Tolerance to declare convergence.

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
        self.fpr          = fpr
        self.max_iter     = max_iter
        self.n_clusters   = n_clusters
        self.n_jobs       = n_jobs
        self.random_state = random_state
        self.tol          = tol

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0.0 or 1.0 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.max_iter <= 0:
            raise ValueError(
                'max_iter must be positive but was {0}'.format(
                    self.max_iter
                )
            )

        if self.n_clusters <= 0:
            raise ValueError(
                'n_clusters must be positive but was {0}'.format(
                    self.n_clusters
                )
            )

        if self.tol < 0:
            raise ValueError(
                'tol must be non-negative but was {0}'.format(self.tol)
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

        X                = check_array(X)

        self._kmeans     = KMeans(
            max_iter     = self.max_iter,
            n_clusters   = self.n_clusters,
            n_jobs       = self.n_jobs,
            random_state = self.random_state,
            tol          = self.tol
        ).fit(X)

        self.y_score_    = self.anomaly_score(X)
        self.threshold_  = np.percentile(
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

        check_is_fitted(self, '_kmeans')

        if X is None:
            return self.y_score_
        else:
            X  = check_array(X)

            return np.min(self._kmeans.transform(X), axis=1)
