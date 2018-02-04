import numpy as np
from sklearn.cluster import MiniBatchKMeans as SKLearnMiniBatchKMeans
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import BaseDetector
from ..utils import timeit

__all__ = ['MiniBatchKMeans']


class MiniBatchKMeans(BaseDetector):
    """Outlier detector using K-means clustering.

    Parameters
    ----------
    batch_size : int, optional, default 100
        Size of the mini batches.

    contamination : float, default 0.01
        Proportion of outliers in the data set. Used to define the threshold.

    init : str or array-like, default 'k-means++'
        Method for initialization.

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

    verbose : bool, default False
        Enable verbose output.

    Attributes
    ----------
    cluster_centers_ : array-like of shape (n_clusters, n_features)
        Coordinates of cluster centers.

    inertia_ : float
        Value of the inertia criterion associated with the chosen partition.

    labels_ : array-like of shape (n_samples,)
        Label of each point.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
    """

    @property
    def cluster_centers_(self):
        return self._kmeans.cluster_centers_

    @property
    def inertia_(self):
        return self._kmeans.inertia_

    @property
    def labels_(self):
        return self._kmeans.labels_

    def __init__(
        self,                    batch_size=100,
        contamination=0.01,      init='k-means++',
        init_size=None,          max_iter=100,
        max_no_improvement=10,   n_clusters=8,
        n_init=3,                random_state=None,
        reassignment_ratio=0.01, tol=0.0,
        verbose=False
    ):
        super().__init__(contamination=contamination, verbose=verbose)

        self.batch_size         = batch_size
        self.init               = init
        self.init_size          = init_size
        self.max_iter           = max_iter
        self.max_no_improvement = max_no_improvement
        self.n_clusters         = n_clusters
        self.n_init             = n_init
        self.random_state       = random_state
        self.reassignment_ratio = reassignment_ratio
        self.tol                = tol

    def check_params(self, X, y=None):
        """Check validity of parameters and raise ValueError if not valid."""

        super().check_params(X)

    @timeit
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : ignored

        Returns
        -------
        self : MiniBatchKMeans
            Return self.
        """

        self.check_params(X)

        self.X_                = check_array(X, estimator=self)
        self._kmeans           = SKLearnMiniBatchKMeans(
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
        self.threshold_        = np.percentile(
            self.anomaly_score(), 100. * (1. - self.contamination)
        )

        return self

    def anomaly_score(self, X=None):
        """Compute the anomaly score for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If not provided, the anomaly score for each training sample
            is returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        check_is_fitted(self, '_kmeans')

        if X is None:
            X = self.X_
        else:
            X = check_array(X, estimator=self)

        return np.min(self._kmeans.transform(X), axis=1)

    def score(self, X, y=None):
        """Compute the opposite value of the given data on the K-means
        objective.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Opposite value of the given data on the K-means objective.
        """

        check_is_fitted(self, '_kmeans')

        return self._kmeans.score(X)
