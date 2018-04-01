from sklearn.ensemble import IsolationForest

from .base import BaseOutlierDetector

__all__ = ['IForest']


class IForest(BaseOutlierDetector):
    """Isolation forest (iForest).

    Parameters
    ----------
    bootstrap : bool, False
        If True, individual trees are fit on random subsets of the training
        data sampled with replacement. If False, sampling without replacement
        is performed.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    max_features : int or float, default 1.0
        Number of features to draw from X to train each base estimator.

    max_samples : int ,float or str, default 'auto'
        Number of samples to draw from X to train each base estimator.

    n_estimators : int, default 100
        Number of base estimators in the ensemble.

    n_jobs : int
        Number of jobs to run in parallel. If -1, then the number of jobs is
        set to the number of CPU cores.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    estimators_ : list
        Collection of fitted sub-estimators.

    estimators_samples_ : int
        Subset of drawn samples for each base estimator.

    max_samples_ : int
        Actual number of samples.

    References
    ----------
    .. [#liu08] Liu, F. T., Ting K. M., and Zhou, Z.-H.,
        "Isolation forest,"
        In Proceedings of ICDM'08, pp. 413-422, 2008.
    """

    @property
    def estimators_(self):
        return self._estimator.estimators_

    @property
    def estimators_samples_(self):
        return self._estimator.estimators_samples_

    @property
    def max_samples_(self):
        return self._estimator.max_samples_

    def __init__(
        self, bootstrap=False, contamination=0.1, max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None
    ):
        super().__init__(contamination=contamination)

        self.bootstrap    = bootstrap
        self.max_features = max_features
        self.max_samples  = max_samples
        self.n_estimators = n_estimators
        self.n_jobs       = n_jobs
        self.random_state = random_state

    def _fit(self, X):
        self._estimator  = IsolationForest(
            bootstrap    = self.bootstrap,
            max_features = self.max_features,
            max_samples  = self.max_samples,
            n_estimators = self.n_estimators,
            n_jobs       = self.n_jobs,
            random_state = self.random_state
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return 0.5 - self._estimator.decision_function(X)
