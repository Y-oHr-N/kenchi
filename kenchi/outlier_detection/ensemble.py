from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_is_fitted

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

    contamination : float, default 'auto'
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

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    References
    ----------
    .. [#liu08] Liu, F. T., Ting, K. M., and Zhou, Z.-H.,
        "Isolation forest,"
        In Proceedings of ICDM, pp. 413-422, 2008.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import IForest
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = IForest(random_state=0)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def estimators_(self):
        """list: Collection of fitted sub-estimators.
        """

        return self.estimator_.estimators_

    @property
    def estimators_samples_(self):
        """int: Subset of drawn samples for each base estimator.
        """

        return self.estimator_.estimators_samples_

    @property
    def max_samples_(self):
        """int: Actual number of samples.
        """

        return self.estimator_.max_samples_

    def __init__(
        self, bootstrap=False, contamination='auto', max_features=1.0,
        max_samples='auto', n_estimators=100, n_jobs=1, random_state=None
    ):
        self.bootstrap     = bootstrap
        self.contamination = contamination
        self.max_features  = max_features
        self.max_samples   = max_samples
        self.n_estimators  = n_estimators
        self.n_jobs        = n_jobs
        self.random_state  = random_state

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, ['estimators_', 'estimators_samples_', 'max_samples_']
        )

    def _get_threshold(self):
        return -self.estimator_.offset_

    def _fit(self, X):
        self.estimator_   = IsolationForest(
            behaviour     = 'new',
            bootstrap     = self.bootstrap,
            contamination = self.contamination,
            max_features  = self.max_features,
            max_samples   = self.max_samples,
            n_estimators  = self.n_estimators,
            n_jobs        = self.n_jobs,
            random_state  = self.random_state
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return -self.estimator_.score_samples(X)
