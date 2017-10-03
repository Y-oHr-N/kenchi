import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class VMFOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector in Von Mises–Fisher distribution.

    Parameters
    ----------
    assume_normalized : boolean, default False
        If False, data are normalized before computation.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    Attributes
    ----------
    mean_direction_ : ndarray, shape = (n_features,)
        Mean direction.

    threshold_ : float
        Threshold.
    """

    def __init__(self, assume_normalized=False, fpr=0.01):
        self.assume_normalized = assume_normalized
        self.fpr               = fpr

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

        X                    = check_array(X)

        if not self.assume_normalized:
            self._normalizer = Normalizer().fit(X)
            X                = self._normalizer.transform(X)

        mean                 = np.mean(X, axis=0)
        self.mean_direction_ = mean / np.linalg.norm(mean)

        scores               = self.anomaly_score(X)
        df, loc, scale       = chi2.fit(scores)
        self.threshold_      = chi2.ppf(1.0 - self.fpr, df, loc, scale)

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

        check_is_fitted(self, 'mean_direction_')

        X     = check_array(X)

        if not self.assume_normalized:
            X = self._normalizer.transform(X)

        return 1.0 - X @ self.mean_direction_
