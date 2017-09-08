import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_array

from .base import DetectorMixin


class VMFOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector in Von Mises–Fisher distribution.

    Parameters
    ----------
    assume_normalized : bool
        If False, data are normalized before computation.

    fpr : float
        False positive rate. Used to compute the threshold.

    Attributes
    ----------
    mean_direction_ : ndarray, shape = (n_features)
        Mean direction.

    threshold_ : float
        Threshold.
    """

    def __init__(self, assume_normalized=False, fpr=0.01):
        self.assume_normalized = assume_normalized
        self.fpr               = fpr

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : object
            Return self.
        """

        X                    = check_array(X)

        if not self.assume_normalized:
            self._normalizer = Normalizer().fit(X)
            X                = self._normalizer.transform(X)

        mean                 = np.mean(X, axis=0)
        self.mean_direction_ = mean / np.linalg.norm(mean)

        scores               = self.decision_function(X)
        mo1                  = np.mean(scores)
        mo2                  = np.mean(scores ** 2)
        m_mo                 = 2.0 * mo1 ** 2 / (mo2 - mo1 ** 2)
        s_mo                 = 0.5 * (mo2 - mo1 ** 2) / mo1
        self.threshold_      = chi2.ppf(1.0 - self.fpr, m_mo, scale=s_mo)

        return self

    def decision_function(self, X):
        """Compute the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples)
            Anomaly score for test samples.
        """

        if not self.assume_normalized:
            X = self._normalizer.transform(X)

        return 1.0 - X @ self.mean_direction_
