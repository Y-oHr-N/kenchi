import numpy as np
from scipy.stats import chi2
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector, DetectorMixin


class VMFDetector(BaseDetector, DetectorMixin):
    """Detector in Von Mises–Fisher distribution.

    Parameters
    ----------
    assume_normalized : bool
        If False, data are normalized before computation.

    fpr : float
        False positive rate. Used to compute the threshold.

    norm : ‘l1’, ‘l2’, or ‘max’
        Norm to use to normalize each non zero sample.

    threshold : float or None
        Threshold. If None, it is computed automatically.

    Attributes
    ----------
    mean_direction_ : ndarray, shape = (n_features)
        Mean direction.
    """

    def __init__(
        self, assume_normalized=False, fpr=0.01, norm='l2', threshold=None
    ):
        self.assume_normalized = assume_normalized
        self.fpr               = fpr
        self.norm              = norm
        self.threshold         = threshold

    def fit(self, X, y=None):
        """Fits the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self : object
            Returns self.
        """

        X                    = check_array(X)

        if not self.assume_normalized:
            self._normalizer = Normalizer(norm=self.norm).fit(X)
            X                = self._normalizer.transform(X)

        mean                 = np.mean(X, axis=0)
        self.mean_direction_ = mean / np.sqrt(mean @ mean)

        if self.threshold is None:
            scores           = self.compute_anomaly_score(X)
            mo1              = np.mean(scores)
            mo2              = np.mean(scores ** 2)
            m_mo             = 2.0 * mo1 ** 2 / (mo2 - mo1 ** 2)
            s_mo             = (mo2 - mo1 ** 2) / 2.0 / mo1
            self._threshold  = chi2.ppf(1.0 - self.fpr, m_mo, scale=s_mo)

        else:
            self._threshold  = self.threshold

        return self

    def compute_anomaly_score(self, X):
        """Computes the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples)
            Anomaly score for test samples.
        """

        check_is_fitted(self, ['mean_direction_'])

        if not self.assume_normalized:
            X = self._normalizer.transform(X)

        return 1.0 - X @ self.mean_direction_
