import numpy as np
import scipy.linalg as LA
from scipy.stats import chi2
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector, DetectorMixin


class GaussianDetector(BaseDetector, DetectorMixin):
    """"Detector in Gaussian distribution.

    Parameters
    ----------
    assume_independent : bool
        If True, each feature is conditionally independent of every other
        feature.

    fpr : float
        False positive rate. Used to compute the threshold.

    n_jobs : integer
        Number of jobs to run in parallel. If -1, then the number of jobs is set
        to the number of CPU cores. Doesn't affect fit method.

    threshold : float or None
        Threshold. If None, it is computed automatically.

    use_method_of_moments : bool
        If True, the method of moments is used to compute the threshold.

    Attributes
    ----------
    mean_ : ndarray, shape = (n_features)
        Mean value for each feature in the training set.

    var_ : ndarray, shape = (n_features)
        Variance for each feature in the training set.

    covariance_ : ndarray, shape = (n_features, n_features)
        Estimated covariance matrix. Stored only if assume_independent is set to
        False.
    """

    def __init__(
        self,           assume_independent=False,
        fpr=0.01,       n_jobs=1,
        threshold=None, use_method_of_moments=False
    ):
        self.assume_independent    = assume_independent
        self.fpr                   = fpr
        self.n_jobs                = n_jobs
        self.threshold             = threshold
        self.use_method_of_moments = use_method_of_moments

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

        X                       = check_array(X)
        n_samples, n_features   = X.shape

        self.mean_              = np.mean(X, axis=0)
        self.var_               = np.var(X, axis=0)

        if not self.assume_independent:
            self.covariance_    = np.cov(X, rowvar=False, bias=1)

        if self.threshold is None:
            if self.use_method_of_moments:
                scores          = self.compute_anomaly_score(X)
                mo1             = np.mean(scores)
                mo2             = np.mean(scores ** 2)
                m_mo            = 2.0 * mo1 ** 2 / (mo2 - mo1 ** 2)
                s_mo            = (mo2 - mo1 ** 2) / 2.0 / mo1
                self._threshold = chi2.ppf(1.0 - self.fpr, m_mo, scale=s_mo)

            else:
                self._threshold = chi2.ppf(1.0 - self.fpr, n_features, scale=1.0)

        else:
            self._threshold     = self.threshold

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

        check_is_fitted(self, ['mean_', 'var_'])

        n_samples, n_features = X.shape

        if self.assume_independent:
            return np.sum(
                ((X - self.mean_) / self.var_) ** 2, axis=1
            )

        else:
            return np.ravel(
                pairwise_distances(
                    X         = X,
                    Y         = np.reshape(self.mean_, (1, n_features)),
                    metric    = 'mahalanobis',
                    n_jobs    = self.n_jobs,
                    V         = self.covariance_
                )
            )
