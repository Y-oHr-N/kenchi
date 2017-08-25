import numpy as np
from scipy.stats import chi2
from sklearn.covariance import empirical_covariance, graph_lasso, ledoit_wolf
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector, DetectorMixin


class GaussianDetector(BaseDetector, DetectorMixin):
    """Detector in Gaussian distribution.

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

    def compute_anomaly_score(self, X):
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
                self._threshold = chi2.ppf(
                    1.0 - self.fpr, n_features, scale=1.0
                )

        else:
            self._threshold     = self.threshold

        return self


class GGMDetector(BaseDetector, DetectorMixin):
    """Detector using Gaussian graphical models.

    Parameters
    ----------
    alpha : float
        Regularization parameter.

    assume_centered : bool
        If True, data are not centered before computation.

    max_iter : integer
        Maximum number of iterations.

    q : float
        Percentile to compute, which must be between 0 and 100 inclusive.

    threshold : float or None
        Threshold. If None, it is computed automatically.

    tol : float
        The tolerance to declare convergence. If the dual gap goes below this
        value, iterations are stopped.

    Attributes
    ----------
    covariance_ : ndarray, shape = (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray, shape = (n_features, n_features)
        Estimated pseudo inverse matrix.
    """

    def __init__(
        self, alpha=0.01,     assume_centered=False, max_iter=100,
        q=99, threshold=None, tol=0.0001
    ):
        self.alpha           = alpha
        self.assume_centered = assume_centered
        self.max_iter        = max_iter
        self.q               = q
        self.threshold       = threshold
        self.tol             = tol

    def compute_anomaly_score(self, X):
        """Compute the anomaly scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples, n_features)
            Anomaly scores for test samples.
        """

        check_is_fitted(self, ['covariance_', 'precision_'])

        first_term  = 0.5 * np.log(2.0 * np.pi / np.diag(self.precision_))
        second_term = 0.5 / np.diag(
            self.precision_
        ) * (X @ self.precision_) ** 2

        return first_term + second_term

    def fit(self, X, y=None, X_valid=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        X_valid : array-like, shape = (n_samples, n_features)
            Validation samples. used to compute to the threshold.

        Returns
        -------
        self : object
            Return self.
        """

        X                                 = check_array(X)

        emp_cov                           = empirical_covariance(
            X                             = X,
            assume_centered               = self.assume_centered
        )

        self.covariance_, self.precision_ = graph_lasso(
            emp_cov                       = emp_cov,
            alpha                         = self.alpha,
            max_iter                      = self.max_iter,
            tol                           = self.tol
        )

        if self.threshold is None:
            if X_valid is None:
                scores                    = self.compute_anomaly_score(X)
            else:
                scores                    = self.compute_anomaly_score(X_valid)

            self._threshold               = np.percentile(
                a                         = scores,
                q                         = self.q,
                axis                      = 0
            )

        else:
            self._threshold               = self.threshold

        return self
