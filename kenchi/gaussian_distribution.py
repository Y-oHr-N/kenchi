import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.covariance import MinCovDet, graph_lasso
from sklearn.utils.validation import check_array

from .base import DetectorMixin


class GaussianOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector in Gaussian distribution.

    Parameters
    ----------
    assume_centered : bool
        If True, data are not centered before computation.

    fpr : float
        False positive rate. Used to compute the threshold.

    random_state : int, RandomState instance or None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    support_fraction : float
        Proportion of points to be included in the support of the raw MCD
        estimate.

    use_method_of_moments : bool
        If True, the method of moments is used to compute the threshold.

    Attributes
    ----------
    threshold_ : float
        Threshold.
    """

    def __init__(
        self,                  assume_centered=False,
        fpr=0.01,              random_state=None,
        support_fraction=None, use_method_of_moments=False
    ):
        self.assume_centered       = assume_centered
        self.fpr                   = fpr
        self.random_state          = random_state
        self.support_fraction      = support_fraction
        self.use_method_of_moments = use_method_of_moments

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
        _, n_features        = X.shape

        self._mcd            = MinCovDet(
            assume_centered  = self.assume_centered,
            random_state     = self.random_state,
            support_fraction = self.support_fraction
        ).fit(X)

        if self.use_method_of_moments:
            scores           = self._mcd.dist_
            mo1              = np.mean(scores)
            mo2              = np.mean(scores ** 2)
            m_mo             = 2.0 * mo1 ** 2 / (mo2 - mo1 ** 2)
            s_mo             = 0.5 * (mo2 - mo1 ** 2) / mo1
            self.threshold_  = chi2.ppf(1.0 - self.fpr, m_mo, scale=s_mo)

        else:
            self.threshold_  = chi2.ppf(1.0 - self.fpr, n_features, scale=1.0)

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

        return self._mcd.mahalanobis(X)


class GGMOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector using Gaussian graphical models.

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

    random_state : int, RandomState instance or None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    support_fraction : float
        Proportion of points to be included in the support of the raw MCD
        estimate.

    tol : float
        The tolerance to declare convergence. If the dual gap goes below this
        value, iterations are stopped.

    Attributes
    ----------
    covariance_ : ndarray, shape = (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray, shape = (n_features, n_features)
        Estimated pseudo inverse matrix.

    threshold_ : ndarray, shape = (n_features)
        Threshold.
    """

    def __init__(
        self,       alpha=0.01,        assume_centered=False, max_iter=100,
        q=99.9,     random_state=None, support_fraction=None, tol=0.0001
    ):
        self.alpha            = alpha
        self.assume_centered  = assume_centered
        self.max_iter         = max_iter
        self.q                = q
        self.random_state     = random_state
        self.support_fraction = support_fraction
        self.tol              = tol

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

        X                                 = check_array(X)

        self._mcd                         = MinCovDet(
            assume_centered               = self.assume_centered,
            random_state                  = self.random_state,
            support_fraction              = self.support_fraction
        ).fit(X)

        self.covariance_, self.precision_ = graph_lasso(
            emp_cov                       = self._mcd.covariance_,
            alpha                         = self.alpha,
            max_iter                      = self.max_iter,
            tol                           = self.tol
        )

        scores                            = self.decision_function(X)
        self.threshold_                   = np.percentile(
            a                             = scores,
            q                             = self.q,
            axis                          = 0
        )

        return self

    def decision_function(self, X):
        """Compute the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : ndarray, shape = (n_samples, n_features)
            Anomaly score for test samples.
        """

        first_term  = 0.5 * np.log(2.0 * np.pi / np.diag(self.precision_))

        second_term = 0.5 / np.diag(
            self.precision_
        ) * ((X - self._mcd.location_) @ self.precision_) ** 2

        return first_term + second_term
