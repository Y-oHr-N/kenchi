import numpy as np
from sklearn.base import BaseEstimator
from sklearn.covariance import graph_lasso, MinCovDet
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import holdattr, window_generator


class GGMChangeDetector(BaseEstimator, DetectorMixin):
    """Change detector using Gaussian graphical models.

    Parameters
    ----------
    alpha : float
        Regularization parameter.

    assume_centered : bool
        If True, data are not centered before computation.

    fpr : float
        False positive rate. Used to compute the threshold.

    max_iter : integer
        Maximum number of iterations.

    random_state : integer, RandomState instance or None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    shift : integer
        Shift size.

    support_fraction : float
        Proportion of points to be included in the support of the raw MCD
        estimate.

    tol : float
        The tolerance to declare convergence. If the dual gap goes below this
        value, iterations are stopped.

    window : integer
        Window size.

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
        self,                  alpha=0.01,
        assume_centered=False, max_iter=100,
        fpr=0.01,              random_state=None,
        shift=50,              support_fraction=None,
        tol=0.0001,            window=100
    ):
        self.alpha            = alpha
        self.assume_centered  = assume_centered
        self.max_iter         = max_iter
        self.fpr              = fpr
        self.random_state     = random_state
        self.shift            = shift
        self.support_fraction = support_fraction
        self.tol              = tol
        self.window           = window

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
            q                             = 100.0 * (1.0 - self.fpr),
            axis                          = 0
        )

        return self

    @holdattr
    def decision_function(self, X):
        """Compute the anomaly score.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : array-like, shape = (n_windows, n_features)
            Anomaly score for test samples.
        """

        check_is_fitted(self, ['_mcd'])

        X                        = check_array(X)
        n_samples, n_features    = X.shape

        scores                   = np.empty(
            ((n_samples - self.window + self.shift) // self.shift, n_features)
        )

        for i, X_window in enumerate(
            window_generator(X, self.window, self.shift)
        ):
            mcd_window           = MinCovDet(
                assume_centered  = self.assume_centered,
                random_state     = self.random_state,
                support_fraction = self.support_fraction
            ).fit(X_window)

            _, precision_window  = graph_lasso(
                emp_cov          = mcd_window.covariance_,
                alpha            = self.alpha,
                max_iter         = self.max_iter,
                tol              = self.tol
            )

            scores[i]            = 0.5 * np.log(
                np.diag(self.precision_) / np.diag(precision_window)
            ) - 0.5 * (
                np.diag(
                    self.precision_ @ self._mcd.covariance_ @ self.precision_
                ) / np.diag(self.precision_) - np.diag(
                    precision_window @ self._mcd.covariance_ @ precision_window
                ) / np.diag(precision_window)
            )

        return scores
