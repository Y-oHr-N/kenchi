import numpy as np
from scipy.stats import chi2
from sklearn.covariance import GraphLasso
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import AnalyzerMixin, DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class GaussianOutlierDetector(GraphLasso, AnalyzerMixin, DetectorMixin):
    """Outlier detector in Gaussian distribution.

    Parameters
    ----------
    alpha : float, default 0.01
        Regularization parameter.

    assume_centered : boolean, default False
        If True, data are not centered before computation.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    max_iter : int, default 100
        Maximum number of iterations.

    tol : float, default 0.0001
        Tolerance to declare convergence. If the dual gap goes below this
        value, iterations are stopped.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    threshold_ : float
        Threshold.

    feature_wise_threshold_ : ndarray of shape (n_features,)
        Feature-wise threshold.
    """

    def __init__(
        self,                  alpha=0.01,
        assume_centered=False, fpr=0.01,
        max_iter=100,          tol=0.0001
    ):
        super().__init__(
            alpha           = alpha,
            assume_centered = assume_centered,
            max_iter        = max_iter,
            tol             = tol
        )

        self.fpr            = fpr

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.alpha < 0 or 1 < self.alpha:
            raise ValueError(
                'alpha must be between 0 and 1 inclusive but was {0}'.format(
                    self.alpha
                )
            )

        if self.fpr < 0 or 1 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.max_iter <= 0:
            raise ValueError(
                'max_iter must be positive but was {0}'.format(
                    self.max_iter
                )
            )

        if self.tol < 0:
            raise ValueError(
                'tol must be non-negative but was {0}'.format(self.tol)
            )

    @assign_info_on_pandas_obj
    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        self : detector
            Return self.
        """

        X                            = check_array(X)

        super().fit(X)

        y_score                      = self.anomaly_score(X)
        df, loc, scale               = chi2.fit(y_score)
        self.threshold_              = chi2.ppf(1.0 - self.fpr, df, loc, scale)

        y_score                      = self.feature_wise_anomaly_score(X)
        self.feature_wise_threshold_ = np.percentile(
            a                        = y_score,
            q                        = 100.0 * (1.0 - self.fpr),
            axis                     = 0
        )

        return self

    @construct_pandas_obj
    def anomaly_score(self, X):
        """Compute anomaly scores for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples,)
            Anomaly scores for test samples.
        """

        check_is_fitted(self, ['covariance_', 'precision_'])

        X = check_array(X)

        return self.mahalanobis(X)

    @construct_pandas_obj
    def feature_wise_anomaly_score(self, X):
        """Compute feature-wise anomaly scores for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for test samples.
        """

        check_is_fitted(self, ['covariance_', 'precision_'])

        X = check_array(X)

        return 0.5 * np.log(
            2.0 * np.pi / np.diag(self.precision_)
        ) + 0.5 / np.diag(
            self.precision_
        ) * ((X - self.location_) @ self.precision_) ** 2
