import numpy as np
from scipy.stats import chi2
from sklearn.covariance import GraphLasso
from sklearn.utils.validation import check_array, check_is_fitted

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

    max_iter : integer, default 100
        Maximum number of iterations.

    tol : float, default 0.0001
        Tolerance to declare convergence. If the dual gap goes below this
        value, iterations are stopped.

    Attributes
    ----------
    covariance_ : ndarray, shape = (n_features, n_features)
        Estimated covariance matrix.

    precision_ : ndarray, shape = (n_features, n_features)
        Estimated pseudo inverse matrix.

    threshold_ : float
        Threshold.

    feature_wise_threshold_ : ndarray, shape = (n_features,)
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

        X                            = check_array(X)

        super().fit(X)

        scores                       = self.anomaly_score(X)
        df, loc, scale               = chi2.fit(scores)
        self.threshold_              = chi2.ppf(1.0 - self.fpr, df, loc, scale)

        feature_wise_scores          = self.feature_wise_anomaly_score(X)
        self.feature_wise_threshold_ = np.percentile(
            a                        = feature_wise_scores,
            q                        = 100.0 * (1.0 - self.fpr),
            axis                     = 0
        )

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

        check_is_fitted(self, ['covariance_', 'precision_'])

        X = check_array(X)

        return self.mahalanobis(X)

    @construct_pandas_obj
    def feature_wise_anomaly_score(self, X, y=None):
        """Compute feature-wise anomaly scores for test samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        feature_wise_scores : array-like, shape = (n_samples, n_features)
            Feature-wise anomaly scores for test samples.
        """

        check_is_fitted(self, ['covariance_', 'precision_'])

        X = check_array(X)

        return 0.5 * np.log(
            2.0 * np.pi / np.diag(self.precision_)
        ) + 0.5 / np.diag(
            self.precision_
        ) * ((X - self.location_) @ self.precision_) ** 2
