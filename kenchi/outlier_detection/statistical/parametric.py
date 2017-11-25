import numpy as np
from scipy.stats import chi2
from sklearn.base import BaseEstimator
from sklearn.covariance import GraphLasso
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import AnalyzerMixin, DetectorMixin
from ...utils import assign_info_on_pandas_obj, construct_pandas_obj

VALID_COVARIANCE_TYPES = ['full', 'tied', 'diag', 'spherical']


class GaussianOutlierDetector(GraphLasso, AnalyzerMixin, DetectorMixin):
    """Outlier detector in Gaussian distribution.

    Parameters
    ----------
    alpha : float, default 0.01
        Regularization parameter.

    assume_centered : bool, default False
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

    References
    ----------
    T. Ide, C. Lozano, N. Abe and Y. Liu,
    "Proximity-based anomaly detection using sparse structure learning,"
    In Proceedings of SDM'09, pp. 97-108, 2009.
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


class GaussianMixtureOutlierDetector(GaussianMixture, DetectorMixin):
    """Outlier detector using Gaussian mixture models.

    Parameters
    ----------
    covariance_type : ['full', 'tied', 'diag', 'spherical'], default 'full'
        String describing the type of covariance parameters to use.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    max_iter : int, default 100
        Maximum number of iterations.

    means_init : array-like of shape (n_components, n_features), default None
        User-provided initial means.

    n_components : int, default 1
        Number of mixture components.

    precisions_init : array-like, default None
        User-provided initial precisions.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    tol : float, default 1e-03
        Convergence threshold.

    warm_start : bool, default False
        If True, the solution of the last fitting is used as initialization for
        the next call of fit().

    weights_init : array-like of shape (n_components,), default None
        User-provided initial weights.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Weight of each mixture component.

    means_ : ndarray of shape (n_components, n_features)
        Mean of each mixture component.

    covariances_ : ndarray
        Covariance of each mixture component.

    precisions_ : ndarray
        Precision matrix of each mixture component.

    threshold_ : float
        Threshold.
    """

    def __init__(
        self,                 covariance_type='full',
        fpr=0.01,             max_iter=100,
        means_init=None,      n_components=1,
        precisions_init=None, random_state=None,
        tol=1e-03,            warm_start=False,
        weights_init=None
    ):
        super().__init__(
            covariance_type = covariance_type,
            max_iter        = max_iter,
            means_init      = means_init,
            n_components    = n_components,
            precisions_init = precisions_init,
            random_state    = random_state,
            tol             = tol,
            warm_start      = warm_start,
            weights_init    = weights_init
        )

        self.fpr            = fpr

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.covariance_type not in VALID_COVARIANCE_TYPES:
            raise ValueError(
                'invalid covariance_type: {0}'.format(self.covariance_type)
            )

        if self.fpr < 0 or 1 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
            )

        if self.max_iter <= 0:
            raise ValueError(
                'max_iter must be positive but was {0}'.format(self.max_iter)
            )

        if self.n_components <= 0:
            raise ValueError(
                'n_components must be positive but was {0}'.format(
                    self.n_components
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

        X               = check_array(X)

        super().fit(X)

        y_score         = self.anomaly_score(X)
        self.threshold_ = np.percentile(y_score, 100.0 * (1.0 - self.fpr))

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

        check_is_fitted(
            self, ['weights_', 'means_', 'covariances_', 'precisions_']
        )

        X = check_array(X)

        return -self.score_samples(X)


class VMFOutlierDetector(BaseEstimator, DetectorMixin):
    """Outlier detector in Von Mises–Fisher distribution.

    Parameters
    ----------
    assume_normalized : bool, default False
        If False, data are normalized before computation.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    Attributes
    ----------
    mean_direction_ : ndarray of shape (n_features,)
        Mean direction.

    threshold_ : float
        Threshold.
    """

    def __init__(self, assume_normalized=False, fpr=0.01):
        self.assume_normalized = assume_normalized
        self.fpr               = fpr

        self.check_params()

    def check_params(self):
        """Check validity of parameters and raise ValueError if not valid."""

        if self.fpr < 0 or 1 < self.fpr:
            raise ValueError(
                'fpr must be between 0 and 1 inclusive but was {0}'.format(
                    self.fpr
                )
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

        X                    = check_array(X)

        if not self.assume_normalized:
            self._normalizer = Normalizer().fit(X)
            X                = self._normalizer.transform(X)

        mean                 = np.mean(X, axis=0)
        self.mean_direction_ = mean / np.linalg.norm(mean)

        y_score              = self.anomaly_score(X)
        df, loc, scale       = chi2.fit(y_score)
        self.threshold_      = chi2.ppf(1.0 - self.fpr, df, loc, scale)

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

        check_is_fitted(self, 'mean_direction_')

        X     = check_array(X)

        if not self.assume_normalized:
            X = self._normalizer.transform(X)

        return 1.0 - X @ self.mean_direction_
