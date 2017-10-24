import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj

VALID_COVARIANCE_TYPES = ['full', 'tied', 'diag', 'spherical']


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

    warm_start : boolean, default False
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
