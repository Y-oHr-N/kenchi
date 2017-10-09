import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import DetectorMixin
from ..utils import assign_info_on_pandas_obj, construct_pandas_obj


class GaussianMixtureOutlierDetector(GaussianMixture, DetectorMixin):
    """Outlier detector using Gaussian mixture models.

    Parameters
    ----------
    covariance_type : [‘full’, ‘tied’, ‘diag’, ‘spherical’], default 'full'
        String describing the type of covariance parameters to use.

    fpr : float, default 0.01
        False positive rate. Used to compute the threshold.

    max_iter : integer, default 100
        Maximum number of iterations.

    means_init : array-like, shape = (n_components, n_features), default None
        User-provided initial means.

    n_components : integer, default 1
        Number of mixture components.

    precisions_init : array-like, default None
        User-provided initial precisions.

    random_state : integer, RandomState instance, default None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    tol : float, default 1e-03
        Convergence threshold.

    warm_start : boolean, default False
        If True, the solution of the last fitting is used as initialization for
        the next call of fit().

    weights_init : array-like, shape = (n_components,), default None
        User-provided initial weights.

    Attributes
    ----------
    weights_ : ndarray, shape = (n_components,)
        Weight of each mixture component.

    means_ : ndarray, shape = (n_components, n_features)
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

        X               = check_array(X)

        super().fit(X)

        scores          = self.anomaly_score(X)
        self.threshold_ = np.percentile(scores, 100.0 * (1.0 - self.fpr))

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

        check_is_fitted(
            self, ['weights_', 'means_', 'covariances_', 'precisions_']
        )

        X = check_array(X)

        return -self.score_samples(X)
