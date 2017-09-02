import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted

from .base import BaseDetector, DetectorMixin


class GaussianMixtureDetector(BaseDetector, DetectorMixin):
    """Detector using Gaussian mixture models.

    Parameters
    ----------
    fpr : float
        False positive rate. Used to compute the threshold.

    max_iter : integer
        Maximum number of iterations.

    means_init : array-like, shape = (n_components, n_features)
        User-provided initial means.

    n_components : integer
        Number of mixture components.

    precisions_init : array-like
        User-provided initial precisions.

    random_state : integer, RandomState instance or None
        If integer, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.

    threshold : float
        Threshold. If None, it is computed automatically.

    tol : float
        Convergence threshold.

    weights_init : array-like, shape = (n_components)
        User-provided initial weights.

    Attributes
    ----------
    weights_ : ndarray, shape = (n_components)
        Weight of each mixture component.

    means_ : ndarray, shape = (n_components, n_features)
        Mean of each mixture component.

    covariances_ : ndarray
        Covariance of each mixture component.

    precisions_ : ndarray
        Precision matrix of each mixture component.
    """

    def __init__(
        self,              fpr=0.01,
        max_iter=100,      means_init=None,
        n_components=1,    precisions_init=None,
        random_state=None, threshold=None,
        tol=1e-03,         weights_init=None
    ):
        self.fpr             = fpr
        self.max_iter        = max_iter
        self.means_init      = means_init
        self.n_components    = n_components
        self.precisions_init = precisions_init
        self.random_state    = random_state
        self.threshold       = threshold
        self.tol             = tol
        self.weights_init    = weights_init

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

        X                   = check_array(X)

        gmm                 = GaussianMixture(
            max_iter        = self.max_iter,
            means_init      = self.means_init,
            n_components    = self.n_components,
            precisions_init = self.precisions_init,
            random_state    = self.random_state,
            tol             = self.tol,
            weights_init    = self.weights_init
        ).fit(X)

        self.weights_       = gmm.weights_
        self.means_         = gmm.means_
        self.covariances_   = gmm.covariances_
        self.precisions_    = gmm.precisions_

        if self.threshold is None:
            if X_valid is None:
                scores      = self.compute_anomaly_score(X)

            else:
                scores      = self.compute_anomaly_score(X_valid)

            self._threshold = np.percentile(scores, 100.0 * (1.0 - self.fpr))

        else:
            self._threshold = self.threshold

        return self

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

        check_is_fitted(
            self, ['weights_', 'means_', 'covariances_', 'precisions_']
        )

        return -np.log(
            np.sum([
                weight * multivariate_normal.pdf(
                    X, mean=mean, cov=cov
                ) for weight, mean, cov in zip(
                    self.weights_, self.means_, self.covariances_
                )
            ], axis=0)
        )
