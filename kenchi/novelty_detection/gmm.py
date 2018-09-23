from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

from ..base import BaseOutlierDetector

__all__ = ['GMM']


class GMM(BaseOutlierDetector):
    """Outlier detector using Gaussian Mixture Models (GMMs).

    Parameters
    ----------
    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    covariance_type : str, default 'full'
        String describing the type of covariance parameters to use. Valid
        options are ['full'|'tied'|'diag'|'spherical'].

    init_params : str, default 'kmeans'
        Method used to initialize the weights, the means and the precisions.
        Valid options are ['kmeans'|'random'].

    max_iter : int, default 100
        Maximum number of iterations.

    means_init : array-like of shape (n_components, n_features), default None
        User-provided initial means.

    n_init : int, default 1
        Number of initializations to perform.

    n_components : int, default 1
        Number of mixture components.

    precisions_init : array-like, default None
        User-provided initial precisions.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    reg_covar : float, default 1e-06
        Non-negative regularization added to the diagonal of covariance.

    tol : float, default 1e-03
        Tolerance to declare convergence.

    warm_start : bool, default False
        If True, the solution of the last fitting is used as initialization for
        the next call of ``fit``.

    weights_init : array-like of shape (n_components,), default None
        User-provided initial weights.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    converged_ : bool
        True when convergence was reached in ``fit``, False otherwise.

    covariances_ : array-like
        Covariance of each mixture component.

    lower_bound_ : float
        Log-likelihood of the best fit of EM.

    means_ : array-like of shape (n_components, n_features)
        Mean of each mixture component.

    n_iter_ : int
        Number of step used by the best fit of EM to reach the convergence.

    precisions_ : array-like
        Precision matrix for each component in the mixture.

    precisions_cholesky_ : array-like
        Cholesky decomposition of the precision matrices of each mixture
        component.

    weights_ : array-like of shape (n_components,)
        Weight of each mixture components.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.novelty_detection import GMM
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = GMM(random_state=0)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def converged_(self):
        return self.estimator_.converged_

    @property
    def covariances_(self):
        return self.estimator_.covariances_

    @property
    def lower_bound_(self):
        return self.estimator_.lower_bound_

    @property
    def means_(self):
        return self.estimator_.means_

    @property
    def n_iter_(self):
        return self.estimator_.n_iter_

    @property
    def precisions_(self):
        return self.estimator_.precisions_

    @property
    def precisions_cholesky_(self):
        return self.estimator_.precisions_cholesky_

    @property
    def weights_(self):
        return self.estimator_.weights_

    def __init__(
        self, contamination=0.1, covariance_type='full', init_params='kmeans',
        max_iter=100, means_init=None, n_components=1, n_init=1,
        precisions_init=None, random_state=None, reg_covar=1e-06, tol=1e-03,
        warm_start=False, weights_init=None
    ):
        self.contamination   = contamination
        self.covariance_type = covariance_type
        self.init_params     = init_params
        self.max_iter        = max_iter
        self.means_init      = means_init
        self.n_components    = n_components
        self.n_init          = n_init
        self.precisions_init = precisions_init
        self.random_state    = random_state
        self.reg_covar       = reg_covar
        self.tol             = tol
        self.warm_start      = warm_start
        self.weights_init    = weights_init

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, [
                'converged_', 'covariances_', 'lower_bound_', 'means_',
                'n_iter_', 'precisions_', 'precisions_cholesky_', 'weights_'
            ]
        )

    def _fit(self, X):
        self.estimator_     = GaussianMixture(
            covariance_type = self.covariance_type,
            init_params     = self.init_params,
            max_iter        = self.max_iter,
            means_init      = self.means_init,
            n_components    = self.n_components,
            n_init          = self.n_init,
            precisions_init = self.precisions_init,
            random_state    = self.random_state,
            reg_covar       = self.reg_covar,
            tol             = self.tol,
            warm_start      = self.warm_start,
            weights_init    = self.weights_init
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return -self.estimator_.score_samples(X)
