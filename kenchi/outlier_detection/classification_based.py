from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector

__all__ = ['OCSVM']


class OCSVM(BaseOutlierDetector):
    """One Class Support Vector Machines (only RBF kernel).

    Parameters
    ----------
    cache_size : float, default 200
        Specify the size of the kernel cache (in MB).

    gamma : float, default 'scale'
        Kernel coefficient. If gamma is 'scale', 1 / (n_features * np.std(X))
        will be used instead.

    max_iter : int, optional default -1
        Maximum number of iterations.

    nu : float, default 0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Should be in the interval (0, 1].

    shrinking : bool, default True
        If True, use the shrinking heuristic.

    tol : float, default 0.001
        Tolerance to declare convergence.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import OCSVM
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = OCSVM(gamma=1e-03, nu=0.25)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def dual_coef_(self):
        """array-like of shape (1, n_SV): Coefficients of the support vectors
        in the decision function.
        """

        return self.estimator_.dual_coef_ / self.nu_l_

    @property
    def support_(self):
        """array-like of shape (n_SV): Indices of support vectors.
        """

        return self.estimator_.support_

    @property
    def support_vectors_(self):
        """array-like of shape (n_SV, n_features): Support vectors.
        """

        return self.estimator_.support_vectors_

    @property
    def intercept_(self):
        """array-like of shape (1,): Constant in the decision function.
        """

        return self.estimator_.intercept_ / self.nu_l_

    def __init__(
        self, cache_size=200, gamma='scale', max_iter=-1,
        nu=0.5, shrinking=True, tol=0.001
    ):
        self.cache_size   = cache_size
        self.gamma        = gamma
        self.max_iter     = max_iter
        self.nu           = nu
        self.shrinking    = shrinking
        self.tol          = tol

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, ['dual_coef_', 'intercept_', 'support_', 'support_vectors_']
        )

    def _get_threshold(self):
        return self.R2_

    def _fit(self, X):
        self.estimator_  = OneClassSVM(
            cache_size   = self.cache_size,
            gamma        = self.gamma,
            max_iter     = self.max_iter,
            nu           = self.nu,
            shrinking    = self.shrinking,
            tol          = self.tol
        ).fit(X)

        l,               = self.support_.shape
        self.nu_l_       = self.nu * l

        Q                = rbf_kernel(
            self.support_vectors_, gamma=self.estimator_._gamma
        )
        c2               = (self.dual_coef_ @ Q @ self.dual_coef_.T)[0, 0]
        self.R2_         = c2 + 2. * self.intercept_[0] + 1.

        return self

    def _anomaly_score(self, X):
        return self.R2_ \
            - 2. / self.nu_l_ * self.estimator_.decision_function(X)
