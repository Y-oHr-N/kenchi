import numpy as np
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector


class OCSVM(BaseOutlierDetector):
    """One Class Support Vector Machines.

    Parameters
    ----------
    cache_size : float, default 200
        Specify the size of the kernel cache (in MB).

    coef0 : float, default 0.0
        Independent term in kernel function. It is only significant in 'poly'
        and 'sigmoid'.

    degree : int, default 3
        Degree of the polynomial kernel function ('poly'). Ignored by all other
        kernels.

    gamma : float, default 'auto'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto',
        1 / n_features will be used instead.

    kernel : callable or str, default 'rbf'
        Kernel to use. Valid kernels are
        ['linear'|'poly'|'rbf'|'sigmoid'|'precomputed'].

    max_iter : int, optional default -1
        Maximum number of iterations.

    nu : float, default 0.5
        An upper bound on the fraction of training errors and a lower bound of
        the fraction of support vectors. Should be in the interval (0, 1].

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    shrinking : bool, default True
        If True, use the shrinking heuristic.

    tol : float, default 0.001
        Tolerance to declare convergence.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    support_ : array-like of shape (n_SV)
        Indices of support vectors.

    support_vectors_ : array-like of shape (n_SV, n_features)
        Support vectors.

    dual_coef_ : array-like of shape (1, n_SV)
        Coefficients of the support vectors in the decision function.

    coef_ : array-like of shape (1, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`

    intercept_ : array-like of shape (1,)
        Constant in the decision function.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import OCSVM
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = OCSVM(random_state=0)
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def dual_coef_(self):
        return self.estimator_.dual_coef_ / self.nu_l_

    @property
    def support_(self):
        return self.estimator_.support_

    @property
    def support_vectors_(self):
        return self.estimator_.support_vectors_

    @property
    def intercept_(self):
        return self.estimator_.intercept_ / self.nu_l_

    def __init__(
        self, cache_size=200, coef0=0., degree=3,
        gamma='auto', kernel='rbf', max_iter=-1, nu=0.5,
        shrinking=True, tol=0.001, random_state=None
    ):
        super().__init__()

        self.cache_size   = cache_size
        self.coef0        = coef0
        self.degree       = degree
        self.gamma        = gamma
        self.kernel       = kernel
        self.max_iter     = max_iter
        self.nu           = nu
        self.shrinking    = shrinking
        self.tol          = tol
        self.random_state = random_state

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
            coef0        = self.coef0,
            degree       = self.degree,
            gamma        = self.gamma,
            kernel       = self.kernel,
            max_iter     = self.max_iter,
            nu           = self.nu,
            shrinking    = self.shrinking,
            tol          = self.tol,
            random_state = self.random_state
        ).fit(X)

        l,               = self.support_.shape
        self.nu_l_       = self.nu * l

        mettric_params   = {'coef0': self.coef0, 'degree': self.degree}
        self.K_          = PairwiseKernel(
            gamma        = self.estimator_._gamma,
            metric       = self.kernel,
            pairwise_kernels_kwargs = mettric_params
        )

        Q                = self.K_(self.support_vectors_)
        i                = np.where(self.dual_coef_ < 1. / self.nu_l_)[1][0]
        self.Q_ii_       = Q[i, i]

        c2               = (self.dual_coef_ @ Q @ self.dual_coef_.T)[0, 0]
        self.R2_         = c2 + 2. * self.intercept_[0] + self.Q_ii_

        return self

    def _anomaly_score(self, X):
        return self.R2_ \
            - 2. / self.nu_l_ * self.estimator_.decision_function(X).flat[:] \
            - self.Q_ii_ + self.K_.diag(X)
