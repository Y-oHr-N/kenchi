import numpy as np
from sklearn.decomposition import PCA as _PCA
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector

__all__ = ['PCA']


class PCA(BaseOutlierDetector):
    """Outlier detector using Principal Component Analysis (PCA).

    Parameters
    ----------
    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    iterated_power : int, default 'auto'
        Number of iterations for the power method computed by svd_solver ==
        'randomized'.

    n_components : int, float, or string, default None
        Number of components to keep.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    svd_solver : string, default 'auto'
        SVD solver to use. Valid solvers are
        ['auto'|'full'|'arpack'|'randomized'].

    tol : float, default 0.0
        Tolerance to declare convergence for singular values computed by
        svd_solver == 'arpack'.

    whiten : bool, default False
        If True, the ``components_`` vectors are multiplied by the square root
        of n_samples and then divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances.

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
    >>> from kenchi.outlier_detection import PCA
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = PCA()
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def components_(self):
        """array-like of shape (n_components, n_features): Principal axes in
        feature space, representing the directions of maximum variance in the
        data.
        """

        return self.estimator_.components_

    @property
    def explained_variance_(self):
        """array-like of shape (n_components,): Amount of variance explained by
        each of the selected components.
        """

        return self.estimator_.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """array-like of shape (n_components,): Percentage of variance
        explained by each of the selected components.
        """

        return self.estimator_.explained_variance_ratio_

    @property
    def mean_(self):
        """array-like of shape (n_features,): Per-feature empirical mean,
        estimated from the training set.
        """

        return self.estimator_.mean_

    @property
    def noise_variance_(self):
        """float: Estimated noise covariance following the Probabilistic PCA
        model from Tipping and Bishop 1999.
        """

        return self.estimator_.noise_variance_

    @property
    def n_components_(self):
        """int: Estimated number of components.
        """

        return self.estimator_.n_components_

    @property
    def singular_values_(self):
        """array-like of shape (n_components,): Singular values corresponding
        to each of the selected components.
        """

        return self.estimator_.singular_values_

    def __init__(
        self, contamination=0.1, iterated_power='auto', n_components=None,
        random_state=None, svd_solver='auto', tol=0., whiten=False
    ):
        self.contamination  = contamination
        self.iterated_power = iterated_power
        self.n_components   = n_components
        self.random_state   = random_state
        self.svd_solver     = svd_solver
        self.tol            = tol
        self.whiten         = whiten

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, [
                'components_', 'explained_variance_',
                'explained_variance_ratio_', 'mean_',
                'noise_variance_', 'n_components_',
                'singular_values_'
            ]
        )

    def _fit(self, X):
        self.estimator_    = _PCA(
            iterated_power = self.iterated_power,
            n_components   = self.n_components,
            random_state   = self.random_state,
            svd_solver     = self.svd_solver,
            tol            = self.tol,
            whiten         = self.whiten
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return np.sum((X - self._reconstruct(X)) ** 2, axis=1)

    def _reconstruct(self, X):
        """Apply dimensionality reduction to the given data, and transform the
        data back to its original space.
        """

        return self.estimator_.inverse_transform(self.estimator_.transform(X))
