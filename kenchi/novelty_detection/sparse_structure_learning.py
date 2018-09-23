import numpy as np
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphLasso
from sklearn.utils.validation import check_is_fitted

from ..base import BaseOutlierDetector
from ..plotting import plot_graphical_model, plot_partial_corrcoef

__all__ = ['SparseStructureLearning']


class SparseStructureLearning(BaseOutlierDetector):
    """Outlier detector using sparse structure learning.

    Parameters
    ----------
    alpha : float, default 0.01
        Regularization parameter.

    assume_centered : bool, default False
        If True, data are not centered before computation.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    enet_tol : float, default 1e-04
        Tolerance for the elastic net solver used to calculate the descent
        direction. This parameter controls the accuracy of the search direction
        for a given column update, not of the overall parameter estimate. Only
        used for mode='cd'.

    max_iter : integer, default 100
        Maximum number of iterations.

    mode : str, default 'cd'
        Lasso solver to use: coordinate descent or LARS.

    tol : float, default 1e-04
        Tolerance to declare convergence.

    apcluster_params : dict, default None
        Additional parameters passed to
        ``sklearn.cluster.affinity_propagation``.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    covariance_ : array-like of shape (n_features, n_features)
        Estimated covariance matrix.

    graphical_model_ : networkx Graph
        GGM.

    isolates_ : array-like of shape (n_isolates,)
        Indices of isolates.

    labels_ : array-like of shape (n_features,)
        Label of each feature.

    location_ : array-like of shape (n_features,)
        Estimated location.

    n_iter_ : int
        Number of iterations run.

    partial_corrcoef_ : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    precision_ : array-like of shape (n_features, n_features)
        Estimated pseudo inverse matrix.

    References
    ----------
    .. [#ide09] Ide, T., Lozano, C., Abe, N., and Liu, Y.,
        "Proximity-based anomaly detection using sparse structure learning,"
        In Proceedings of SDM, pp. 97-108, 2009.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.novelty_detection import SparseStructureLearning
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = SparseStructureLearning()
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    @property
    def _apcluster_params(self):
        if self.apcluster_params is None:
            return dict()
        else:
            return self.apcluster_params

    @property
    def covariance_(self):
        return self.estimator_.covariance_

    @property
    def graphical_model_(self):
        import networkx as nx

        return nx.from_numpy_matrix(np.tril(self.partial_corrcoef_, k=-1))

    @property
    def isolates_(self):
        import networkx as nx

        return np.array(list(nx.isolates(self.graphical_model_)))

    @property
    def location_(self):
        return self.estimator_.location_

    @property
    def n_iter_(self):
        return self.estimator_.n_iter_

    @property
    def partial_corrcoef_(self):
        n_features, _    = self.precision_.shape
        diag             = np.diag(self.precision_)[np.newaxis]
        partial_corrcoef = - self.precision_ / np.sqrt(diag.T @ diag)
        partial_corrcoef.flat[::n_features + 1] = 1.

        return partial_corrcoef

    @property
    def precision_(self):
        return self.estimator_.precision_

    def __init__(
        self, alpha=0.01, assume_centered=False, contamination=0.1,
        enet_tol=1e-04, max_iter=100, mode='cd', tol=1e-04,
        apcluster_params=None
    ):
        self.alpha            = alpha
        self.apcluster_params = apcluster_params
        self.assume_centered  = assume_centered
        self.contamination    = contamination
        self.enet_tol         = enet_tol
        self.max_iter         = max_iter
        self.mode             = mode
        self.tol              = tol

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(
            self, [
                'covariance_', 'labels_', 'location_', 'n_iter_',
                'partial_corrcoef_', 'precision_'
            ]
        )

    def _fit(self, X):
        self.estimator_     = GraphLasso(
            alpha           = self.alpha,
            assume_centered = self.assume_centered,
            enet_tol        = self.enet_tol,
            max_iter        = self.max_iter,
            mode            = self.mode,
            tol             = self.tol
        ).fit(X)

        _, self.labels_     = affinity_propagation(
            self.partial_corrcoef_, **self._apcluster_params
        )

        return self

    def _anomaly_score(self, X):
        return self.estimator_.mahalanobis(X)

    def featurewise_anomaly_score(self, X):
        """Compute the feature-wise anomaly scores for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for each sample.
        """

        self._check_is_fitted()

        X = self._check_array(X, estimator=self)

        return 0.5 * np.log(
            2. * np.pi / np.diag(self.precision_)
        ) + 0.5 / np.diag(
            self.precision_
        ) * ((X - self.location_) @ self.precision_) ** 2

    def plot_graphical_model(self, **kwargs):
        """Plot the Gaussian Graphical Model (GGM).

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        random_state : int, RandomState instance, default None
            Seed of the pseudo random number generator.

        title : string, default 'GGM (n_clusters, n_features, n_isolates)'
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to ``nx.draw_networkx``.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        self._check_is_fitted()

        n_clusters  = np.max(self.labels_) + 1
        n_isolates, = self.isolates_.shape
        title       = (
            f'GGM ('
            f'n_clusters={n_clusters}, '
            f'n_features={self.n_features_}, '
            f'n_isolates={n_isolates}'
            f')'
        )
        kwargs['G'] = self.graphical_model_

        kwargs.setdefault('node_color', self.labels_)
        kwargs.setdefault('title', title)

        return plot_graphical_model(**kwargs)

    def plot_partial_corrcoef(self, **kwargs):
        """Plot the partial correlation coefficient matrix.

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        cbar : bool, default True.
            If Ture, to draw a colorbar.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        title : string, default 'Partial correlation'
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to ``ax.pcolormesh``.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        self._check_is_fitted()

        kwargs['partial_corrcoef'] = self.partial_corrcoef_

        return plot_partial_corrcoef(**kwargs)
