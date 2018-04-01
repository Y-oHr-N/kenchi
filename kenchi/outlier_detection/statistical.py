import numpy as np
from sklearn.cluster import affinity_propagation
from sklearn.covariance import GraphLasso
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted

from .base import BaseOutlierDetector
from ..visualization import plot_graphical_model, plot_partial_corrcoef

__all__ = ['GMM', 'HBOS', 'KDE', 'SparseStructureLearning']


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
        the next call of `fit`.

    weights_init : array-like of shape (n_components,), default None
        User-provided initial weights.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    converged_ : bool
        True when convergence was reached in `fit`, False otherwise.

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
    """

    @property
    def converged_(self):
        return self._estimator.converged_

    @property
    def covariances_(self):
        return self._estimator.covariances_

    @property
    def lower_bound_(self):
        return self._estimator.lower_bound_

    @property
    def means_(self):
        return self._estimator.means_

    @property
    def n_iter_(self):
        return self._estimator.n_iter_

    @property
    def precisions_(self):
        return self._estimator.precisions_

    @property
    def precisions_cholesky_(self):
        return self._estimator.precisions_cholesky_

    @property
    def weights_(self):
        return self._estimator.weights_

    def __init__(
        self, contamination=0.1, covariance_type='full', init_params='kmeans',
        max_iter=100, means_init=None, n_components=1, n_init=1,
        precisions_init=None, random_state=None, reg_covar=1e-06, tol=1e-03,
        warm_start=False, weights_init=None
    ):
        super().__init__(contamination=contamination)

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

    def _fit(self, X):
        self._estimator     = GaussianMixture(
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
        return -self._estimator.score_samples(X)

    def score(self, X, y=None):
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored.

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        check_is_fitted(self, '_estimator')

        X = self._check_array(X, n_features=self._n_features, estimator=self)

        return self._estimator.score(X)


class HBOS(BaseOutlierDetector):
    """Histogram-based outlier detector.

    Parameters
    ----------
    bins : int, str or array-like, default 'auto'
        Number of hist bins.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    novelty : bool, default False
        If True, you can use predict, decision_function and anomaly_score on
        new unseen data and not on the training data.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    bin_edges_ : array-like
        Bin edges.

    bin_widths_ : array-like
        Bin widths.

    data_min_ : array-like of shape (n_features,)
        Per feature minimum seen in the data.

    data_max_ : array-like of shape (n_features,)
        Per feature maximum seen in the data.

    hist_ : array-like of shape (n_features, bins)
        Values of the histogram.

    X_ : array-like of shape (n_samples, n_features)
        Training data.

    References
    ----------
    .. [#goldstein12] Goldstein, M., and Dengel, A.,
        "Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm,"
        KI'12: Poster and Demo Track, pp. 59-63, 2012.
    """

    def __init__(self, bins='auto', contamination=0.1, novelty=False):
        super().__init__(contamination=contamination)

        self.bins    = bins
        self.novelty = novelty

    def _fit(self, X):
        self.data_min_          = np.min(X, axis=0)
        self.data_max_          = np.max(X, axis=0)
        self.hist_              = np.empty(self._n_features, dtype=object)
        self.bin_edges_         = np.empty(self._n_features, dtype=object)
        self.bin_widths_        = np.empty(self._n_features)

        for j in range(self._n_features):
            self.hist_[j], self.bin_edges_[j] = np.histogram(
                X[:, j], bins=self.bins, density=True
            )
            self.bin_widths_[j] = (
                self.bin_edges_[j][1] - self.bin_edges_[j][0]
            )

        return self

    def _anomaly_score(self, X):
        n_samples, n_features                = X.shape
        anomaly_score                        = np.zeros(n_samples)

        for j in range(n_features):
            bins                             = self.hist_[j].size

            is_in_range                      = (
                (self.data_min_[j] <= X[:, j]) & (X[:, j] <= self.data_max_[j])
            )

            ind                              = np.digitize(
                X[:, j], self.bin_edges_[j]
            ) - 1
            ind[is_in_range & (ind == bins)] = bins - 1

            prob                             = np.zeros(n_samples)
            prob[is_in_range]                = (
                self.hist_[j][ind[is_in_range]] * self.bin_widths_[j]
            )

            with np.errstate(divide='ignore'):
                anomaly_score               -= np.log(prob)

        return anomaly_score


class KDE(BaseOutlierDetector):
    """Outlier detector using Kernel Density Estimation (KDE).

    Parameters
    ----------
    algorithm : str, default 'auto'
        Tree algorithm to use. Valid algorithms are
        ['kd_tree'|'ball_tree'|'auto'].

    atol : float, default 0.0
        Desired absolute tolerance of the result.

    bandwidth : float, default 1.0
        Bandwidth of the kernel.

    breadth_first : bool, default True
        If true, use a breadth-first approach to the problem. Otherwise use a
        depth-first approach.

    contamination : float, default 0.1
        Proportion of outliers in the data set. Used to define the threshold.

    kernel : str, default 'gaussian'
        Kernel to use. Valid kernels are
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine'].

    leaf_size : int, default 40
        Leaf size of the underlying tree.

    metric : str, default 'euclidean'
        Distance metric to use.

    rtol : float, default 0.0
        Desired relative tolerance of the result.

    metric_params : dict, default None
        Additional parameters to be passed to the requested metric.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

    threshold_ : float
        Threshold.

    X_ : array-like of shape (n_samples, n_features)
        Training data.
    """

    @property
    def X_(self):
        return self._estimator.tree_.data

    def __init__(
        self, algorithm='auto', atol=0., bandwidth=1.,
        breadth_first=True, contamination=0.1, kernel='gaussian', leaf_size=40,
        metric='euclidean', rtol=0., metric_params=None
    ):
        super().__init__(contamination=contamination)

        self.algorithm     = algorithm
        self.atol          = atol
        self.bandwidth     = bandwidth
        self.breadth_first = breadth_first
        self.kernel        = kernel
        self.leaf_size     = leaf_size
        self.metric        = metric
        self.rtol          = rtol
        self.metric_params = metric_params

    def _fit(self, X):
        self._estimator   = KernelDensity(
            algorithm     = self.algorithm,
            atol          = self.atol,
            bandwidth     = self.bandwidth,
            breadth_first = self.breadth_first,
            kernel        = self.kernel,
            leaf_size     = self.leaf_size,
            metric        = self.metric,
            rtol          = self.rtol,
            metric_params = self.metric_params
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return -self._estimator.score_samples(X)

    def score(self, X, y=None):
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        check_is_fitted(self, '_estimator')

        X = self._check_array(X, n_features=self._n_features, estimator=self)

        return np.mean(self._estimator.score_samples(X))


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
        Additional parameters passed to `sklearn.cluster.affinity_propagation`.

    Attributes
    ----------
    anomaly_score_ : array-like of shape (n_samples,)
        Anomaly score for each training data.

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
    .. [#ide09] Ide, T., Lozano, C., Abe N., and Liu, Y.,
        "Proximity-based anomaly detection using sparse structure learning,"
        In Proceedings of SDM'09, pp. 97-108, 2009.
    """

    @property
    def _apcluster_params(self):
        if self.apcluster_params is None:
            return dict()
        else:
            return self.apcluster_params

    @property
    def covariance_(self):
        return self._estimator.covariance_

    @property
    def graphical_model_(self):
        import networkx as nx

        return nx.from_numpy_matrix(np.tril(self.partial_corrcoef_, k=-1))

    @property
    def isolates_(self):
        import networkx as nx

        return np.array(list(nx.isolates(self.graphical_model_)))

    @property
    def labels_(self):
        # cluster using affinity propagation
        _, labels = affinity_propagation(
            self.partial_corrcoef_, **self._apcluster_params
        )

        return labels

    @property
    def location_(self):
        return self._estimator.location_

    @property
    def n_iter_(self):
        return self._estimator.n_iter_

    @property
    def partial_corrcoef_(self):
        n_features, _    = self.precision_.shape
        diag             = np.diag(self.precision_)[np.newaxis]
        partial_corrcoef = - self.precision_ / np.sqrt(diag.T @ diag)
        partial_corrcoef.flat[::n_features + 1] = 1.

        return partial_corrcoef

    @property
    def precision_(self):
        return self._estimator.precision_

    def __init__(
        self, alpha=0.01, assume_centered=False, contamination=0.1,
        enet_tol=1e-04, max_iter=100, mode='cd', tol=1e-04,
        apcluster_params=None
    ):
        super().__init__(contamination=contamination)

        self.alpha            = alpha
        self.apcluster_params = apcluster_params
        self.assume_centered  = assume_centered
        self.enet_tol         = enet_tol
        self.max_iter         = max_iter
        self.mode             = mode
        self.tol              = tol

    def _fit(self, X):
        self._estimator     = GraphLasso(
            alpha           = self.alpha,
            assume_centered = self.assume_centered,
            enet_tol        = self.enet_tol,
            max_iter        = self.max_iter,
            mode            = self.mode,
            tol             = self.tol
        ).fit(X)

        return self

    def _anomaly_score(self, X):
        return self._estimator.mahalanobis(X)

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

        check_is_fitted(self, '_estimator')

        X = self._check_array(X, n_features=self._n_features, estimator=self)

        return 0.5 * np.log(
            2. * np.pi / np.diag(self.precision_)
        ) + 0.5 / np.diag(
            self.precision_
        ) * ((X - self.location_) @ self.precision_) ** 2

    def score(self, X, y=None):
        """Compute the mean log-likelihood of the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Mean log-likelihood of the given data.
        """

        check_is_fitted(self, '_estimator')

        X = self._check_array(X, n_features=self._n_features, estimator=self)

        return self._estimator.score(X)

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
            Other keywords passed to `nx.draw_networkx`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        check_is_fitted(self, '_estimator')

        title       = (
            f'GGM ('
            f'n_clusters={np.max(self.labels_) + 1}, '
            f'n_features={self._n_features}, '
            f'n_isolates={self.isolates_.size}'
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
            Whether to draw a colorbar.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filename : str, default None
            If provided, save the current figure.

        title : string, default 'Partial correlation'
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to `ax.pcolormesh`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        check_is_fitted(self, '_estimator')

        kwargs['partial_corrcoef'] = self.partial_corrcoef_

        return plot_partial_corrcoef(**kwargs)
