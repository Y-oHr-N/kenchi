import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..base import BaseOutlierDetector

__all__ = ['HBOS']


class HBOS(BaseOutlierDetector):
    """Histogram-based outlier detector.

    Parameters
    ----------
    bins : int or str, default 'auto'
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

    contamination_ : float
        Actual proportion of outliers in the data set.

    threshold_ : float
        Threshold.

    bin_edges_ : array-like
        Bin edges.

    data_max_ : array-like of shape (n_features,)
        Per feature maximum seen in the data.

    data_min_ : array-like of shape (n_features,)
        Per feature minimum seen in the data.

    hist_ : array-like
        Values of the histogram.

    References
    ----------
    .. [#goldstein12] Goldstein, M., and Dengel, A.,
        "Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm,"
        KI: Poster and Demo Track, pp. 59-63, 2012.

    Examples
    --------
    >>> import numpy as np
    >>> from kenchi.outlier_detection import HBOS
    >>> X = np.array([
    ...     [0., 0.], [1., 1.], [2., 0.], [3., -1.], [4., 0.],
    ...     [5., 1.], [6., 0.], [7., -1.], [8., 0.], [1000., 1.]
    ... ])
    >>> det = HBOS()
    >>> det.fit_predict(X)
    array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1])
    """

    def __init__(self, bins='auto', contamination=0.1, novelty=False):
        self.bins          = bins
        self.contamination = contamination
        self.novelty       = novelty

    def _check_is_fitted(self):
        super()._check_is_fitted()

        check_is_fitted(self, ['bin_edges_', 'hist_'])

    def _fit(self, X):
        _, n_features   = X.shape

        self.data_max_  = np.max(X, axis=0)
        self.data_min_  = np.min(X, axis=0)
        self.hist_      = np.empty(n_features, dtype=object)
        self.bin_edges_ = np.empty(n_features, dtype=object)

        for j, col in enumerate(X.T):
            self.hist_[j], self.bin_edges_[j] = np.histogram(
                col, bins=self.bins, density=True
            )

        return self

    def _anomaly_score(self, X):
        n_samples, _           = X.shape
        anomaly_score          = np.zeros(n_samples)

        for j, col in enumerate(X.T):
            bins,              = self.hist_[j].shape
            bin_width          = self.bin_edges_[j][1] - self.bin_edges_[j][0]

            is_in_range        = (
                (self.data_min_[j] <= col) & (col <= self.data_max_[j])
            )

            ind                = np.digitize(col, self.bin_edges_[j]) - 1
            ind[is_in_range & (ind == bins)] = bins - 1

            prob               = np.zeros(n_samples)
            prob[is_in_range]  = self.hist_[j][ind[is_in_range]] * bin_width

            with np.errstate(divide='ignore'):
                anomaly_score -= np.log(prob)

        return anomaly_score
