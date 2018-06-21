import numpy as np
from sklearn.metrics import auc, recall_score
from sklearn.utils import check_array, check_random_state, column_or_1d

from .utils import check_contamination

__all__ = ['lee_liu_score', 'mv_curve', 'NegativeMVAUCScorer']


def lee_liu_score(y_true, y_pred, contamination=None):
    """Compute the Lee-Liu metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : array-like of shape (n_samples,)
        Predicted labels as returned by a detector.

    contamination : float, default None
        Proportion of outliers in the data set.

    Returns
    -------
    score : float
        Lee-Liu metric.

    References
    ----------
    .. [lee03] Lee, W. S, and Liu, B.,
        "Learning with positive and unlabeled examples using weighted Logistic
        Regression,"
        In Proceedings of ICML, pp. 448-455, 2003.
    """

    r                 = recall_score(y_true, y_pred)

    if contamination is None:
        is_outlier    = y_pred == -1
        n_samples,    = is_outlier.shape
        n_outliers    = np.sum(is_outlier)
        contamination = n_outliers / n_samples
    else:
        check_contamination(contamination)

    return r ** 2 / (1. - contamination)


def mv_curve(
    score_samples, score_uniform_samples, data_volume, n_offsets=1000
):
    """Compute mass-volume pairs for different offsets.

    Parameters
    ----------
    score_samples : array-like of shape (n_samples,)
        Opposite of the anomaly score for each sample.

    score_uniform_samples : array-like of shape (n_uniform_samples,)
        Opposite of the anomaly score for each sample which is drawn from the
        uniform distribution over the hypercube enclosing the data.

    data_volume : float
        Volume of the hypercube enclosing the data.

    n_offsets : int, default 1000
        Number of offsets.

    Returns
    -------
    mass : array-like of shape (n_offsets,)

    volume : array-like of shape (n_offsets,)

    offsets : array-like of shape (n_offsets,)

    References
    ----------
    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?"
        In ICML Anomaly Detection Workshop, 2016.
    """

    def lebesgue_measure(offset, score_uniform_samples, data_volume):
        return np.mean(score_uniform_samples >= offset) * data_volume

    score_samples         = column_or_1d(score_samples)
    score_uniform_samples = column_or_1d(score_uniform_samples)

    mass                  = np.linspace(0., 1., n_offsets)
    offsets               = np.percentile(score_samples, 100. * (1. - mass))
    volume                = np.vectorize(
        lebesgue_measure, excluded=[1, 2]
    )(offsets, score_uniform_samples, data_volume)

    return mass, volume, offsets


class NegativeMVAUCScorer:
    """Negative MV AUC scorer.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    interval : tuple, default (0.9, 0.999)
        Interval of probabilities.

    n_offsets : int, default 1000
        Number of offsets.

    n_uniform_samples : int, default 1000
        Number of samples which are drawn from the uniform distribution over
        the hypercube enclosing the data.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    References
    ----------
    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?"
        In ICML Anomaly Detection Workshop, 2016.
    """

    max_n_features = 8

    def __init__(
        self, X, interval=(0.9, 0.999), n_offsets=1000,
        n_uniform_samples=1000, random_state=None
    ):
        self.interval    = interval
        self.n_offsets   = n_offsets

        rnd              = check_random_state(random_state)
        X                = self._check_array(X)
        _, n_features    = X.shape
        data_max         = np.max(X, axis=0)
        data_min         = np.min(X, axis=0)

        self.data_volume = np.prod(data_max - data_min)
        self.U           = rnd.uniform(
            low=data_min, high=data_max, size=(n_uniform_samples, n_features)
        )

    def __call__(self, det, X, y=None):
        """Compute the opposite of the area under the Mass-Volume (MV) curve.

        Parameters
        ----------
        det : object
            Detector.

        X : array-like of shape (n_samples, n_features)
            Data.

        y : ignored

        Returns
        -------
        score : float
            Opposite of the area under the MV curve.
        """

        score_samples         = det.score_samples(X)
        score_uniform_samples = det.score_samples(self.U)

        mass, volume, _       = mv_curve(
            score_samples,
            score_uniform_samples,
            self.data_volume,
            n_offsets         = self.n_offsets
        )

        is_in_range           = \
            (self.interval[0] <= mass) & (mass <= self.interval[1])

        return -auc(mass[is_in_range], volume[is_in_range], reorder=True)

    def _check_array(self, X, **kwargs):
        """Raise ValueError if the array is not valid."""

        X             = check_array(X, **kwargs)
        _, n_features = X.shape

        if n_features > self.max_n_features:
            raise ValueError(
                f'X is expected to have {self.max_n_features} or less '
                f'features but had {n_features} features'
            )

        return X
