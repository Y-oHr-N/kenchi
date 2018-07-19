import numpy as np
from sklearn.metrics import auc, recall_score
from sklearn.utils import check_random_state

__all__ = ['LeeLiuScorer', 'NegativeMVAUCScorer']


def _lebesgue_measure(score_samples, offset, data_volume):
    """Compute Lebesgue measure."""

    return np.mean(score_samples >= offset) * data_volume


class LeeLiuScorer:
    """Lee-Liu scorer.

    References
    ----------
    .. [#lee03] Lee, W. S, and Liu, B.,
        "Learning with positive and unlabeled examples using weighted Logistic
        Regression,"
        In Proceedings of ICML, pp. 448-455, 2003.
    """

    def __call__(self, det, X, y=None):
        """Compute the Lee-Liu metric.

        Parameters
        ----------
        det : object
            Detector.

        X : array-like of shape (n_samples, n_features), default None
            Data.

        y : array-like of shape (n_samples,), default None
            Labels. If None, assume that all samples are positive.

        Returns
        -------
        score : float
            Lee-Liu metric.
        """

        y_pred = det.predict(X)

        if y is None:
            y  = np.ones_like(y_pred)

        r      = recall_score(y, y_pred)

        return r ** 2 / (1. - det.contamination_)


class NegativeMVAUCScorer:
    """Negative MV AUC scorer.

    Parameters
    ----------
    data_max : array-like of shape (n_features,)
        Per feature maximum seen in the data.

    data_min : array-like of shape (n_features,)
        Per feature minimum seen in the data.

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

    def __init__(
        self, data_max, data_min, interval=(0.9, 0.999),
        n_offsets=1000, n_uniform_samples=1000, random_state=None
    ):
        self.data_max          = data_max
        self.data_min          = data_min
        self.interval          = interval
        self.n_offsets         = n_offsets
        self.n_uniform_samples = n_uniform_samples
        self.random_state      = random_state
        self.internal_state    = check_random_state(random_state).get_state()

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

        rnd                   = np.random.RandomState()

        rnd.set_state(self.internal_state)

        U                     = rnd.uniform(
            low               = self.data_min,
            high              = self.data_max,
            size              = (self.n_uniform_samples, det.n_features_)
        )

        score_samples         = det.score_samples(X)
        score_uniform_samples = det.score_samples(U)

        mass, volume, _       = self._mv_curve(
            score_samples, score_uniform_samples
        )

        is_in_range           = \
            (self.interval[0] <= mass) & (mass <= self.interval[1])

        return -auc(mass[is_in_range], volume[is_in_range], reorder=True)

    def _mv_curve(self, score_samples, score_uniform_samples):
        """Compute mass-volume pairs for different offsets.

        Parameters
        ----------
        score_samples : array-like of shape (n_samples,)
            Opposite of the anomaly score for each sample.

        score_uniform_samples : array-like of shape (n_uniform_samples,)
            Opposite of the anomaly score for each sample which is drawn from
            the uniform distribution over the hypercube enclosing the data.

        Returns
        -------
        mass : array-like of shape (n_offsets,)

        volume : array-like of shape (n_offsets,)

        offsets : array-like of shape (n_offsets,)
        """

        data_volume           = np.prod(self.data_max - self.data_min)

        mass                  = np.linspace(0., 1., self.n_offsets)
        offsets               = np.percentile(
            score_samples, 100. * (1. - mass)
        )
        volume                = np.vectorize(
            _lebesgue_measure, excluded=[0, 2]
        )(score_uniform_samples, offsets, data_volume)

        return mass, volume, offsets
