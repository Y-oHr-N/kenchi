import numpy as np
from sklearn.metrics import auc
from sklearn.utils import check_random_state

__all__        = ['mv_curve', 'NegativeMVAUCScorer']

MAX_N_FEATURES = 8


def mv_curve(det, X_train, X_test, n_uniform_samples=1000, random_state=None):
    """Compute mass-volume pairs for different offsets.

    Parameters
    ----------
    det : object
        Detector.

    X_train : array-like of shape (n_samples, n_features)
        Training data.

    X_test : array-like of shape (n_samples, n_features)
        Test data.

    n_uniform_samples : int, default 1000
        Number of samples which are drawn from the uniform distribution over
        the hypercube enclosing the data.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

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

    if det.n_features_ > MAX_N_FEATURES:
        raise ValueError(
            f'X is expected to have {MAX_N_FEATURES} or less features '
            f'but had {det.n_features_} features'
        )

    rnd                   = check_random_state(random_state)

    data_min              = np.min(X_train, axis=0)
    data_max              = np.max(X_train, axis=0)
    data_volume           = np.prod(data_max - data_min)

    U                     = rnd.uniform(
        low=data_min, high=data_max, size=(n_uniform_samples, det.n_features_)
    )

    score_samples         = det.score_samples(X_test)
    score_uniform_samples = det.score_samples(U)

    mass                  = np.linspace(0., 1.)
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
        self, X, interval=(0.9, 0.999), n_uniform_samples=1000,
        random_state=None
    ):
        self.X                 = X
        self.interval          = interval
        self.n_uniform_samples = n_uniform_samples
        self.random_state      = random_state

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

        mass, volume, _       = mv_curve(
            det,
            self.X,
            X,
            n_uniform_samples = self.n_uniform_samples,
            random_state      = self.random_state
        )

        is_in_range           = \
            (self.interval[0] <= mass) & (mass <= self.interval[1])

        return -auc(mass[is_in_range], volume[is_in_range], reorder=True)
