from warnings import warn

import numpy as np
from sklearn.metrics import auc
from sklearn.utils import check_random_state

__all__        = ['negative_mv_auc_score', 'mv_curve']

MAX_N_FEATURES = 8


def negative_mv_auc_score(
    detector, X=None, y=None, interval=(0.9, 0.999),
    n_uniform_samples=10000
):
    """Compute the opposite of the area under the Mass-Volume (MV) curve.

    Parameters
    ----------
    detector : object
        Detector.

    X : array-like of shape (n_samples, n_features), default None
        Data.

    y : ignored

    interval : tuple, default (0.9, 0.999)
        Interval of probabilities.

    n_uniform_samples : int, default 10000
        Number of samples which are drawn from the uniform distribution over
        the hypercube enclosing the data.

    random_state : int or RandomState instance, default None
        Seed of the pseudo random number generator.

    Returns
    -------
    score : float
        Opposite of the area under the MV curve.

    References
    ----------
    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?,"
        In ICML Anomaly Detection Workshop, 2016.
    """

    mass, volume, _ = mv_curve(
        detector, X, n_uniform_samples=n_uniform_samples
    )
    is_in_range     = (interval[0] <= mass) & (mass <= interval[1])

    return auc(mass[is_in_range], volume[is_in_range], reorder=True)


def mv_curve(detector, X=None, n_uniform_samples=10000):
    """Compute mass-volume pairs for different offsets.

    Parameters
    ----------
    detector : object
        Detector.

    X : array-like of shape (n_samples, n_features), default None
        Data.

    n_uniform_samples : int, default 10000
        Number of samples which are drawn from the uniform distribution over
        the hypercube enclosing the data.

    Returns
    -------
    mass : array-like

    volume : array-like

    offsets : array-like

    References
    ----------
    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?,"
        In ICML Anomaly Detection Workshop, 2016.
    """

    def lebesgue_measure(offset, score_uniform_samples, data_volume):
        return np.mean(score_uniform_samples >= offset) * data_volume

    if detector._n_features > MAX_N_FEATURES:
        warn(
            f'X is expected to have {MAX_N_FEATURES} or less features '
            f'but had {detector._n_features} features'
        )

    rnd                   = check_random_state(detector.random_state)
    U                     = rnd.uniform(
        low               = detector.data_min_,
        high              = detector.data_max_,
        size              = (n_uniform_samples, detector._n_features)
    )
    score_samples         = -detector.anomaly_score(X)
    score_uniform_samples = -detector.anomaly_score(U)

    mass                  = np.linspace(0., 1.)
    offsets               = np.percentile(score_samples, 100. * (1. - mass))
    volume                = np.vectorize(
        lebesgue_measure, excluded=[1, 2]
    )(offsets, score_uniform_samples, detector.data_volume_)

    return mass, volume, offsets
