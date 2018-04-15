import numpy as np
from sklearn.metrics import auc
from sklearn.utils import column_or_1d


def mv_auc_score(
    score_samples, score_uniform_samples, data_volume, interval=(0.9, 0.999)
):
    """Compute area under the Mass-Volume (MV) curve.

    Parameters
    ----------
    score_samples : array-like of shape (n_samples,)
        Opposite of the anomaly score for each sample.

    score_uniform_samples : array-like of shape (n_uniform_samples,)
        Opposite of the anomaly score for each sample which is drawn from the
        uniform distribution over the hypercube enclosing the data.

    data_volume : float
        Volume of the hypercube enclosing the data.

    interval : tuple, default (0.9, 0.999)
        Interval of probabilities.

    Returns
    -------
    auc : float
        Area under the MV curve.

    References
    ----------
    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?,"
        In ICML Anomaly Detection Workshop, 2016.
    """

    mass, volume, _ = mv_curve(
        score_samples, score_uniform_samples, data_volume
    )
    is_in_range     = (interval[0] <= mass) & (mass <= interval[1])

    return auc(mass[is_in_range], volume[is_in_range], reorder=True)


def mv_curve(score_samples, score_uniform_samples, data_volume):
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

    score_samples         = column_or_1d(score_samples)
    score_uniform_samples = column_or_1d(score_uniform_samples)

    mass                  = np.linspace(0., 1.)
    offsets               = np.percentile(score_samples, 100. * (1. - mass))
    volume                = np.vectorize(
        lebesgue_measure, excluded=[1, 2]
    )(offsets, score_uniform_samples, data_volume)

    return mass, volume, offsets
