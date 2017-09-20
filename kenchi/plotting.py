import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_score(
    det,              X,
    ax=None,          title=None,
    xlim=None,        ylim=None,
    xlabel='Samples', ylabel='Anomaly score',
    grid=True,        **kwargs
):
    """Plot anomaly scores.

    Parameters
    ----------
    det : object
        kenchi Detector

    X : array-like, shape = (n_samples, n_features)
        Test samples.

    ax : matplotlib Axes, default None
        Target axes instance.

    title : string, default None
        Axes title. To disable, pass None.

    xlim : tuple, default None
        Tuple passed to ax.xlim().

    ylim : tuple, default None
        Tuple passed to ax.ylim().

    xlabel : string, default 'Samples'
        X axis title label. To disable, pass None.

    ylabel : string, default 'Anomaly score'
        Y axis title label. To disable, pass None.

    grid : boolean, default True
        If True, turn the axes grids on.

    **kwargs : dictionary
        Other keywords passed to ax.bar().

    Returns
    -------
    ax : matplotlib Axes
    """

    n_samples, _ = X.shape

    xlocs        = np.arange(n_samples)
    scores       = det.anomaly_score(X)

    if ax is None:
        _, ax    = plt.subplots(1, 1)

    if xlim is None:
        xlim     = (-1, n_samples)

    if ylim is None:
        ylim     = (0, 1.1 * max(max(scores), det.threshold_))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(grid)
    ax.bar(xlocs, scores, align='center', **kwargs)
    ax.hlines(det.threshold_, *xlim)

    return ax
