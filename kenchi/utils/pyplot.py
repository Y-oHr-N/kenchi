import matplotlib.pyplot as plt
import numpy as np

# TODO: Implement plot_roc_curve function


def plot_anomaly_score(
    detector,         X,
    ax=None,          grid=True,
    xlim=None,        ylim=None,
    xlabel='Samples', ylabel='Anomaly score',
    title=None,       **kwargs
):
    """Plot anomaly scores for test samples.

    Parameters
    ----------
    detector : detector
        Detector.

    X : array-like of shape (n_samples, n_features)
        Test samples.

    ax : matplotlib Axes, default None
        Target axes instance.

    grid : boolean, default True
        If True, turn the axes grids on.

    xlim : tuple, default None
        Tuple passed to ax.xlim().

    ylim : tuple, default None
        Tuple passed to ax.ylim().

    xlabel : string, default 'Samples'
        X axis title label. To disable, pass None.

    ylabel : string, default 'Anomaly score'
        Y axis title label. To disable, pass None.

    title : string, default None
        Axes title. To disable, pass None.

    **kwargs : dictionary
        Other keywords passed to ax.bar().

    Returns
    -------
    ax : matplotlib Axes
    """

    if X is None:
        n_samples, _ = detector._fit_X.shape
    else:
        n_samples, _ = X.shape

    xlocs            = np.arange(n_samples)
    y_score          = detector.anomaly_score(X)
    color            = np.where(
        detector.detect(X).astype(np.bool), '#ff2800', '#0041ff'
    )

    if ax is None:
        _, ax        = plt.subplots(1, 1)

    if xlim is None:
        xlim         = (-1, n_samples)

    if ylim is None:
        ylim         = (0, 1.1 * max(max(y_score), detector.threshold_))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.grid(grid)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.bar(xlocs, y_score, align='center', color=color, **kwargs)
    ax.hlines(detector.threshold_, *xlim)

    return ax
