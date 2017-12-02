import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_anomaly_score(
    detector,         X=None,
    ax=None,          title=None,
    xlim=None,        ylim=None,
    xlabel='Samples', ylabel='Anomaly score',
    grid=True,        **kwargs
):
    """Plot anomaly scores for test samples.

    Parameters
    ----------
    detector : detector
        Detector.

    X : array-like of shape (n_samples, n_features), default None
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

    **kwargs : dict
        Other keywords passed to ax.bar().

    Returns
    -------
    ax : matplotlib Axes
    """

    import matplotlib.pyplot as plt

    y_score    = detector.anomaly_score(X)

    n_samples, = y_score.shape
    xlocs      = np.arange(n_samples)

    align      = 'center'
    label      = detector.__class__.__name__

    if ax is None:
        _, ax  = plt.subplots()

    if xlim is None:
        xlim   = (-1, n_samples)

    if ylim is None:
        ylim   = (0, 1.1 * max(max(y_score), detector.threshold_))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(grid)

    ax.bar(xlocs, y_score, align=align, label=label, **kwargs)
    ax.hlines(detector.threshold_, xlim[0], xlim[1])

    ax.legend()

    return ax


def plot_roc_curve(detector, X, y, ax=None, grid=True, title=None, **kwargs):
    """Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    detector : detector
        Detector.

    X : array-like of shape (n_samples, n_features)
        Test samples.

    y : array-like of shape (n_samples,)
        Labels for test samples.

    ax : matplotlib Axes, default None
        Target axes instance.

    grid : boolean, default True
        If True, turn the axes grids on.

    title : string, default None
        Axes title. To disable, pass None.

    **kwargs : dict
        Other keywords passed to ax.plot().

    Returns
    -------
    ax : matplotlib Axes
    """

    import matplotlib.pyplot as plt

    y_score     = detector.anomaly_score(X)
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc     = auc(fpr, tpr)

    label       = detector.__class__.__name__ \
        + ' (auc = {0:.3f})'.format(roc_auc)

    if ax is None:
        _, ax   = plt.subplots(1, 1)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(-0.05, 1.00)
    ax.set_ylim(0.00, 1.05)
    ax.grid(grid)

    ax.plot(fpr, tpr, label=label, **kwargs)
    ax.legend()

    return ax
