import numpy as np
from sklearn.metrics import auc, roc_curve

from .utils import Axes, Colormap, Limits, OneDimArray, TwoDimArray

__all__ = [
    'plot_anomaly_score',
    'plot_roc_curve',
    'plot_graphical_model',
    'plot_partial_corrcoef'
]


def plot_anomaly_score(
    anomaly_score: OneDimArray,
    threshold:     float  = None,
    ax:            Axes   = None,
    title:         str    = None,
    xlim:          Limits = None,
    ylim:          Limits = None,
    xlabel:        str    = 'Samples',
    ylabel:        str    = 'Anomaly score',
    grid:          bool   = True,
    filepath:      str    = None,
    **kwargs
) -> Axes:
    """Plot the anomaly score for each sample.

    Parameters
    ----------
    anomaly_score : array-like of shape (n_samples,)
        Anomaly score for each sample.

    threshold : float, default None
        Threshold.

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

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to ax.plot().

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_anomaly_score.png
        :align: center
        :alt: Anomaly score
    """

    import matplotlib.pyplot as plt

    n_samples, = anomaly_score.shape
    xlocs      = np.arange(n_samples)

    if ax is None:
        _, ax  = plt.subplots()

    if xlim is None:
        xlim   = (0., n_samples - 1.)

    if ylim is None:
        ylim   = (0., 1.1 * np.max(anomaly_score))

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(grid)
    ax.plot(xlocs, anomaly_score, **kwargs)

    if threshold is not None:
        ax.hlines(threshold, xlim[0], xlim[1])

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax


def plot_roc_curve(
    y_true:   OneDimArray,
    y_score:  OneDimArray,
    ax:       Axes = None,
    label:    str  = None,
    title:    str  = None,
    xlabel:   str  = 'False Positive Rate',
    ylabel:   str  = 'True Positive Rate',
    grid:     bool = True,
    filepath: str  = None,
    **kwargs
) -> Axes:
    """Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True Labels.

    y_score : array-like of shape (n_samples,)
        Target scores.

    ax : matplotlib Axes, default None
        Target axes instance.

    label : str, default None
        Legend label.

    title : string, default None
        Axes title. To disable, pass None.

    grid : boolean, default True
        If True, turn the axes grids on.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to ax.plot().

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.
    """

    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    if ax is None:
        _, ax   = plt.subplots()

    if label is None:
        label   = f'(area = {roc_auc:1.3f})'

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.05)
    ax.grid(grid)
    ax.plot(fpr, tpr, label=label, **kwargs)
    ax.legend(loc='lower right')

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax


def plot_graphical_model(
    partial_corrcoef: TwoDimArray,
    ax:               Axes = None,
    title:            str  = 'Graphical model',
    filepath:         str  = None,
    **kwargs
) -> Axes:
    """Plot the Gaussian Graphical Model (GGM).

    Parameters
    ----------
    partial_corrcoef : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    ax : matplotlib Axes, default None
        Target axes instance.

    title : string, default 'Graphical model'
        Axes title. To disable, pass None.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to nx.draw_networkx().

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_graphical_model.png
        :align: center
        :alt: Graphical model
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    if ax is None:
        _, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    nx.draw_networkx(
        nx.from_numpy_matrix(partial_corrcoef),
        ax    = ax,
        **kwargs
    )

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax


def plot_partial_corrcoef(
    partial_corrcoef: TwoDimArray,
    ax:               Axes     = None,
    cmap:             Colormap = None,
    vmin:             float    = -1.,
    vmax:             float    = 1.,
    cbar:             bool     = True,
    title:            str      = 'Partial correlation',
    filepath:         str      = None,
    **kwargs
) -> Axes:
    """Plot the partial correlation coefficient matrix.

    Parameters
    ----------
    partial_corrcoef : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    ax : matplotlib Axes, default None
        Target axes instance.

    cmap : matplotlib Colormap, default None
        If None, plt.cm.RdYlBu is used.

    vmin : float, default -1.0
        Used in conjunction with norm to normalize luminance data.

    vmax : float, default 1.0
        Used in conjunction with norm to normalize luminance data.

    cbar : bool, default True.
        Whether to draw a colorbar.

    title : string, default 'Partial correlation'
        Axes title. To disable, pass None.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to ax.imshow().

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_partial_corrcoef.png
        :align: center
        :alt: Partial correlation
    """

    import matplotlib.pyplot as plt

    n_features, _ = partial_corrcoef.shape

    if ax is None:
        _, ax     = plt.subplots()

    if cmap is None:
        cmap      = plt.cm.RdYlBu

    if title is not None:
        ax.set_title(title)

    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_features))

    mappable      = ax.imshow(
        np.ma.masked_equal(partial_corrcoef, 0.),
        cmap      = cmap,
        vmin      = vmin,
        vmax      = vmax,
        **kwargs
    )

    if cbar:
        ax.figure.colorbar(mappable, ax=ax)

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax
