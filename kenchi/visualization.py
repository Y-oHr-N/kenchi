import numpy as np
from sklearn.metrics import auc, roc_curve

__all__ = [
    'plot_anomaly_score',
    'plot_roc_curve',
    'plot_graphical_model',
    'plot_partial_corrcoef'
]


def plot_anomaly_score(
    anomaly_score,          threshold=None,
    ax=None,                figsize=None,
    title=None,             xlim=None,
    ylim=None,              xlabel='Samples',
    ylabel='Anomaly score', grid=True,
    filepath=None,          **kwargs
):
    """Plot the anomaly score for each sample.

    Parameters
    ----------
    anomaly_score : array-like of shape (n_samples,)
        Anomaly score for each sample.

    threshold : float, default None
        Threshold.

    ax : matplotlib Axes, default None
        Target axes instance.

    figsize: tuple, default None
        Tuple denoting figure size of the plot.

    title : string, default None
        Axes title. To disable, pass None.

    xlim : tuple, default None
        Tuple passed to `ax.xlim`.

    ylim : tuple, default None
        Tuple passed to `ax.ylim`.

    xlabel : string, default 'Samples'
        X axis title label. To disable, pass None.

    ylabel : string, default 'Anomaly score'
        Y axis title label. To disable, pass None.

    grid : boolean, default True
        If True, turn the axes grids on.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_anomaly_score.png
    """

    import matplotlib.pyplot as plt

    n_samples, = anomaly_score.shape
    xlocs      = np.arange(n_samples)

    if ax is None:
        _, ax  = plt.subplots(figsize=figsize)

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
    y_true,                       y_score,
    ax=None,                      figsize=None,
    label=None,                   title=None,
    xlabel='False Positive Rate', ylabel='True Positive Rate',
    grid=True,                    filepath=None,
    **kwargs
):
    """Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True Labels.

    y_score : array-like of shape (n_samples,)
        Target scores.

    ax : matplotlib Axes, default None
        Target axes instance.

    figsize: tuple, default None
        Tuple denoting figure size of the plot.

    label : str, default None
        Legend label.

    title : string, default None
        Axes title. To disable, pass None.

    grid : boolean, default True
        If True, turn the axes grids on.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.
    """

    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    if ax is None:
        _, ax   = plt.subplots(figsize=figsize)

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
    partial_corrcoef,        ax=None,
    width=None,              figsize=None,
    title='Graphical model', filepath=None,
    **kwargs
):
    """Plot the Gaussian Graphical Model (GGM).

    Parameters
    ----------
    partial_corrcoef : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    ax : matplotlib Axes, default None
        Target axes instance.

    figsize: tuple, default None
        Tuple denoting figure size of the plot.

    width: float or array-like, default None
        Line width of edges.

    title : string, default 'Graphical model'
        Axes title. To disable, pass None.

    filepath : str, default None
        If provided, save the current figure.

    **kwargs : dict
        Other keywords passed to `nx.draw_networkx`.

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_graphical_model.png
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    tril      = np.tril(partial_corrcoef)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if width is None:
        width = np.abs(tril.flat[tril.flat[:] != 0.])

    if title is not None:
        ax.set_title(title)

    nx.draw_networkx(nx.from_numpy_matrix(tril), ax=ax, width=width, **kwargs)

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax


def plot_partial_corrcoef(
    partial_corrcoef, ax=None,
    figsize=None,     cmap=None,
    vmin=-1.,         vmax=1.,
    cbar=True,        title='Partial correlation',
    filepath=None,    **kwargs
):
    """Plot the partial correlation coefficient matrix.

    Parameters
    ----------
    partial_corrcoef : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    ax : matplotlib Axes, default None
        Target axes instance.

    figsize: tuple, default None
        Tuple denoting figure size of the plot.

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
        Other keywords passed to `ax.imshow`.

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_partial_corrcoef.png
    """

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if ax is None:
        _, ax     = plt.subplots(figsize=figsize)

    if cmap is None:
        cmap      = plt.cm.RdBu

    if title is not None:
        ax.set_title(title)

    mappable      = ax.imshow(
        np.ma.masked_equal(partial_corrcoef, 0.),
        cmap      = cmap,
        vmin      = vmin,
        vmax      = vmax,
        **kwargs
    )

    ax.set_facecolor('grey')

    if cbar:
        divider   = make_axes_locatable(ax)
        cax       = divider.append_axes('right', size='5%', pad=0.05)

        ax.figure.colorbar(mappable, cax=cax)

    if filepath is not None:
        ax.figure.savefig(filepath)

    return ax
