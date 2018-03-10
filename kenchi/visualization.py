import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import auc, roc_curve
from sklearn.utils.validation import check_array, check_symmetric, column_or_1d

__all__ = [
    'plot_anomaly_score',
    'plot_roc_curve',
    'plot_graphical_model',
    'plot_partial_corrcoef'
]


def plot_anomaly_score(
    anomaly_score, ax=None, bins='auto', figsize=None,
    filename=None, hist=True, kde=True, threshold=None,
    title=None, xlabel='Samples', xlim=None, ylabel='Anomaly score',
    ylim=None, **kwargs
):
    """Plot the anomaly score for each sample.

    Parameters
    ----------
    anomaly_score : array-like of shape (n_samples,)
        Anomaly score for each sample.

    ax : matplotlib Axes, default None
        Target axes instance.

    bins : int, str or array-like, default 'auto'
        Number of hist bins.

    figsize : tuple, default None
        Tuple denoting figure size of the plot.

    filename : str, default None
        If provided, save the current figure.

    hist : bool, default True
        If True, plot a histogram of anomaly scores.

    kde : bool, default True
        If True, plot a gaussian kernel density estimate.

    threshold : float, default None
        Threshold.

    title : string, default None
        Axes title. To disable, pass None.

    xlabel : string, default 'Samples'
        X axis title label. To disable, pass None.

    xlim : tuple, default None
        Tuple passed to `ax.xlim`.

    ylabel : string, default 'Anomaly score'
        Y axis title label. To disable, pass None.

    ylim : tuple, default None
        Tuple passed to `ax.ylim`.

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
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def _get_ax_hist(ax):
        locator          = ax.get_axes_locator()

        if locator is None:
            # Create an axes on the right side of ax
            divider      = make_axes_locatable(ax)
            ax_hist      = divider.append_axes(
                'right', '20%', pad=0.1, sharey=ax
            )

            return ax_hist

        for ax_hist in ax.get_figure().get_axes():
            locator_hist = ax_hist.get_axes_locator()

            if ax_hist == ax:
                continue

            if locator_hist is None:
                continue

            if locator_hist._axes_divider == locator._axes_divider:
                return ax_hist

    anomaly_score        = column_or_1d(anomaly_score)
    n_samples,           = anomaly_score.shape
    xlocs                = np.arange(n_samples)

    if ax is None:
        _, ax            = plt.subplots(figsize=figsize)

    ax.grid(True, linestyle=':')

    if xlim is None:
        xlim             = (0., n_samples - 1.)

    ax.set_xlim(xlim)

    if ylim is None:
        ylim             = (0., 1.05 * np.max(anomaly_score))

    ax.set_ylim(ylim)

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    line,                = ax.plot(xlocs, anomaly_score, **kwargs)
    color                = line.get_color()

    if threshold is not None:
        ax.hlines(threshold, xlim[0], xlim[1], color=color)

    if hist or kde:
        ax_hist          = _get_ax_hist(ax)

        ax_hist.grid(True, linestyle=':')
        ax_hist.tick_params(axis='y', labelleft=False)
        ax_hist.set_ylim(ylim)

    if hist:
        # Draw a histogram
        ax_hist.hist(
            anomaly_score,
            alpha        = 0.4,
            bins         = bins,
            color        = color,
            density      = True,
            orientation  = 'horizontal'
        )

    if kde:
        kernel           = gaussian_kde(anomaly_score)
        ylocs            = np.linspace(ylim[0], ylim[1])

        # Draw a gaussian kernel density estimate
        ax_hist.plot(kernel(ylocs), ylocs, color=color)

    if 'label' in kwargs:
        ax.legend(loc='upper left')

    if filename is not None:
        ax.get_figure().savefig(filename)

    return ax


def plot_roc_curve(
    y_true, y_score, ax=None, figsize=None,
    filename=None, title='ROC curve', xlabel='FPR', ylabel='TPR',
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

    figsize : tuple, default None
        Tuple denoting figure size of the plot.

    filename : str, default None
        If provided, save the current figure.

    title : string, default 'ROC curve'
        Axes title. To disable, pass None.

    xlabel : string, default 'FPR'
        X axis title label. To disable, pass None.

    ylabel : string, default 'TPR'
        Y axis title label. To disable, pass None.

    **kwargs : dict
        Other keywords passed to `ax.plot`.

    Returns
    -------
    ax : matplotlib Axes
        Axes on which the plot was drawn.

    Examples
    --------
    .. image:: images/plot_roc_curve.png
    """

    import matplotlib.pyplot as plt

    fpr, tpr, _          = roc_curve(y_true, y_score)
    roc_auc              = auc(fpr, tpr)

    if ax is None:
        _, ax            = plt.subplots(figsize=figsize)

    ax.grid(True, linestyle=':')
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.05)

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if 'label' in kwargs:
        kwargs['label'] += f' (area={roc_auc:1.3f})'
    else:
        kwargs['label']  = f'area={roc_auc:1.3f}'

    ax.plot(fpr, tpr, **kwargs)

    ax.legend(loc='lower right')

    if filename is not None:
        ax.get_figure().savefig(filename)

    return ax


def plot_graphical_model(
    G, ax=None, figsize=None, filename=None,
    random_state=None, title='GGM', **kwargs
):
    """Plot the Gaussian Graphical Model (GGM).

    Parameters
    ----------
    G : networkx Graph
        GGM.

    ax : matplotlib Axes, default None
        Target axes instance.

    figsize : tuple, default None
        Tuple denoting figure size of the plot.

    filename : str, default None
        If provided, save the current figure.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    title : string, default 'GGM'
        Axes title. To disable, pass None.

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

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title)

    node_size = np.array([30. * (d + 1.) for _, d in G.degree])
    pos       = nx.spring_layout(G, random_state=random_state)
    width     = np.abs([3. * w for _, _, w in G.edges(data='weight')])

    # Add the draw_networkx kwargs here
    kwargs.setdefault('cmap', 'Spectral')
    kwargs.setdefault('node_size', node_size)
    kwargs.setdefault('pos', pos)
    kwargs.setdefault('width', width)

    # Draw the Gaussian grapchical model
    nx.draw_networkx(G, ax=ax, **kwargs)

    # Turn off tick visibility
    ax.tick_params('x', labelbottom=False, bottom=False)
    ax.tick_params('y', labelleft=False, left=False)

    if filename is not None:
        ax.get_figure().savefig(filename)

    return ax


def plot_partial_corrcoef(
    partial_corrcoef, ax=None, cbar=True, figsize=None,
    filename=None, title='Partial correlation', **kwargs
):
    """Plot the partial correlation coefficient matrix.

    Parameters
    ----------
    partial_corrcoef : array-like of shape (n_features, n_features)
        Partial correlation coefficient matrix.

    ax : matplotlib Axes, default None
        Target axes instance.

    cbar : bool, default True.
        Whether to draw a colorbar.

    figsize : tuple, default None
        Tuple denoting figure size of the plot.

    filename : str, default None
        If provided, save the current figure.

    title : string, default 'Partial correlation'
        Axes title. To disable, pass None.

    **kwargs : dict
        Other keywords passed to `ax.pcolormesh`.

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

    partial_corrcoef = check_array(partial_corrcoef)
    partial_corrcoef = check_symmetric(partial_corrcoef, raise_exception=True)

    if ax is None:
        _, ax        = plt.subplots(figsize=figsize)

    if title is not None:
        ax.set_title(title)

    # Add the pcolormesh kwargs here
    kwargs.setdefault('cmap', 'RdBu')
    kwargs.setdefault('edgecolors', 'white')
    kwargs.setdefault('vmin', -1.)
    kwargs.setdefault('vmax', 1.)

    # Draw the heatmap
    mesh             = ax.pcolormesh(
        np.ma.masked_equal(partial_corrcoef, 0.), **kwargs
    )

    ax.set_aspect('equal')
    ax.set_facecolor('grey')

    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()

    if cbar:
        # Create an axes on the right side of ax
        divider      = make_axes_locatable(ax)
        cax          = divider.append_axes('right', '5%', pad=0.1)

        ax.get_figure().colorbar(mesh, cax=cax)

    if filename is not None:
        ax.get_figure().savefig(filename)

    return ax
