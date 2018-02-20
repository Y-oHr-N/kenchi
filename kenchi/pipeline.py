from sklearn.pipeline import _name_estimators, Pipeline as SKLearnPipeline
from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['make_pipeline', 'Pipeline']


def make_pipeline(*steps):
    """Construct a Pipeline from the given estimators. This is a shorthand for
    the Pipeline constructor; it does not require, and does not permit, naming
    the estimators. Instead, their names will be set to the lowercase of their
    types automatically.

    Parameters
    ----------
    *steps : list
        List of estimators.

    Returns
    -------
    p : Pipeline
    """

    return Pipeline(_name_estimators(steps))


class Pipeline(SKLearnPipeline):
    """Pipeline of transforms with a final estimator.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : instance of joblib.Memory or string, default None
        Used to cache the fitted transformers of the pipeline. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the
        transformers before fitting. Therefore, the transformer instance given
        to the pipeline cannot be inspected directly. Use the attribute
        ``named_steps`` or ``steps`` to inspect estimators within the pipeline.
        Caching the transformers is advantageous when fitting is time
        consuming.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    @if_delegate_has_method(delegate='_final_estimator')
    def anomaly_score(self, X=None):
        """Apply transforms, and compute the anomaly score for each sample with
        the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, the anomaly score for each training sample is
            returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples,)
            Anomaly score for each sample.
        """

        if X is not None:
            for _, transform in self.steps[:-1]:
                if transform is not None:
                    X = transform.transform(X)

        return self._final_estimator.anomaly_score(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def featurewise_anomaly_score(self, X=None):
        """Apply transforms, and compute the feature-wise anomaly scores for
        each sample with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, the feature_wise anomaly scores for each training
            sample are returned.

        Returns
        -------
        anomaly_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for each sample.
        """

        if X is not None:
            for _, transform in self.steps[:-1]:
                if transform is not None:
                    X = transform.transform(X)

        return self._final_estimator.featurewise_anomaly_score(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def plot_anomaly_score(self, X=None, **kwargs):
        """Apply transoforms, and plot the anomaly score for each sample with
        the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default None
            Data. If None, plot the anomaly score for each training samples.

        ax : matplotlib Axes, default None
            Target axes instance.

        bins : int, str or array-like, default 'fd'
            Number of hist bins.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filepath : str, default None
            If provided, save the current figure.

        grid : boolean, default True
            If True, turn the axes grids on.

        hist : bool, default True
            If True, plot a histogram of anomaly scores.

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
        """

        if X is not None:
            for _, transform in self.steps[:-1]:
                if transform is not None:
                    X = transform.transform(X)

        return self._final_estimator.plot_anomaly_score(X, **kwargs)

    @if_delegate_has_method(delegate='_final_estimator')
    def plot_roc_curve(self, X, y, **kwargs):
        """Apply transoforms, and plot the Receiver Operating Characteristic
        (ROC) curve with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data.

        y : array-like of shape (n_samples,)
            Labels.

        ax : matplotlib Axes, default None
            Target axes instance.

        figsize: tuple, default None
            Tuple denoting figure size of the plot.

        filepath : str, default None
            If provided, save the current figure.

        grid : boolean, default True
            If True, turn the axes grids on.

        label : str, default None
            Legend label.

        title : string, default None
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
        """

        for _, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X)

        return self._final_estimator.plot_roc_curve(X, y, **kwargs)

    @property
    def plot_graphical_model(self):
        """Apply transforms, and plot the Gaussian Graphical Model (GGM) with
        the final estimator.

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        cmap : str or matplotlib Colormap, default 'Spectral'
            Colormap or Registered colormap name.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filepath : str, default None
            If provided, save the current figure.

        random_state : int, RandomState instance, default None
            Seed of the pseudo random number generator.

        title : string, default 'GGM (n_features=%d, n_clusters=%d)'
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to `nx.draw_networkx`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        return self._final_estimator.plot_graphical_model

    @property
    def plot_partial_corrcoef(self):
        """Apply transforms, and plot the partial correlation coefficient
        matrix with the final estimator.

        Parameters
        ----------
        ax : matplotlib Axes, default None
            Target axes instance.

        cbar : bool, default True.
            Whether to draw a colorbar.

        cmap : str or matplotlib Colormap, default 'RdBu'
            Colormap or Registered colormap name.

        figsize : tuple, default None
            Tuple denoting figure size of the plot.

        filepath : str, default None
            If provided, save the current figure.

        linecolor : str, default 'white'
            Color of the lines that will divide each cell.

        linewidths : float, default 0.5
            Width of the lines that will divide each cell.

        title : string, default 'Partial correlation'
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to `ax.pcolormesh`.

        Returns
        -------
        ax : matplotlib Axes
            Axes on which the plot was drawn.
        """

        return self._final_estimator.plot_partial_corrcoef
