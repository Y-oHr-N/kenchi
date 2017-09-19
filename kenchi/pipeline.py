from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method

from .utils import construct_pandas_object


class ExtendedPipeline(Pipeline):
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
    named_steps : dictionary
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def fit_predict(self, X, y=None, **fit_param):
        """Applies fit_predict of last step in pipeline after transforms.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        **fit_params : dictionary of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Labels for samples.
        """

        return super(ExtendedPipeline, self).fit_predict(X, y, **fit_param)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def anomaly_score(self, X):
        """Apply transforms, and compute anomaly scores with the final estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : array-like, shape = (n_samples,)
            Anomaly scores for test samples.
        """

        for _, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X)

        return self._final_estimator.anomaly_score(X)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def predict(self, X):
        """Apply transforms, and predict with the final estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Labels for test samples.
        """

        return super(ExtendedPipeline, self).predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    def plot_anomaly_score(
        self,             X,
        ax=None,          title=None,
        xlim=None,        ylim=None,
        xlabel='Samples', ylabel='Anomaly score',
        grid=True,        **kwargs
    ):
        """Apply transforms, and plot anomaly scores with the final estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        ax : matplotlib Axes, default None
            Target axes instance.

        title : string, default None
            Axes title. To disable, pass None.

        xlim : tuple, default None
            Tuple passed to axes.xlim().

        ylim : tuple, default None
            Tuple passed to axes.ylim().

        xlabel : string, default "Samples"
            X axis title label. To disable, pass None.

        ylabel : string, default "Anomaly score"
            Y axis title label. To disable, pass None.

        grid : boolean, default True
            If True, turn the axes grids on.

        **kwargs : dictionary
            Other keywords passed to ax.bar().

        Returns
        -------
        ax : matplotlib Axes
        """

        for _, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X)

        return self._final_estimator.plot_anomaly_score(
            X, ax, title, xlim, ylim, xlabel, ylabel, grid, **kwargs
        )
