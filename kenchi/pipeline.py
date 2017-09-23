from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method

from .utils import construct_pandas_object, plot_anomaly_score


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

    plot_anomaly_score = if_delegate_has_method(
        delegate       = '_final_estimator'
    )(plot_anomaly_score)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def anomaly_score(self, X, y=None):
        """Apply transforms, and compute anomaly scores for test samples with
        the final estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        scores : array-like, shape = (n_samples,)
            Anomaly scores for test samples.
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.anomaly_score(Xt, y)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def detect(self, X, y=None):
        """Apply transforms, and detect if a particular sample is an outlier or
        not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples,), default None
            Targets.

        Returns
        -------
        is_outlier : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.detect(Xt, y)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_object
    def fit_detect(self, X, y=None, **fit_params):
        """Applies fit_detect of last step in pipeline after transforms.

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
        is_outlier : array-like, shape = (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.fit_detect(Xt, y)
