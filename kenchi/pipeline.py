from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method

from .utils import assign_info_on_pandas_obj, construct_pandas_obj


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
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    # TODO: Implement plot_roc_curve method

    @if_delegate_has_method(delegate='_final_estimator')
    @assign_info_on_pandas_obj
    def fit(self, X, y=None, **fit_params):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples,), default None
            Targets.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        self : pipeline
            Return self.
        """

        return super().fit(X, y, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    def predict(self, X):
        """Apply transforms to the data, and predict with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
        """

        return super().predict(X)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    @assign_info_on_pandas_obj
    def fit_predict(self, X, y=None, **fit_params):
        """Apply fit_predict of last step in pipeline after transforms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        y : array-like of shape (n_samples,), default None
            Targets.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
        """

        return super().fit_predict(X, y, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    def anomaly_score(self, X):
        """Apply transforms, and compute anomaly scores for test samples with
        the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples,)
            Anomaly scores for test samples.
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.anomaly_score(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    def detect(self, X, threshold=None):
        """Apply transforms, and detect if a particular sample is an outlier or
        not.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        threshold : float, default None
            User-provided threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        Xt             = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt     = transform.transform(Xt)

        if threshold is None:
            threshold  = self._final_estimator.threshold_

        return self._final_estimator.detect(Xt, threshold)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    @assign_info_on_pandas_obj
    def fit_detect(self, X, y=None, **fit_params):
        """Applies fit_detect of last step in pipeline after transforms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Return 0 for inliers and 1 for outliers.
        """

        Xt, fit_params = self._fit(X, **fit_params)

        return self._final_estimator.fit_detect(Xt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    def feature_wise_anomaly_score(self, X):
        """Apply transforms, and compute feature-wise anomaly scores for test
        samples with the final estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_score : array-like of shape (n_samples, n_features)
            Feature-wise anomaly scores for test samples.
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.feature_wise_anomaly_score(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    def analyze(self, X, feature_wise_threshold=None):
        """Apply transforms, and analyze which features contribute to anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        feature_wise_threshold : ndarray of shape (n_features,), default None
            User-provided feature-wise threshold.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_features)
        """

        Xt                         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt                 = transform.transform(Xt)

        if feature_wise_threshold is None:
            feature_wise_threshold = \
                self._final_estimator.feature_wise_threshold_

        return self._final_estimator.analyze(Xt, feature_wise_threshold)

    @if_delegate_has_method(delegate='_final_estimator')
    @construct_pandas_obj
    @assign_info_on_pandas_obj
    def fit_analyze(self, X, y=None, **fit_params):
        """Appliy fit_analyze of last step in pipeline after transforms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.

        Returns
        -------
        y_pred : array-like of shape (n_samples, n_features)
        """

        Xt, fit_params = self._fit(X, **fit_params)

        return self._final_estimator.fit_analyze(Xt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def plot_anomaly_score(
        self,             X,
        ax=None,          grid=True,
        xlim=None,        ylim=None,
        xlabel='Samples', ylabel='Anomaly score',
        title=None,       **kwargs
    ):
        """Apply transoforms, and plot anomaly scores for test samples.

        Parameters
        ----------
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

        xlabel : str, default 'Samples'
            X axis title label. To disable, pass None.

        ylabel : str, default 'Anomaly score'
            Y axis title label. To disable, pass None.

        title : str, default None
            Axes title. To disable, pass None.

        **kwargs : dict
            Other keywords passed to ax.bar().

        Returns
        -------
        ax : matplotlib Axes
        """

        Xt         = X

        for _, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        return self._final_estimator.plot_anomaly_score(
            Xt, ax, grid, xlim, ylim, xlabel, ylabel, title, **kwargs
        )
