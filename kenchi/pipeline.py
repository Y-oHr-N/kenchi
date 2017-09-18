from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import if_delegate_has_method


class ExtendedPipeline(Pipeline):
    """Pipeline of transforms with a final estimator.

    Parameters
    ----------
    steps : list

    memory : instance of joblib.Memory or string

    Attributes
    ----------
    named_steps : dict
    """

    @if_delegate_has_method(delegate='_final_estimator')
    def anomaly_score(self, X):
        """Apply transforms, and compute anomaly scores with the final estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        scores : array-like
            Anomaly scores for test samples.
        """

        for _, transform in self.steps[:-1]:
            if transform is not None:
                X = transform.transform(X)

        return self._final_estimator.anomaly_score(X)
