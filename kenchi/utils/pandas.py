from functools import wraps

import pandas as pd


def assign_info_on_pandas_obj(func):
    """Return the wrapper function that assigns infomation on pandas objects
    to the attributes.

    Parameters
    ----------
    func : callable
        Wrapped function.

    Returns
    -------
    wrapper : function
        Wrapper function.
    """

    @wraps(func)
    def wrapper(estimator, X, *args, **kargs):
        """Wrapper function.

        Parameters
        ----------
        estimator : estimator
            Estimator.

        X : array-like of shape (n_samples, n_features)
            Samples.

        *args : tuple

        **kwargs : dict

        Returns
        -------
        result
        """

        result                       = func(estimator, X, *args, **kargs)

        if isinstance(X, pd.DataFrame):
            estimator.feature_names_ = X.columns

        return result

    return wrapper


def construct_pandas_obj(func):
    """Return the wrapper function that constructs a pandas object.

    Parameters
    ----------
    func : callable
        Wrapped function.

    Returns
    -------
    wrapper : function
        Wrapper function.
    """

    @wraps(func)
    def wrapper(estimator, X, *args, **kargs):
        """Wrapper function.

        Parameters
        ----------
        estimator : estimator
            Estimator.

        X : array-like of shape (n_samples, n_features), default None
            Test samples.

        *args : tuple

        **kwargs : dict

        Returns
        -------
        result
        """

        result                = func(estimator, X, *args, **kargs)

        if hasattr(estimator, 'feature_names_'):
            if X is None:
                index         = None
            else:
                index         = X.index

            if result.ndim == 1:
                result        = pd.Series(
                    data      = result,
                    index     = index
                )
            else:
                _, n_features = result.shape

                if estimator.feature_names_.size == n_features:
                    columns   = estimator.feature_names_
                else:
                    columns   = None

                result        = pd.DataFrame(
                    data      = result,
                    index     = index,
                    columns   = columns
                )

        return result

    return wrapper
