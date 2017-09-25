from functools import wraps

import pandas as pd


def assign_info_on_pandas_obj(func):
    """Return the wrapper function that assigns infomation on pandas objects
    to the attributes.

    Parameters
    ----------
    func : function
        Wrapped function.

    Returns
    -------
    wrapper : function
        Wrapper function.
    """

    @wraps(func)
    def wrapper(estimator, X, *args, **kargs):
        result                       = func(estimator, X, *args, **kargs)
        estimator._use_dataframe     = isinstance(X, pd.DataFrame)

        if estimator._use_dataframe:
            estimator._feature_names = X.columns

        return result

    return wrapper


def construct_pandas_obj(func):
    """Return the wrapper function that constructs a pandas object.

    Parameters
    ----------
    func : function
        Wrapped function.

    Returns
    -------
    wrapper : function
        Wrapper function.
    """

    @wraps(func)
    def wrapper(estimator, X, *args, **kargs):
        result                = func(estimator, X, *args, **kargs)

        if hasattr(estimator, '_use_dataframe') and estimator._use_dataframe:
            index             = X.index

            if result.ndim == 1:
                result        = pd.Series(
                    data      = result,
                    index     = index
                )

            else:
                _, n_features = result.shape

                if estimator._feature_names.size == n_features:
                    columns   = estimator._feature_names

                else:
                    columns   = None

                result        = pd.DataFrame(
                    data      = result,
                    index     = index,
                    columns   = columns
                )

        return result

    return wrapper
