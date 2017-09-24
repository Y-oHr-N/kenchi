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
        result                   = func(estimator, X, *args, **kargs)
        estimator.use_dataframe_ = isinstance(X, pd.DataFrame)

        if estimator.use_dataframe_:
            estimator.columns_   = X.columns

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
        result        = func(estimator, X, *args, **kargs)

        if hasattr(estimator, 'use_dataframe_') and estimator.use_dataframe_:
            result    = pd.Series(
                data  = result,
                index = X.index
            )

        return result

    return wrapper
