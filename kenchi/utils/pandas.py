from functools import wraps

import pandas as pd


def construct_pandas_object(func):
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
    def wrapper(estimator, X, **kargs):
        result        = func(estimator, X, **kargs)

        if isinstance(X, pd.DataFrame):
            result    = pd.Series(
                data  = result,
                index = X.index
            )

        return result

    return wrapper
