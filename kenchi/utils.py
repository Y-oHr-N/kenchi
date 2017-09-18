from functools import wraps

import pandas as pd
from sklearn.utils.validation import check_array


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
        use_dataframe          = isinstance(X, pd.DataFrame)

        if use_dataframe:
            index              = X.index
            columns            = X.columns

        result                 = func(estimator, X, **kargs)

        if use_dataframe:
            is_change_detector = hasattr(estimator, 'shift') \
                and hasattr(estimator, 'window')

            if is_change_detector:
                index          = index[estimator.window - 1::estimator.shift]

            if result.ndim == 1:
                result         = pd.Series(
                    data       = result,
                    index      = index
                )

            else:
                result         = pd.DataFrame(
                    data       = result,
                    index      = index,
                    columns    = columns
                )

        return result

    return wrapper


def window_generator(X, window=1, shift=1):
    """Return the generator that yields windows from the given data.

    Parameters
    ----------
    X : array-like, shpae = (n_samples, n_features)
        Samples.

    window : integer
        Window size.

    shift : integer
        Shift size.

    Returns
    -------
    gen : generator
        Generator.
    """

    if window < shift:
        raise ValueError('window must be greater than or equal to shift.')

    X            = check_array(X)
    n_samples, _ = X.shape

    for i in range((n_samples - window + shift) // shift):
        yield X[i * shift:i * shift + window]
