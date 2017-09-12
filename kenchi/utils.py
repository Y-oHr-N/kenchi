from functools import wraps

import pandas as pd
from sklearn.utils.validation import check_array


def holdattr(func):
    """Return wrapper function that holds attributes of the DataFrame.
    """

    @wraps(func)
    def wrapper(self, X, **kargs):
        isdataframe         = isinstance(X, pd.DataFrame)

        if isdataframe:
            index           = X.index
            columns         = X.columns

        arr                 = func(self, X, **kargs)

        if isdataframe:
            if hasattr(self, 'shift') and hasattr(self, 'window'):
                index       = index[self.window - 1::self.shift]

            if arr.ndim == 1:
                arr         = pd.Series(
                    data    = arr,
                    index   = index
                )

            else:
                arr         = pd.DataFrame(
                    data    = arr,
                    index   = index,
                    columns = columns
                )

        return arr

    return wrapper


def window_generator(X, window=1, shift=1):
    """Generator that yields windows from given data.

    parameters
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
