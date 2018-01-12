import functools
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd

__all__     = ['timeit', 'OneDimArray', 'TwoDimArray', 'RandomState', 'Limits']

OneDimArray = Union[np.ndarray, pd.Series]
TwoDimArray = Union[np.ndarray, pd.DataFrame]
RandomState = Union[int, np.random.RandomState]
Limits      = Tuple[float, float]


def short_format_time(t: float) -> str:
    if t > 60.:
        return f'{t / 60.:5.1f} min'
    else:
        return f'{t:5.1f} sec'


def timeit(func):
    """Return the wrapper function that measures the elapsed time.

    Parameters
    ----------
    func : callable
        Wrapped function.

    Returns
    -------
    wrapper : callable
        Wrapper function.
    """

    @functools.wraps(func)
    def wrapper(estimator, *args, **kwargs):
        start   = time.time()
        result  = func(estimator, *args, **kwargs)
        elapsed = time.time() - start

        if estimator.verbose:
            print(f'elaplsed: {short_format_time(elapsed)}')

        return result

    return wrapper
