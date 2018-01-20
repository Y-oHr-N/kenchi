import functools
import time
from typing import Tuple, Union

import numpy as np

__all__     = [
    'timeit',
    'Limits',
    'RandomState',
    'OneDimArray',
    'TwoDimArray',
    'Axes',
    'Colormap'
]

Limits          = Tuple[float, float]
RandomState     = Union[int, np.random.RandomState]

try:
    import pandas as pd

    OneDimArray = Union[np.ndarray, pd.Series]
    TwoDimArray = Union[np.ndarray, pd.DataFrame]

except ImportError:
    OneDimArray = np.ndarray
    TwoDimArray = np.ndarray

try:
    import matplotlib.axes
    import matplotlib.colors

    Axes        = matplotlib.axes.Axes
    Colormap    = matplotlib.colors.Colormap

except ImportError:
    Axes        = object
    Colormap    = object


def short_format_time(t: float) -> str:
    if t > 60.:
        return f'{t / 60.:5.1f} min'
    else:
        return f'{t:5.1f} sec'


def timeit(func):
    """Decorator that measures the elapsed time.

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

        if getattr(estimator, 'verbose', False):
            print(f'elaplsed: {short_format_time(elapsed)}')

        return result

    return wrapper
