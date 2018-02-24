import functools
import time

from sklearn.externals.joblib import logger

__all__ = ['timeit']


def timeit(func):
    """Decorator that measures the time spent for fitting.

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
        start_time          = time.time()
        result              = func(estimator, *args, **kwargs)
        estimator.fit_time_ = time.time() - start_time

        if getattr(estimator, 'verbose', False):
            print(f'elaplsed: {logger.short_format_time(estimator.fit_time_)}')

        return result

    return wrapper
