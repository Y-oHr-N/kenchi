import functools
import time

__all__ = ['timeit']


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
