import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import check_random_state, shuffle as _shuffle, Bunch

__all__ = ['load_pendigits', 'load_pima', 'load_wdbc']


def load_pima(return_X_y=False):
    """Load and return the Pima Indians diabetes dataset.

    ============= =======
    anomaly class class 1
    n_samples     768
    n_outliers    268
    n_features    8
    contamination 0.349
    ============= =======

    Parameters
    ----------
    return_X_y : bool, False
        If True, return `(data, target)` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository,"
        2017.

    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?"
        In ICML Anomaly Detection Workshop, 2016.

    .. [#liu08] Liu, F. T., Ting, K. M., and Zhou, Z.-H.,
        "Isolation forest,"
        In Proceedings of ICDM, pp. 413-422, 2008.

    .. [#sugiyama13] Sugiyama, M., and Borgwardt, K.,
        "Rapid distance-based outlier detection via sampling,"
        Advances in NIPS, pp. 467-475, 2013.

    Examples
    --------
    >>> from kenchi.datasets import load_pima
    >>> pima = load_pima()
    >>> pima.data.shape
    (768, 8)
    """

    module_path    = os.path.dirname(__file__)
    filename       = os.path.join(module_path, 'data', 'pima.csv.gz')

    data           = np.loadtxt(filename, delimiter=',', skiprows=1)
    X              = data[:, :-1]
    y              = data[:, -1].astype(int)

    is_outlier     = y == 1
    y[~is_outlier] = 1
    y[is_outlier]  = -1

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y)


def load_wdbc(return_X_y=False, contamination=0.0272, random_state=None, shuffle=True):
    """Load and return the breast cancer wisconsin dataset.

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return `(data, target)` instead of a Bunch object.

    contamination : float, default 0.0272
        Proportion of outliers in the data set.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    shuffle : bool, default True
        If True, shuffle samples.

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository,"
        2017.

    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.
    """

    rnd                    = check_random_state(random_state)
    X, y                   = load_breast_cancer(return_X_y=True)

    is_inlier              = y != 0
    n_inliers              = np.sum(is_inlier)
    X_inliers              = X[is_inlier]
    y_inliers              = y[is_inlier]

    n_outliers             = int(
        np.round(contamination / (1. - contamination) * n_inliers)
    )
    X_outliers             = X[~is_inlier]
    y_outliers             = y[~is_inlier]
    X_outliers, y_outliers = _shuffle(
        X_outliers, y_outliers, n_samples=n_outliers, random_state=rnd
    )
    y_outliers[:]          = -1

    X                      = np.concatenate([X_outliers, X_inliers])
    y                      = np.concatenate([y_outliers, y_inliers])

    if shuffle:
        X, y               = _shuffle(X, y, random_state=rnd)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y)


def load_pendigits(return_X_y=False, contamination=0.002, random_state=None, shuffle=True):
    """Load and return the pendigits dataset.

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return `(data, target)` instead of a Bunch object.

    contamination : float, default 0.002
        Proportion of outliers in the data set.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    shuffle : bool, default True
        If True, shuffle samples.

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository,"
        2017.

    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.
    """

    rnd                    = check_random_state(random_state)
    module_path            = os.path.dirname(__file__)
    data                   = np.loadtxt(
        os.path.join(module_path, 'data', 'pendigits.csv.gz'), delimiter=','
    )
    X                      = data[:, :-1]
    y                      = data[:, -1].astype(np.int)

    is_inlier              = y != 4
    n_inliers              = np.sum(is_inlier)
    X_inliers              = X[is_inlier]
    y_inliers              = y[is_inlier]
    y_inliers[:]           = 1

    n_outliers             = int(
        np.round(contamination / (1. - contamination) * n_inliers)
    )
    X_outliers             = X[~is_inlier]
    y_outliers             = y[~is_inlier]
    X_outliers, y_outliers = _shuffle(
        X_outliers, y_outliers, n_samples=n_outliers, random_state=rnd
    )
    y_outliers[:]          = -1

    X                      = np.concatenate([X_outliers, X_inliers])
    y                      = np.concatenate([y_outliers, y_inliers])

    if shuffle:
        X, y               = _shuffle(X, y, random_state=rnd)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y)
