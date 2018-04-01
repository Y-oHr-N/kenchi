import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import check_random_state, shuffle as _shuffle

__all__ = ['load_wdbc', 'load_pendigits']


def load_wdbc(contamination=0.0272, random_state=None, shuffle=True):
    """Load and return the breast cancer wisconsin dataset.

    contamination : float, default 0.0272
        Proportion of outliers in the data set.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    shuffle : bool, default True
        If True, shuffle samples.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Data.

    y : ndarray of shape (n_samples,)
        Return -1 (malignant) for outliers and +1 (benign) for inliers.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM'11, pp. 13-24, 2011.
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

    return X, y


def load_pendigits(contamination=0.002, random_state=None, shuffle=True):
    """Load and return the pendigits dataset.

    contamination : float, default 0.002
        Proportion of outliers in the data set.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    shuffle : bool, default True
        If True, shuffle samples.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Data.

    y : ndarray of shape (n_samples,)
        Return -1 (digit 4) for outliers and +1 (otherwise) for inliers.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM'11, pp. 13-24, 2011.
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

    return X, y
