from typing import Union

import numpy as np
from sklearn.datasets import make_blobs as sklearn_make_blobs
from sklearn.utils import check_random_state

from .utils import Limits, OneDimArray, RandomState, TwoDimArray

__all__ = ['make_blobs']


def make_blobs(
    n_inliers:    int                       = 490,
    n_outliers:   int                       = 10,
    n_features:   int                       = 25,
    centers:      Union[int,   TwoDimArray] = 5,
    cluster_std:  Union[float, OneDimArray] = 1.,
    center_box:   Limits                    = (-10., 10.),
    shuffle:      bool                      = True,
    random_state: RandomState               = None
) -> Union[TwoDimArray, OneDimArray]:
    """Generate isotropic Gaussian blobs with outliers.

    Parameters
    ----------
    n_inliers : int, default 490
        Number of inliers.

    n_outliers : int, default 10
        Number of outliers.

    n_features : int, default 25
        Number of features for each sample.

    centers : int or array-like of shape (n_centers, n_features), default 5
        Number of centers to generate, or the fixed center locations.

    cluster_std : float or array-like of shape (n_centers,), default 1.0
        Standard deviation of the clusters.

    center_box : pair of floats (min, max), default (-10.0, 10.0)
        Bounding box for each cluster center when centers are generated at
        random.

    shuffle : boolean, default True
        If True, shuffle samples.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    Returns
    -------
    X : ndarray of shape (n_inliers + n_outliers, n_features)
        Generated data.

    y : ndarray of shape (n_inliers + n_outliers,)
        Return 1 for inliers and -1 for outliers.

    References
    ----------
    H.-P. Kriegel, M. Schubert and A. Zimek,
    "Angle-based outlier detection in high-dimensional data,"
    In Proceedings of SIGKDD'08, pp. 444-452, 2008.

    M. Sugiyama, and K. Borgwardt,
    "Rapid distance-based outlier detection via sampling,"
    Advances in NIPS'13, pp. 467-475, 2013.
    """

    rnd              = check_random_state(random_state)

    X_inliers, _     = sklearn_make_blobs(
        n_samples    = n_inliers,
        n_features   = n_features,
        centers      = centers,
        cluster_std  = cluster_std,
        shuffle      = False,
        random_state = rnd
    )

    X_outliers       = rnd.uniform(
        low          = np.min(X_inliers),
        high         = np.max(X_inliers),
        size         = (n_outliers, n_features)
    )

    X                = np.concatenate([X_inliers, X_outliers])
    y                = np.empty(n_inliers + n_outliers, dtype=np.int64)
    y[:n_inliers]    = 1
    y[n_inliers:]    = -1

    if shuffle:
        indices      = np.arange(n_inliers + n_outliers)

        rnd.shuffle(indices)

        X            = X[indices]
        y            = y[indices]

    return X, y
