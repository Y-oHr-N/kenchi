import numpy as np
from sklearn.datasets import make_blobs as _make_blobs
from sklearn.utils import check_random_state, shuffle as _shuffle

__all__ = ['make_blobs']


def make_blobs(
    centers=5, center_box=(-10., 10.), cluster_std=1., contamination=0.02,
    n_features=25, n_samples=500, random_state=None, shuffle=True
):
    """Generate isotropic Gaussian blobs with outliers.

    Parameters
    ----------
    centers : int or array-like of shape (n_centers, n_features), default 5
        Number of centers to generate, or the fixed center locations.

    center_box : pair of floats (min, max), default (-10.0, 10.0)
        Bounding box for each cluster center when centers are generated at
        random.

    cluster_std : float or array-like of shape (n_centers,), default 1.0
        Standard deviation of the clusters.

    contamination : float, default 0.02
        Proportion of outliers in the data set.

    n_features : int, default 25
        Number of features for each sample.

    n_samples : int, default 500
        Number of samples.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    shuffle : bool, default True
        If True, shuffle samples.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated data.

    y : ndarray of shape (n_samples,)
        Return -1 for outliers and +1 for inliers.

    References
    ----------
    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM'11, pp. 13-24, 2011.
    """

    rnd              = check_random_state(random_state)

    n_inliers        = int(np.round((1. - contamination) * n_samples))
    X_inliers, _     = _make_blobs(
        centers      = centers,
        center_box   = center_box,
        cluster_std  = cluster_std,
        n_features   = n_features,
        n_samples    = n_inliers,
        random_state = rnd,
        shuffle      = False
    )

    n_outliers       = n_samples - n_inliers
    X_outliers       = rnd.uniform(
        low          = np.min(X_inliers, axis=0),
        high         = np.max(X_inliers, axis=0),
        size         = (n_outliers, n_features)
    )

    X                = np.concatenate([X_outliers, X_inliers])
    y                = np.ones(n_samples, dtype=np.int)
    y[:n_outliers]   = -1

    if shuffle:
        X, y         = _shuffle(X, y, random_state=rnd)

    return X, y
