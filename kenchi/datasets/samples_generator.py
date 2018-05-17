import numpy as np
from sklearn.datasets import make_blobs as _make_blobs
from sklearn.utils import check_random_state, shuffle as _shuffle

from .base import NEG_LABEL, POS_LABEL
from ..utils import check_contamination

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
    X : array-like of shape (n_samples, n_features)
        Generated data.

    y : array-like of shape (n_samples,)
        Return -1 for outliers and +1 for inliers.

    References
    ----------
    .. [#kriegel08] Kriegel, H.-P., Schubert, M., and Zimek, A.,
        "Angle-based outlier detection in high-dimensional data,"
        In Proceedings of SIGKDD, pp. 444-452, 2008.

    .. [#sugiyama13] Sugiyama, M., and Borgwardt, K.,
        "Rapid distance-based outlier detection via sampling,"
        Advances in NIPS, pp. 467-475, 2013.

    Examples
    --------
    >>> from kenchi.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=10, n_features=2, contamination=0.1)
    >>> X.shape
    (10, 2)
    >>> y.shape
    (10,)
    """

    check_contamination(contamination)

    rnd              = check_random_state(random_state)

    n_inliers        = int(np.round((1. - contamination) * n_samples))
    X_inlier, _      = _make_blobs(
        centers      = centers,
        center_box   = center_box,
        cluster_std  = cluster_std,
        n_features   = n_features,
        n_samples    = n_inliers,
        random_state = rnd,
        shuffle      = False
    )

    data_max         = np.max(X_inlier, axis=0)
    data_min         = np.min(X_inlier, axis=0)

    n_outliers       = n_samples - n_inliers
    X_outlier        = rnd.uniform(
        low          = np.minimum(center_box[0], data_min),
        high         = np.maximum(center_box[1], data_max),
        size         = (n_outliers, n_features)
    )

    X                = np.concatenate([X_inlier, X_outlier])
    y                = np.empty(n_samples, dtype=int)
    y[:n_inliers]    = POS_LABEL
    y[n_inliers:]    = NEG_LABEL

    if shuffle:
        X, y         = _shuffle(X, y, random_state=rnd)

    return X, y
