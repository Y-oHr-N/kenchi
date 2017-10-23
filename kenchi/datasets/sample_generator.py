import numpy as np
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state


def make_blobs_with_outliers(
    n_inliers=90,    n_outliers=10,
    n_features=2,    centers=3,
    cluster_std=1.0, center_box=(-10, 10),
    shuffle=True,    random_state=None
):
    """Generate isotropic Gaussian blobs with outliers.

    Parameters
    ----------
    n_inliers : int, default 90
        Number of inliers.

    n_outliers : int, default 10
        Number of outliers.

    n_features : int, default 2
        Number of features for each sample.

    centers : int or array-like of shape (n_centers, n_features), default 3
        Number of centers to generate, or the fixed center locations.

    cluster_std : float or array-like of shape (n_centers,), default 1.0
        Standard deviation of the clusters.

    center_box : pair of floats (min, max), default (-10.0, 10.0)
        Bounding box for each cluster center when centers are generated at
        random.

    shuffle : boolean, default True
        If True, shuffle samples.

    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator to use when shuffling the
        data.

    Returns
    -------
    X : ndarray of shape (n_inliers + n_outliers, n_features)
        Generated samples.

    y : ndarray of shape (n_inliers + n_outliers,)
        Return 0 for inliers and 1 for outliers.
    """

    generator        = check_random_state(random_state)

    X_inliers, _     = make_blobs(
        n_samples    = n_inliers,
        n_features   = n_features,
        centers      = centers,
        cluster_std  = cluster_std,
        shuffle      = False,
        random_state = generator
    )

    X_outliers       = generator.uniform(
        low          = center_box[0],
        high         = center_box[1],
        size         = (n_outliers, n_features)
    )

    X                = np.concatenate([X_inliers, X_outliers])

    y                = np.concatenate([
        np.zeros(n_inliers), np.ones(n_outliers)
    ])

    if shuffle:
        indices      = np.arange(n_inliers + n_outliers)

        generator.shuffle(indices)

        X            = X[indices]
        y            = y[indices]

    return X, y
