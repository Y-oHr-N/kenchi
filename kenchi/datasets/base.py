import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.utils import check_random_state, Bunch

__all__     = ['load_pendigits', 'load_pima', 'load_wdbc', 'load_wilt']

MODULE_PATH = os.path.dirname(__file__)
NEG_LABEL   = -1
POS_LABEL   = 1


def load_pendigits(random_state=None, return_X_y=False, subset='kriegel11'):
    """Load and return the pendigits dataset.

    Kriegel's structure (subset='kriegel11') :

    =============== =======
    anomalous class class 4
    n_samples       9868
    n_outliers      20
    n_features      16
    contamination   0.002
    =============== =======

    Goldstein's global structure (subset='goldstein12-global') :

    =============== =================================
    anomalous class classes 0, 1, 2, 3, 4, 5, 6, 7, 9
    n_samples       809
    n_outliers      90
    n_features      16
    contamination   0.111
    =============== =================================

    Goldstein's local structure (subset='goldstein12-local') :

    =============== =======
    anomalous class class 4
    n_samples       6724
    n_outliers      10
    n_features      16
    contamination   0.001
    =============== =======

    Parameters
    ----------
    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    return_X_y : bool, default False
        If True, return ``(data, target)`` instead of a Bunch object.

    subset : str, default 'kriegel11'
        Specify the structure. Valid options are
        ['goldstein12-global'|'goldstein12-local'|'kriegel11'].

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository," 2017.

    .. [#goldstein12] Goldstein, M., and Dengel, A.,
        "Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm,"
        KI: Poster and Demo Track, pp. 59-63, 2012.

    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.

    Examples
    --------
    >>> from kenchi.datasets import load_pendigits
    >>> pendigits = load_pendigits(subset='kriegel11')
    >>> pendigits.data.shape
    (9868, 16)
    >>> pendigits = load_pendigits(subset='goldstein12-global')
    >>> pendigits.data.shape
    (809, 16)
    >>> pendigits = load_pendigits(subset='goldstein12-local')
    >>> pendigits.data.shape
    (6724, 16)
    """

    filename_train           = os.path.join(
        MODULE_PATH, 'data', 'pendigits_train.csv.gz'
    )
    data_train               = np.loadtxt(filename_train, delimiter=',')
    X_train                  = data_train[:, :-1]
    y_train                  = data_train[:, -1]

    if subset not in ['goldstein12-global', 'goldstein12-local', 'kriegel11']:
        raise ValueError(f'invalid subset: {subset}')

    if subset == 'goldstein12-global':
        X                    = X_train
        y                    = y_train

        n_outliers_per_class = 10
        boolarr              = np.array([y == i for i in np.unique(y)])
        is_outlier           = boolarr[8]
        s                    = np.empty(0, dtype=int)

        for i, row in enumerate(boolarr):
            idx              = np.flatnonzero(row)

            if i == 8:
                s            = np.union1d(s, idx)
            else:
                s            = np.union1d(s, idx[:n_outliers_per_class])

    if subset == 'goldstein12-local':
        X                    = X_train
        y                    = y_train

        n_outliers           = 10
        is_outlier           = y == 4
        idx_inlier           = np.flatnonzero(~is_outlier)
        idx_outlier          = np.flatnonzero(is_outlier)

        s                    = np.union1d(idx_inlier, idx_outlier[:n_outliers])

    if subset == 'kriegel11':
        filename_test        = os.path.join(
            MODULE_PATH, 'data', 'pendigits_test.csv.gz'
        )
        data_test            = np.loadtxt(filename_test, delimiter=',')
        X_test               = data_test[:, :-1]
        y_test               = data_test[:, -1]

        X                    = np.concatenate([X_train, X_test])
        y                    = np.concatenate([y_train, y_test])

        n_outliers           = 20
        is_outlier           = y == 4
        idx_inlier           = np.flatnonzero(~is_outlier)
        idx_outlier          = np.flatnonzero(is_outlier)

        rnd                  = check_random_state(random_state)
        s                    = np.union1d(
            idx_inlier,
            rnd.choice(idx_outlier, size=n_outliers, replace=False)
        )

    y[~is_outlier]           = POS_LABEL
    y[is_outlier]            = NEG_LABEL

    # Downsample outliers
    X                        = X[s]
    y                        = y[s]

    y                        = y.astype(int)
    _, n_features            = X.shape
    feature_names            = np.arange(n_features)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, feature_names=feature_names)


def load_pima(return_X_y=False):
    """Load and return the Pima Indians diabetes dataset.

    =============== =======
    anomalous class class 1
    n_samples       768
    n_outliers      268
    n_features      8
    contamination   0.349
    =============== =======

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository," 2017.

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

    filename       = os.path.join(MODULE_PATH, 'data', 'pima.csv.gz')
    data           = np.loadtxt(filename, delimiter=',', skiprows=1)
    X              = data[:, :-1]
    y              = data[:, -1]

    is_outlier     = y == 1
    y[~is_outlier] = POS_LABEL
    y[is_outlier]  = NEG_LABEL

    y              = y.astype(int)
    feature_names  = np.array([
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    ])

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, feature_names=feature_names)


def load_wdbc(random_state=None, return_X_y=False, subset='kriegel11'):
    """Load and return the breast cancer Wisconsin dataset.

    Goldstein's structure (subset='goldstein12') :

    =============== =========
    anomalous class malignant
    n_samples       367
    n_outliers      10
    n_features      30
    contamination   0.027
    =============== =========

    Kriegel's structure (subset='kriegel11') :

    =============== =========
    anomalous class malignant
    n_samples       367
    n_outliers      10
    n_features      30
    contamination   0.027
    =============== =========

    Sugiyama's structure (subset='sugiyama13') :

    =============== =========
    anomalous class malignant
    n_samples       569
    n_outliers      212
    n_features      30
    contamination   0.373
    =============== =========

    Parameters
    ----------
    random_state : int, RandomState instance, default None
        Seed of the pseudo random number generator.

    return_X_y : bool, default False
        If True, return ``(data, target)`` instead of a Bunch object.

    subset : str, default 'kriegel11'
        Specify the structure. Valid options are
        ['goldstein12'|'kriegel11'|'sugiyama13'].

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository," 2017.

    .. [#goldstein12] Goldstein, M., and Dengel, A.,
        "Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm,"
        KI: Poster and Demo Track, pp. 59-63, 2012.

    .. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
        "Interpreting and unifying outlier scores,"
        In Proceedings of SDM, pp. 13-24, 2011.

    .. [#sugiyama13] Sugiyama, M., and Borgwardt, K.,
        "Rapid distance-based outlier detection via sampling,"
        Advances in NIPS, pp. 467-475, 2013.

    Examples
    --------
    >>> from kenchi.datasets import load_wdbc
    >>> wdbc = load_wdbc(subset='goldstein12')
    >>> wdbc.data.shape
    (367, 30)
    >>> wdbc = load_wdbc(subset='kriegel11')
    >>> wdbc.data.shape
    (367, 30)
    >>> wdbc = load_wdbc(subset='sugiyama13')
    >>> wdbc.data.shape
    (569, 30)
    """

    wdbc          = load_breast_cancer()
    X             = wdbc.data
    y             = wdbc.target
    feature_names = wdbc.feature_names

    n_outliers    = 10
    is_outlier    = y == 0
    idx_inlier    = np.flatnonzero(~is_outlier)
    idx_outlier   = np.flatnonzero(is_outlier)
    y[is_outlier] = NEG_LABEL

    if subset not in ['goldstein12', 'kriegel11', 'sugiyama13']:
        raise ValueError(f'invalid subset: {subset}')

    if subset == 'goldstein12':
        s         = np.union1d(idx_inlier, idx_outlier[:n_outliers])

    if subset == 'kriegel11':
        rnd       = check_random_state(random_state)
        s         = np.union1d(
            idx_inlier,
            rnd.choice(idx_outlier, size=n_outliers, replace=False)
        )

    if subset != 'sugiyama13':
        # Downsample outliers
        X         = X[s]
        y         = y[s]

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, feature_names=feature_names)


def load_wilt(return_X_y=False):
    """Load and return the wilt dataset.

    =============== =========
    anomalous class class 'w'
    n_samples       4839
    n_outliers      261
    n_features      5
    contamination   0.053
    =============== =========

    Parameters
    ----------
    return_X_y : bool, default False
        If True, return ``(data, target)`` instead of a Bunch object.

    Returns
    -------
    data : Bunch
        Dictionary-like object.

    References
    ----------
    .. [#dua17] Dua, D., and Karra Taniskidou, E.,
        "UCI Machine Learning Repository," 2017.

    .. [#goix16] Goix, N.,
        "How to evaluate the quality of unsupervised anomaly detection
        algorithms?"
        In ICML Anomaly Detection Workshop, 2016.

    Examples
    --------
    >>> from kenchi.datasets import load_wilt
    >>> wilt = load_wilt()
    >>> wilt.data.shape
    (4839, 5)
    """

    filename_train = os.path.join(MODULE_PATH, 'data', 'wilt_train.csv.gz')
    data_train     = np.loadtxt(
        filename_train, delimiter=',', dtype=object, skiprows=1
    )
    X_train        = data_train[:, 1:]
    y_train        = data_train[:, 0]

    filename_test  = os.path.join(MODULE_PATH, 'data', 'wilt_test.csv.gz')
    data_test      = np.loadtxt(
        filename_test, delimiter=',', dtype=object, skiprows=1
    )
    X_test         = data_test[:, 1:]
    y_test         = data_test[:, 0]

    X              = np.concatenate([X_train, X_test])
    y              = np.concatenate([y_train, y_test])

    is_outlier     = y == 'w'
    y[~is_outlier] = POS_LABEL
    y[is_outlier]  = NEG_LABEL

    X              = X.astype(float)
    y              = y.astype(int)
    feature_names  = np.array([
        'GLCM_pan', 'Mean_Green', 'Mean_Red', 'Mean_NIR',
        'SD_pan'
    ])

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y, feature_names=feature_names)
