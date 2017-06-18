kenchi
======

This is a set of python modules for anomaly detection.

Requirements
------------

-  numpy
-  scipy
-  scikit-learn

Installation
------------

You can install via pip.

::

    pip install kenchi

Usage
-----

.. code:: python

    >>> import numpy as np
    >>> from kenchi import GaussianDetector
    >>> train_size = 1000
    >>> test_size  = 100
    >>> n_outliers = 10
    >>> n_features = 10
    >>> rnd        = np.random.RamdomState(0)
    >>> X_train    = rnd.normal(size=(train_size, n_features))
    >>> X_test     = np.concatenate(
    ...     (
    ...         rnd.normal(size=(test_size - n_outliers, n_features)),
    ...         rnd.uniform(-10.0, 10.0, size=(n_outliers, n_features))
    ...     )
    ... )
    >>> det        = GaussianDetector(use_method_of_moments=True).fit(X_train)
    >>> det.predict(X_test)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

License
-------

`The MIT License <./LICENSE>`__
