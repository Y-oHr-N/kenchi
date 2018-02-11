.. image:: https://travis-ci.org/Y-oHr-N/kenchi.svg?branch=master
    :target: https://travis-ci.org/Y-oHr-N/kenchi

.. image:: https://ci.appveyor.com/api/projects/status/5cjkl0jrxo7gmug0/branch/master?svg=true
    :target: https://ci.appveyor.com/project/Y-oHr-N/kenchi/branch/master

.. image:: https://coveralls.io/repos/github/Y-oHr-N/kenchi/badge.svg?branch=master
    :target: https://coveralls.io/github/Y-oHr-N/kenchi?branch=master

.. image:: https://codeclimate.com/github/Y-oHr-N/kenchi/badges/gpa.svg
    :target: https://codeclimate.com/github/Y-oHr-N/kenchi

.. image:: https://badge.fury.io/py/kenchi.svg
    :target: https://badge.fury.io/py/kenchi

.. image:: https://anaconda.org/Y_oHr_N/kenchi/badges/version.svg
    :target: https://anaconda.org/Y_oHr_N/kenchi

.. image:: https://readthedocs.org/projects/kenchi/badge/?version=latest
    :target: http://kenchi.readthedocs.io/en/latest/?badge=latest

kenchi
======

This is a set of python modules for anomaly detection.

Requirements
------------

-  Python (>=3.6)
-  matplotlib (>=2.1.1)
-  networkx (>=2.0)
-  numpy (>=1.14.0)
-  pandas (>=0.22.0)
-  scikit-learn (>=0.19.1)
-  scipy (>=1.0.0)

Installation
------------

You can install via ``pip``

::

    pip install kenchi

or ``conda``.

::

    conda install -c y_ohr_n kenchi

Usage
-----

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from kenchi.datasets import make_blobs
    from kenchi.outlier_detection import SparseStructureLearning

    train_size       = 1000
    test_size        = 250
    n_outliers       = 10
    n_features       = 25
    centers          = np.zeros((1, n_features))

    # Generate the training data
    X_train, y_train = make_blobs(
        n_inliers    = train_size,
        n_outliers   = 0,
        n_features   = n_features,
        centers      = centers,
        random_state = 1
    )

    # Generate the test data that contains outliers
    X_test, _        = make_blobs(
        n_inliers    = test_size - n_outliers,
        n_outliers   = n_outliers,
        n_features   = n_features,
        centers      = centers,
        random_state = 2,
        shuffle      = False
    )

    # Fit the model according to the given training data
    det              = SparseStructureLearning().fit(X_train)

    # Plot the anomaly score for each training sample
    det.plot_anomaly_score(X_test, linestyle='', marker='.')

    plt.show()

.. image:: https://raw.githubusercontent.com/Y-oHr-N/kenchi/master/docs/images/plot_anomaly_score.png

License
-------

The MIT License (MIT)

Copyright (c) 2017 Kon
