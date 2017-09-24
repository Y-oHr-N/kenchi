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

.. image:: https://readthedocs.org/projects/kenchi/badge/?version=latest
    :target: http://kenchi.readthedocs.io/en/latest/?badge=latest

kenchi
======

This is a set of python modules for anomaly detection.

Requirements
------------

-  Python (>=3.5)
-  matplotlib (>=2.0.2)
-  numpy (>=1.11.2)
-  pandas (>=0.20.3)
-  scipy (>=0.18.1)
-  scikit-learn (>=0.18.0)

Installation
------------

You can install via pip.

::

    pip install kenchi

Usage
-----

.. code:: python

    >>> import numpy as np
    >>> from kenchi.outlier_detection import GaussianOutlierDetector
    >>> train_size                = 1000
    >>> test_size                 = 100
    >>> n_outliers                = 10
    >>> n_features                = 10
    >>> rnd                       = np.random.RandomState(0)
    >>> mean                      = np.zeros(n_features)
    >>> cov                       = np.eye(n_features)
    >>> # Generate the training data
    >>> X_train                   = rnd.multivariate_normal(
    ...     mean                  = mean,
    ...     cov                   = cov,
    ...     size                  = train_size
    ... )
    >>> # Generate the test data that contains outliers
    >>> X_test                    = np.concatenate((
    ...     rnd.multivariate_normal(
    ...         mean              = mean,
    ...         cov               = cov,
    ...         size              = test_size - n_outliers
    ...     ),
    ...     rnd.uniform(-10.0, 10.0, size=(n_outliers, n_features))
    ... ))
    >>> # Fit the model according to the given training data
    >>> det                       = GaussianOutlierDetector(
    ...     use_method_of_moments = True
    ... ).fit(X_train)
    >>> # Compute anomaly scores for test samples
    >>> det.anomaly_score(X_test)
    array([  10.2279816 ,    8.42753754,   18.84554722,    6.08748561,
              4.90015449,    5.89225341,   11.21254526,   12.13837514,
              9.86375701,   13.45280559,    8.78519218,   11.17455207,
              8.67487908,   10.11605552,   10.82527373,    3.78523018,
             14.78783263,   16.41395906,    8.34334961,   20.50342821,
              4.22005031,    8.7572077 ,   10.05161427,   12.27857842,
             19.92948184,   10.27584865,    8.43020974,    6.30639236,
             11.1476077 ,    9.99646324,   12.28092523,   13.34627085,
             22.89043576,    3.99504068,    7.73977065,    7.79284252,
             13.55122948,    5.97303153,    8.52601583,    8.15659127,
              3.52298486,    8.93405888,   14.94932098,    6.4061224 ,
             12.66524369,    8.99643725,   13.75216927,    7.45615584,
              3.45030169,   11.24393062,    7.20534393,   18.25601135,
             13.1444944 ,   15.10554285,    9.29049118,    5.33213834,
             17.6554761 ,    2.21379148,    5.93073279,    9.93749574,
             11.21154344,   17.91035606,   12.82094055,    5.90893524,
             14.74798855,    6.2915951 ,    8.52391203,   16.21922731,
             16.41482967,    7.33060691,   16.42103029,   16.46678505,
             24.36289356,   12.37283908,    6.00125687,   15.69825891,
             22.0956111 ,    4.8476199 ,   11.68724234,    4.10000542,
              6.96881479,   12.35809065,   16.8156008 ,    7.91235384,
              6.58459794,   15.74858497,    3.47960267,    8.2400362 ,
              8.07578519,   12.90307535,  312.98229821,  317.97220149,
            355.38275459,  319.76722516,  324.12299603,  501.27083702,
            223.06641889,  332.14889184,  327.98300878,  295.4773678 ])
    >>> # Detect if a particular sample is an outlier or not
    >>> det.detect(X_test)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1], dtype=int32)

License
-------

The MIT License (MIT)

Copyright (c) 2017 Kon
