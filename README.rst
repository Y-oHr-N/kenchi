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

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/Y-oHr-N/kenchi/master

kenchi
======

This is a set of python modules for anomaly detection.

Dependencies
------------

-  Python (>=3.6)
-  `numpy <http://www.numpy.org/>`_ (>=1.14.0)
-  `scikit-learn <http://scikit-learn.org/>`_ (>=0.19.1)
-  `scipy <https://www.scipy.org/scipylib/>`_ (>=1.0.0)

Installation
------------

You can install via ``pip``

::

    pip install kenchi

or ``conda``.

::

    conda install -c y_ohr_n kenchi

Algorithms
----------

#. FastABOD [#kriegel08]_
#. OCSVM [#scholkopf01]_
#. MiniBatchKMeans
#. LOF [#breunig00]_
#. KNN [#angiulli02]_, [#ramaswamy00]_
#. OneTimeSampling [#sugiyama13]_
#. IForest [#liu08]_
#. PCA
#. GMM
#. HBOS [#goldstein12]_
#. KDE
#. SparseStructureLearning [#ide09]_

Examples
--------

.. code:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from kenchi.datasets import load_pima
    from kenchi.outlier_detection import *
    from kenchi.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    np.random.seed(0)

    scaler = StandardScaler()

    detectors = [
        FastABOD(novelty=True, n_jobs=-1), OCSVM(),
        MiniBatchKMeans(), LOF(novelty=True, n_jobs=-1),
        KNN(novelty=True, n_jobs=-1), IForest(n_jobs=-1),
        PCA(), KDE()
    ]

    # Load the Pima Indians diabetes dataset.
    X, y = load_pima(return_X_y=True)
    X_train, X_test, _, y_test = train_test_split(X, y)

    # Get the current Axes instance
    ax = plt.gca()

    for det in detectors:
        # Fit the model according to the given training data
        pipeline = make_pipeline(scaler, det).fit(X_train)

        # Plot the Receiver Operating Characteristic (ROC) curve
        pipeline.plot_roc_curve(X_test, y_test, ax=ax)

    # Display the figure
    plt.show()

.. figure:: https://raw.githubusercontent.com/Y-oHr-N/kenchi/master/docs/images/readme.png
    :align: center

License
-------

BSD 3-Clause "New" or "Revised" License

Copyright (c) 2018, Kon

References
----------

.. [#angiulli02] Angiulli, F., and Pizzuti, C.,
    `"Fast outlier detection in high dimensional spaces," <https://doi.org/10.1007/3-540-45681-3_2>`_
    In Proceedings of PKDD, pp. 15-27, 2002.

.. [#breunig00] Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J.,
    `"LOF: identifying density-based local outliers," <https://doi.org/10.1145/335191.335388>`_
    In Proceedings of SIGMOD, pp. 93-104, 2000.

.. [#dua17] Dua, D., and Karra Taniskidou, E.,
    `"UCI Machine Learning Repository," <https://archive.ics.uci.edu/ml>`_
    2017.

.. [#goix16] Goix, N.,
    `"How to evaluate the quality of unsupervised anomaly detection algorithms?" <https://arxiv.org/abs/1607.01152>`_
    In ICML Anomaly Detection Workshop, 2016.

.. [#goldstein12] Goldstein, M., and Dengel, A.,
    "Histogram-based outlier score (HBOS): A fast unsupervised anomaly detection algorithm,"
    KI: Poster and Demo Track, pp. 59-63, 2012.

.. [#ide09] Ide, T., Lozano, C., Abe, N., and Liu, Y.,
    `"Proximity-based anomaly detection using sparse structure learning," <https://doi.org/10.1137/1.9781611972795.9>`_
    In Proceedings of SDM, pp. 97-108, 2009.

.. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert, E., and Zimek, A.,
    `"Interpreting and unifying outlier scores," <https://doi.org/10.1137/1.9781611972818.2>`_
    In Proceedings of SDM, pp. 13-24, 2011.

.. [#kriegel08] Kriegel, H.-P., Schubert, M., and Zimek, A.,
    `"Angle-based outlier detection in high-dimensional data," <https://doi.org/10.1145/1401890.1401946>`_
    In Proceedings of SIGKDD, pp. 444-452, 2008.

.. [#lee03] Lee, W. S, and Liu, B.,
    "Learning with positive and unlabeled examples using weighted Logistic Regression,"
    In Proceedings of ICML, pp. 448-455, 2003.

.. [#liu08] Liu, F. T., Ting, K. M., and Zhou, Z.-H.,
    `"Isolation forest," <https://doi.org/10.1145/2133360.2133363>`_
    In Proceedings of ICDM, pp. 413-422, 2008.

.. [#ramaswamy00] Ramaswamy, S., Rastogi, R., and Shim, K.,
    `"Efficient algorithms for mining outliers from large data sets," <https://doi.org/10.1145/335191.335437>`_
    In Proceedings of SIGMOD, pp. 427-438, 2000.

.. [#scholkopf01] Scholkopf, B., Platt, J. C., Shawe-Taylor, J. C., Smola, A. J., and Williamson, R. C.,
    `"Estimating the Support of a High-Dimensional Distribution," <https://doi.org/10.1162/089976601750264965>`_
    Neural Computation, 13(7), pp. 1443-1471, 2001.

.. [#sugiyama13] Sugiyama, M., and Borgwardt, K.,
    "Rapid distance-based outlier detection via sampling,"
    Advances in NIPS, pp. 467-475, 2013.
