.. image:: https://img.shields.io/pypi/v/kenchi.svg
    :target: https://pypi.org/project/kenchi

.. image:: https://img.shields.io/pypi/pyversions/kenchi.svg
    :target: https://pypi.org/project/kenchi

.. image:: https://img.shields.io/pypi/l/HazureChi/kenchi.svg
    :target: https://github.com/HazureChi/kenchi/blob/master/LICENSE

.. image:: https://img.shields.io/conda/v/Y_oHr_N/kenchi.svg
    :target: https://anaconda.org/Y_oHr_N/kenchi

.. image:: https://img.shields.io/conda/pn/Y_oHr_N/kenchi.svg
    :target: https://anaconda.org/Y_oHr_N/kenchi

.. image:: https://img.shields.io/readthedocs/kenchi/stable.svg
    :target: http://kenchi.rtfd.io/en/stable

.. image:: https://img.shields.io/travis/HazureChi/kenchi/master.svg
    :target: https://travis-ci.org/HazureChi/kenchi

.. image:: https://img.shields.io/appveyor/ci/Y-oHr-N/kenchi/master.svg
    :target: https://ci.appveyor.com/project/Y-oHr-N/kenchi/branch/master

.. image:: https://img.shields.io/coveralls/github/HazureChi/kenchi/master.svg
    :target: https://coveralls.io/github/HazureChi/kenchi?branch=master

.. image:: https://img.shields.io/codeclimate/maintainability/HazureChi/kenchi.svg
    :target: https://codeclimate.com/github/HazureChi/kenchi

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/HazureChi/kenchi/master?urlpath=lab

kenchi
======

This is a scikit-learn compatible library for anomaly detection.

Dependencies
------------

- Required dependencies
    #. `numpy>=1.13.3 <http://www.numpy.org/>`_ (BSD 3-Clause License)
    #. `scikit-learn>=0.20.0 <http://scikit-learn.org/>`_ (BSD 3-Clause License)
    #. `scipy>=0.19.1 <https://www.scipy.org/scipylib/>`_ (BSD 3-Clause License)
- Optional dependencies
    #. `matplotlib>=2.1.2 <https://matplotlib.org/>`_ (PSF-based License)
    #. `networkx>=2.2 <https://networkx.github.io/>`_ (BSD 3-Clause License)

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

- Outlier detection
    #. FastABOD [#kriegel08]_
    #. LOF [#breunig00]_ (scikit-learn wrapper)
    #. KNN [#angiulli02]_, [#ramaswamy00]_
    #. OneTimeSampling [#sugiyama13]_
    #. HBOS [#goldstein12]_
- Novelty detection
    #. OCSVM [#scholkopf01]_ (scikit-learn wrapper)
    #. MiniBatchKMeans
    #. IForest [#liu08]_ (scikit-learn wrapper)
    #. PCA
    #. GMM (scikit-learn wrapper)
    #. KDE [#parzen62]_ (scikit-learn wrapper)
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

.. figure:: https://raw.githubusercontent.com/HazureChi/kenchi/master/docs/images/readme.png
    :align: center

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

.. [#parzen62] Parzen, E.,
    `"On estimation of a probability density function and mode," <https://doi.org/10.1214/aoms/1177704472>`_
    Ann. Math. Statist., 33(3), pp. 1065-1076, 1962.

.. [#ramaswamy00] Ramaswamy, S., Rastogi, R., and Shim, K.,
    `"Efficient algorithms for mining outliers from large data sets," <https://doi.org/10.1145/335191.335437>`_
    In Proceedings of SIGMOD, pp. 427-438, 2000.

.. [#scholkopf01] Scholkopf, B., Platt, J. C., Shawe-Taylor, J. C., Smola, A. J., and Williamson, R. C.,
    `"Estimating the Support of a High-Dimensional Distribution," <https://doi.org/10.1162/089976601750264965>`_
    Neural Computation, 13(7), pp. 1443-1471, 2001.

.. [#sugiyama13] Sugiyama, M., and Borgwardt, K.,
    `"Rapid distance-based outlier detection via sampling," <http://papers.nips.cc/paper/5127-rapid-distance-based-outlier-detection-via-sampling>`_
    Advances in NIPS, pp. 467-475, 2013.
