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

Dependencies
------------

-  Python (>=3.6)
-  matplotlib (>=2.1.1)
-  networkx (>=2.0)
-  numpy (>=1.14.0)
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

Anomaly detection methods
-------------------------

#. FastABOD [#kriegel08]_
#. MiniBatchKMeans
#. KNN [#angiulli02]_, [#ramaswamy00]_
#. OneTimeSampling [#sugiyama13]_
#. LOF [#breunig00]_
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
    from kenchi.datasets import load_wdbc
    from kenchi.outlier_detection import *

    # Load the breast cancer wisconsin dataset
    X, y      = load_wdbc(random_state=0)

    f, ax     = plt.subplots()
    detectors = [
        FastABOD(),
        MiniBatchKMeans(random_state=0),
        LOF(),
        KNN(),
        IForest(random_state=0),
        PCA(),
        KDE()
    ]

    for det in detectors:
        # Fit the model, and plot the ROC curve
        det.fit(X).plot_roc_curve(X=None, y=y, ax=ax)

    plt.show()

.. image:: https://raw.githubusercontent.com/Y-oHr-N/kenchi/master/docs/images/plot_roc_curve.png

License
-------

BSD 3-Clause "New" or "Revised" License

Copyright (c) 2018, Kon

References
----------

.. [#angiulli02] Angiulli, F., and Pizzuti, C.,
    `"Fast outlier detection in high dimensional spaces," <https://doi.org/10.1007/3-540-45681-3_2>`_
    In Proceedings of PKDD'02, pp. 15-27, 2002.

.. [#breunig00] Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J.,
    `"LOF: identifying density-based local outliers," <https://doi.org/10.1145/335191.335388>`_
    In ACM sigmod record, pp. 93-104, 2000.

.. [#goldstein12] Goldstein, M., and Dengel, A.,
    "Histogram-based outlier score (HBOS): A fast unsupervised anomaly detection algorithm,"
    KI'12: Poster and Demo Track, pp. 59-63, 2012.

.. [#ide09] Ide, T., Lozano, C., Abe N., and Liu, Y.,
    `"Proximity-based anomaly detection using sparse structure learning," <https://doi.org/10.1137/1.9781611972795.9>`_
    In Proceedings of SDM'09, pp. 97-108, 2009.

.. [#kriegel11] Kriegel, H.-P., Kroger, P., Schubert E., and Zimek, A.,
    `"Interpreting and unifying outlier scores," <https://doi.org/10.1137/1.9781611972818.2>`_
    In Proceedings of SDM'11, pp. 13-24, 2011.

.. [#kriegel08] Kriegel, H.-P., Schubert M., and Zimek, A.,
    `"Angle-based outlier detection in high-dimensional data," <https://doi.org/10.1145/1401890.1401946>`_
    In Proceedings of SIGKDD'08, pp. 444-452, 2008.

.. [#liu08] Liu, F. T., Ting K. M., and Zhou, Z.-H.,
    `"Isolation forest," <https://doi.org/10.1145/2133360.2133363>`_
    In Proceedings of ICDM'08, pp. 413-422, 2008.

.. [#ramaswamy00] Ramaswamy, S., Rastogi R., and Shim, K.,
    `"Efficient algorithms for mining outliers from large data sets," <https://doi.org/10.1145/335191.335437>`_
    In Proceedings of SIGMOD'00, pp. 427-438, 2000.

.. [#sugiyama13] Sugiyama M., and Borgwardt, K.,
    "Rapid distance-based outlier detection via sampling,"
    Advances in NIPS'13, pp. 467-475, 2013.
