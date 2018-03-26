import unittest

import matplotlib
import matplotlib.axes
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import PCA

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class PCATest(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(random_state=1)
        self.sut       = PCA()
        _, self.ax     = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_check_estimator(self):
        self.assertIsNone(check_estimator(self.sut))

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), PCA)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).score(self.X), float
        )

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_anomaly_score(ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_roc_curve(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_roc_curve(self.X, self.y, ax=self.ax),
            matplotlib.axes.Axes
        )
