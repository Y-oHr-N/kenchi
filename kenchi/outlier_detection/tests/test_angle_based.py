import unittest

import matplotlib
import matplotlib.axes
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import FastABOD

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class FastABODTest(unittest.TestCase):
    def setUp(self):
        self.X_train, _          = make_blobs(random_state=1)
        self.X_test, self.y_test = make_blobs(random_state=2)
        self.sut                 = FastABOD()
        _, self.ax               = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_check_estimator(self):
        with self.assertRaises(FloatingPointError):
            check_estimator(self.sut)

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X_train), FastABOD)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X_train), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X_train)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X_train)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).plot_anomaly_score(ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_roc_curve(self):
        self.assertIsInstance(
            self.sut.fit(
                self.X_train
            ).plot_roc_curve(self.X_test, self.y_test, ax=self.ax),
            matplotlib.axes.Axes
        )
