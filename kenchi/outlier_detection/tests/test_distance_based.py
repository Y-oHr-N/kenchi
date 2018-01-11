import unittest

import matplotlib.axes
import numpy as np
from sklearn.exceptions import NotFittedError

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import KNN, OneTimeSampling


class KNNTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_outliers=0)
        self.sut  = KNN()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), KNN)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_anomaly_score(), matplotlib.axes.Axes
        )


class OneTimeSamplingTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_outliers=0)
        self.sut  = OneTimeSampling()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), OneTimeSampling)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_anomaly_score(), matplotlib.axes.Axes
        )
