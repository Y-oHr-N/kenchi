import unittest

import matplotlib
from matplotlib.axes import Axes
import numpy as np
from sklearn.exceptions import NotFittedError

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import GMM, KDE, SparseStructureLearning

matplotlib.use('Agg')


class GMMTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(random_state=1)
        self.sut  = GMM()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), GMM)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_feature_wise_anomaly_score_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            self.sut.feature_wise_anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(self.sut.fit(self.X).plot_anomaly_score(), Axes)


class KDETest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(random_state=1)
        self.sut  = KDE()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), KDE)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_feature_wise_anomaly_score_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            self.sut.feature_wise_anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(self.sut.fit(self.X).plot_anomaly_score(), Axes)


class SparseStructureLearningTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(centers=1, random_state=1)
        self.sut  = SparseStructureLearning()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), SparseStructureLearning)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_feature_wise_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.feature_wise_anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(self.sut.fit(self.X).plot_anomaly_score(), Axes)

    def test_plot_partial_corrcoeff(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_partial_corrcoef(), Axes
        )
