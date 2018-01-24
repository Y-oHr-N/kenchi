import unittest

import matplotlib
import matplotlib.axes
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import GMM, KDE, SparseStructureLearning

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class GMMTest(unittest.TestCase):
    def setUp(self):
        self.X_train, _          = make_blobs(random_state=1)
        self.X_test, self.y_test = make_blobs(random_state=2)
        self.sut                 = GMM()
        _, self.ax               = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_check_estimator(self):
        self.assertIsNone(check_estimator(self.sut))

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X_train), GMM)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X_train), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X_train)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X_train)

    def test_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).score(self.X_test), float
        )

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X_train)

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


class KDETest(unittest.TestCase):
    def setUp(self):
        self.X_train, _          = make_blobs(random_state=1)
        self.X_test, self.y_test = make_blobs(random_state=2)
        self.sut                 = KDE()
        _, self.ax               = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_check_estimator(self):
        self.assertIsNone(check_estimator(self.sut))

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X_train), KDE)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X_train), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X_train)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X_train)

    def test_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).score(self.X_test), float
        )

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X_train)

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


class SparseStructureLearningTest(unittest.TestCase):
    def setUp(self):
        self.X_train, _          = make_blobs(centers=1, random_state=1)
        self.X_test, self.y_test = make_blobs(random_state=2)
        self.sut                 = SparseStructureLearning(
            glasso_params        = {'tol': 0.02}
        )
        _, self.ax               = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_check_estimator(self):
        self.assertIsNone(check_estimator(self.sut))

    def test_fit(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train), SparseStructureLearning
        )

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X_train), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X_train)

    def test_featurewise_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.featurewise_anomaly_score(self.X_train)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X_train)

    def test_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).score(self.X_test), float
        )

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X_train)

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

    def test_plot_graphical_model(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).plot_graphical_model(ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_partial_corrcoeff(self):
        self.assertIsInstance(
            self.sut.fit(self.X_train).plot_partial_corrcoef(ax=self.ax),
            matplotlib.axes.Axes
        )
