import doctest
import unittest

import matplotlib
from kenchi import pipeline
from kenchi.datasets import make_blobs
from kenchi.outlier_detection import SparseStructureLearning
from matplotlib.axes import Axes
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')

import matplotlib.pyplot as plt


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(pipeline))

    return tests


class PipelineTest(unittest.TestCase):
    def setUp(self):
        _, self.ax     = plt.subplots()
        self.X, self.y = make_blobs(centers=1, random_state=0)
        self.sut       = pipeline.Pipeline([
            ('scaler', StandardScaler()),
            ('det', SparseStructureLearning(assume_centered=True))
        ])

    def tearDown(self):
        plt.close()

    def test_score_samples(self):
        self.sut.fit(self.X)

        score_samples = self.sut.score_samples(self.X)

        self.assertEqual(self.y.shape, score_samples.shape)

    def test_anomaly_score(self):
        self.sut.fit(self.X)

        anomaly_score = self.sut.anomaly_score(self.X)

        self.assertEqual(self.y.shape, anomaly_score.shape)

    def test_featurewise_anomaly_score(self):
        self.sut.fit(self.X)

        anomaly_score = self.sut.featurewise_anomaly_score(self.X)

        self.assertEqual(self.X.shape, anomaly_score.shape)

    def test_plot_anomaly_score(self):
        self.sut.fit(self.X)

        self.assertIsInstance(
            self.sut.plot_anomaly_score(self.X, ax=self.ax), Axes
        )

    def test_plot_roc_curve(self):
        self.sut.fit(self.X)

        self.assertIsInstance(
            self.sut.plot_roc_curve(self.X, self.y, ax=self.ax), Axes
        )

    def test_plot_graphical_model(self):
        self.sut.fit(self.X)
        self.assertIsInstance(self.sut.plot_graphical_model(ax=self.ax), Axes)

    def test_plot_partial_corrcoef(self):
        self.sut.fit(self.X)
        self.assertIsInstance(self.sut.plot_partial_corrcoef(ax=self.ax), Axes)

    def test_score_samples_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.score_samples)

    def test_anomaly_score_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.anomaly_score)

    def test_featurewise_anomaly_score_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.featurewise_anomaly_score, self.X
        )

    def test_plot_anomaly_score_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_anomaly_score)

    def test_plot_roc_curve_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.plot_roc_curve, self.X, self.y
        )

    def test_plot_graphical_model_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_graphical_model)

    def test_plot_partial_corrcoef_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_partial_corrcoef)
