import unittest

import matplotlib
import matplotlib.axes
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import SparseStructureLearning
from kenchi.pipeline import Pipeline

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(centers=1, random_state=1)
        self.sut       = Pipeline([
            ('standardize', StandardScaler()),
            ('detect',      SparseStructureLearning(assume_centered=True))
        ])
        _, self.ax     = plt.subplots()

    def tearDown(self):
        plt.close()

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_featurewise_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.featurewise_anomaly_score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_anomaly_score(self.X, ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_roc_curve(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_roc_curve(self.X, self.y, ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_graphical_model(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_graphical_model(ax=self.ax),
            matplotlib.axes.Axes
        )

    def test_plot_partial_corrcoef(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_partial_corrcoef(ax=self.ax),
            matplotlib.axes.Axes
        )
