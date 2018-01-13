import unittest

import matplotlib
from matplotlib.axes import Axes
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import SparseStructureLearning
from kenchi.pipeline import Pipeline

matplotlib.use('Agg')


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(centers=1, random_state=1)
        self.sut  = Pipeline([
            ('standardize', StandardScaler()),
            ('detect',      SparseStructureLearning(assume_centered=True))
        ])

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_feature_wise_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.feature_wise_anomaly_score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(self.sut.fit(self.X).plot_anomaly_score(), Axes)

    def test_plot_graphical_model(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_graphical_model(), Axes
        )

    def test_plot_partial_corrcoef(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_partial_corrcoef(), Axes
        )
