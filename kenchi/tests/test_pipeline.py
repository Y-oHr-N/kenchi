import doctest
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from kenchi import pipeline
from kenchi.outlier_detection import SparseStructureLearning
from kenchi.tests.common_tests import OutlierDetectorTestMixin
from matplotlib.axes import Axes
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(pipeline))

    return tests


class PipelineTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = pipeline.Pipeline([
            ('scaler', StandardScaler()),
            ('det', SparseStructureLearning(assume_centered=True))
        ])

    def tearDown(self):
        plt.close()

    def test_featurewise_anomaly_score(self):
        self.sut.fit(self.X_train)

        anomaly_score = self.sut.featurewise_anomaly_score(self.X_test)

        self.assertEqual(self.X_test.shape, anomaly_score.shape)

    def test_plot_graphical_model(self):
        self.sut.fit(self.X_train)

        ax = self.sut.plot_graphical_model()

        self.assertIsInstance(ax, Axes)

    def test_plot_partial_corrcoef(self):
        self.sut.fit(self.X_train)

        ax = self.sut.plot_partial_corrcoef()

        self.assertIsInstance(ax, Axes)

    def test_featurewise_anomaly_score_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.featurewise_anomaly_score, self.X_test
        )

    def test_plot_graphical_model_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_graphical_model)

    def test_plot_partial_corrcoef_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_partial_corrcoef)
