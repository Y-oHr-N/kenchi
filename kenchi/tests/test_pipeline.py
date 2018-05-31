import doctest
import unittest

from kenchi import pipeline
from kenchi.outlier_detection import SparseStructureLearning
from kenchi.tests.common_tests import OutlierDetectorTestMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.testing import if_matplotlib


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

    def test_featurewise_anomaly_score(self):
        self.sut.fit(self.X_train)

        anomaly_score = self.sut.featurewise_anomaly_score(self.X_test)

        self.assertEqual(anomaly_score.shape, self.X_test.shape)

    @if_matplotlib
    def test_plot_graphical_model(self):
        import matplotlib.pyplot as plt

        self.sut.fit(self.X_train)

        ax = self.sut.plot_graphical_model()

        plt.close('all')

        self.assertTrue(ax.has_data())

    @if_matplotlib
    def test_plot_partial_corrcoef(self):
        import matplotlib.pyplot as plt

        self.sut.fit(self.X_train)

        ax = self.sut.plot_partial_corrcoef()

        plt.close('all')

        self.assertTrue(ax.has_data())

    def test_featurewise_anomaly_score_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.featurewise_anomaly_score, self.X_test
        )

    @if_matplotlib
    def test_plot_graphical_model_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_graphical_model)

    @if_matplotlib
    def test_plot_partial_corrcoef_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.plot_partial_corrcoef)
