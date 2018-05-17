import doctest
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from kenchi.outlier_detection import statistical
from kenchi.tests.common_tests import ModelTestMixin, OutlierDetectorTestMixin
from sklearn.exceptions import NotFittedError


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(statistical))

    return tests


class GMMTest(unittest.TestCase, ModelTestMixin, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.GMM(random_state=0)

    def tearDown(self):
        plt.close()


class KDETest(unittest.TestCase, ModelTestMixin, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.KDE()

    def tearDown(self):
        plt.close()


class HBOSTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.HBOS()

    def tearDown(self):
        plt.close()

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_roc_auc_score():
        pass

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_plot_anomaly_score():
        pass

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_plot_roc_curve():
        pass


class SparseStructureLearningTest(
    unittest.TestCase, ModelTestMixin, OutlierDetectorTestMixin
):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.SparseStructureLearning()

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

    def test_plot_partial_corrcoeff(self):
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
