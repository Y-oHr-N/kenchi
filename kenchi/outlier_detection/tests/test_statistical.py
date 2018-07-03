import doctest
import unittest

from kenchi.outlier_detection import statistical
from kenchi.tests.common_tests import OutlierDetectorTestMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import if_matplotlib


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(statistical))

    return tests


class GMMTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.GMM(random_state=0)


class KDETest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.KDE()


class HBOSTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.HBOS()

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_roc_auc_score(self):
        pass

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_plot_anomaly_score(self):
        pass

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_plot_roc_curve(self):
        pass


class SparseStructureLearningTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = statistical.SparseStructureLearning()

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
    def test_plot_partial_corrcoeff(self):
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
