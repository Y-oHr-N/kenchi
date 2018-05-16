import doctest
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from kenchi.outlier_detection import reconstruction_based
from kenchi.tests.common_tests import ModelTestMixin, OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(reconstruction_based))

    return tests


class PCATest(unittest.TestCase, ModelTestMixin, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = reconstruction_based.PCA()

    def tearDown(self):
        plt.close()
