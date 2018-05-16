import doctest
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from kenchi.outlier_detection import ensemble
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(ensemble))

    return tests


class IForestTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = ensemble.IForest(random_state=0)

    def tearDown(self):
        plt.close()
