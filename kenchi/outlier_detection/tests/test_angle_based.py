import doctest
import unittest

from kenchi.outlier_detection import angle_based
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(angle_based))

    return tests


class FastABODTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = angle_based.FastABOD(n_neighbors=3)
