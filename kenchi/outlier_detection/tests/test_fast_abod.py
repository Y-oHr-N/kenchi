import doctest
import unittest

from kenchi.outlier_detection import fast_abod
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(fast_abod))

    return tests


class FastABODTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = fast_abod.FastABOD(n_neighbors=3)
