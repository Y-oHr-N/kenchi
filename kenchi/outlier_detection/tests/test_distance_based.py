import doctest
import unittest

from kenchi.outlier_detection import distance_based
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(distance_based))

    return tests


class KNNTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = distance_based.KNN(n_neighbors=3)


class OneTimeSamplingTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = distance_based.OneTimeSampling(
            n_subsamples=3, random_state=0
        )
