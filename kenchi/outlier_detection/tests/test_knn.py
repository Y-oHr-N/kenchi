import doctest
import unittest

from kenchi.outlier_detection import knn
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(knn))

    return tests


class KNNTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = knn.KNN(n_neighbors=3)
