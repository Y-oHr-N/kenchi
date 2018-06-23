import doctest
import unittest

from kenchi.outlier_detection import clustering_based
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(clustering_based))

    return tests


class MiniBatchKMeansTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = clustering_based.MiniBatchKMeans(random_state=0)
