import doctest
import unittest

from kenchi.outlier_detection import one_time_sampling
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(one_time_sampling))

    return tests


class OneTimeSamplingTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = one_time_sampling.OneTimeSampling(
            n_subsamples=3, random_state=0
        )
