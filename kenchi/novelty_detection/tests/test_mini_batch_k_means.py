import doctest
import unittest

from kenchi.novelty_detection import mini_batch_k_means
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(mini_batch_k_means))

    return tests


class MiniBatchKMeansTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = mini_batch_k_means.MiniBatchKMeans(random_state=0)
