import doctest
import unittest

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from kenchi.outlier_detection import clustering_based
from kenchi.tests.common_tests import ModelTestMixin, OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(clustering_based))

    return tests


class MiniBatchKMeansTest(
    unittest.TestCase, ModelTestMixin, OutlierDetectorTestMixin
):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = clustering_based.MiniBatchKMeans(random_state=0)

    def tearDown(self):
        plt.close()
