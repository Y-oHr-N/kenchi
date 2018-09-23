import doctest
import unittest

from kenchi.novelty_detection import kde
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(kde))

    return tests


class KDETest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = kde.KDE()
