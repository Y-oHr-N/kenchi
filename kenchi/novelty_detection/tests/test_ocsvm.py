import doctest
import unittest

import numpy as np
from kenchi.novelty_detection import ocsvm
from kenchi.tests.common_tests import OutlierDetectorTestMixin


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(ocsvm))

    return tests


class OCSVMTest(unittest.TestCase, OutlierDetectorTestMixin):
    def setUp(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.prepare_data()

        self.sut = ocsvm.OCSVM(random_state=0)

    def test_predict(self):
        super().test_predict()

        y_pred_sut       = self.sut.predict(self.X_test)
        y_pred_estimator = self.sut.estimator_.predict(self.X_test)

        np.testing.assert_equal(y_pred_sut, y_pred_estimator)
