import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from kenchi.outlier_detection import VMFOutlierDetector


class VMFOutlierDetectorTest(unittest.TestCase):
    def setUp(self):
        n_samples  = 1000
        n_features = 10
        rnd        = np.random.RandomState(0)
        self.X     = rnd.multivariate_normal(
            mean   = np.zeros(n_features),
            cov    = np.eye(n_features),
            size   = n_samples
        )
        self.sut   = VMFOutlierDetector(fpr=0.0)

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), VMFOutlierDetector)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_decision_function_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.decision_function(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)
