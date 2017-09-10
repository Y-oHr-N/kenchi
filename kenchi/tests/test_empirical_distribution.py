import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from kenchi.outlier_detection import EmpiricalOutlierDetector


class EmpiricalOutlierDetectorTest(unittest.TestCase):
    def setUp(self):
        n_samples  = 1000
        n_features = 10
        rnd        = np.random.RandomState(0)
        mean       = np.zeros(n_features)
        cov        = np.eye(n_features)
        self.X     = rnd.multivariate_normal(mean, cov, n_samples)
        self.y     = np.zeros(n_samples, dtype=np.int32)
        self.sut   = EmpiricalOutlierDetector(fpr=0.0)

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), EmpiricalOutlierDetector)

    def test_fit_predict(self):
        self.assertTrue(np.allclose(self.sut.fit_predict(self.X), self.y))

    def test_decision_function_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.decision_function(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)
