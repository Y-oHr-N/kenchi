from unittest import TestCase

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ParameterGrid

from kenchi import GaussianOutlierDetector, GGMOutlierDetector


class GaussianOutlierDetectorTest(TestCase):
    def setUp(self):
        n_samples  = 1000
        n_features = 10
        rnd        = np.random.RandomState(0)
        mean       = np.zeros(n_features)
        cov        = np.eye(n_features)
        self.X     = rnd.multivariate_normal(mean, cov, n_samples)
        self.y     = np.zeros(n_samples, dtype=np.int32)
        self.sut   = GaussianOutlierDetector(fpr=0.0)

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), GaussianOutlierDetector)

    def test_fit_predict(self):
        self.assertTrue(np.allclose(self.sut.fit_predict(self.X), self.y))

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)


class GGMOutlierDetectorTest(TestCase):
    def setUp(self):
        n_samples  = 1000
        n_features = 10
        rnd        = np.random.RandomState(0)
        mean       = np.zeros(n_features)
        cov        = np.eye(n_features)
        self.X     = rnd.multivariate_normal(mean, cov, n_samples)
        self.y     = np.zeros(n_samples, dtype=np.int32)
        self.sut   = GGMOutlierDetector(fpr=0.0)

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), GGMOutlierDetector)

    def test_fit_predict(self):
        self.assertTrue(np.allclose(self.sut.fit_predict(self.X), self.y))

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)
