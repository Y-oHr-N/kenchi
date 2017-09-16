import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from kenchi.outlier_detection import GaussianOutlierDetector
from kenchi.pipeline import ExtendedPipeline


class ExtendedPipelineTest(unittest.TestCase):
    def setUp(self):
        n_samples  = 1000
        n_features = 10
        rnd        = np.random.RandomState(0)
        self.X     = rnd.multivariate_normal(
            mean   = np.zeros(n_features),
            cov    = np.eye(n_features),
            size   = n_samples
        )
        self.df    = pd.DataFrame(self.X)
        self.sut   = ExtendedPipeline([
            ('standardize', StandardScaler()),
            ('detect',      GaussianOutlierDetector())
        ])

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), ExtendedPipeline)

    def test_fit_predict_ndarray(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)
