import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from kenchi.outlier_detection import KMeansOutlierDetector


class KMeansOutlierDetectorTest(unittest.TestCase):
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
        self.sut   = KMeansOutlierDetector()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), KMeansOutlierDetector)

    def test_fit_detect_ndarray(self):
        self.assertIsInstance(self.sut.fit_detect(self.X), np.ndarray)

    def test_fit_detect_dataframe(self):
        self.assertIsInstance(self.sut.fit_detect(self.df), pd.Series)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_detect_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.detect(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(
            self.sut.fit(self.X).plot_anomaly_score(self.X), mpl.axes.Axes
        )
