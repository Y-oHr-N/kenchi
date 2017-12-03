import unittest

import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from kenchi.datasets import make_blobs
from kenchi.outlier_detection.statistical import KernelDensityOutlierDetector

mpl.use('Agg')


class KernelDensityOutlierDetectorTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(n_outliers=0)
        self.df   = pd.DataFrame(self.X)
        self.sut  = KernelDensityOutlierDetector()

    def test_fit(self):
        self.assertIsInstance(
            self.sut.fit(self.X), KernelDensityOutlierDetector
        )

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
            self.sut.fit(self.X).plot_anomaly_score(), mpl.axes.Axes
        )
