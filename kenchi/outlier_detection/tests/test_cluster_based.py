import unittest

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from sklearn.exceptions import NotFittedError

from kenchi.datasets import make_blobs
from kenchi.outlier_detection import MiniBatchKMeans

matplotlib.use('Agg')


class MiniBatchKMeansTest(unittest.TestCase):
    def setUp(self):
        self.X, _ = make_blobs(random_state=1)
        self.sut  = MiniBatchKMeans()

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X), MiniBatchKMeans)

    def test_fit_predict(self):
        self.assertIsInstance(self.sut.fit_predict(self.X), np.ndarray)

    def test_anomaly_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.anomaly_score(self.X)

    def test_feature_wise_anomaly_score_notimplemented(self):
        with self.assertRaises(NotImplementedError):
            self.sut.feature_wise_anomaly_score(self.X)

    def test_predict_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.predict(self.X)

    def test_score_notfitted(self):
        with self.assertRaises(NotFittedError):
            self.sut.score(self.X)

    def test_plot_anomaly_score(self):
        self.assertIsInstance(self.sut.fit(self.X).plot_anomaly_score(), Axes)
