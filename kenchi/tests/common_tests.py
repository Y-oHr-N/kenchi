import unittest

import numpy as np
from kenchi.datasets import make_blobs
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import if_matplotlib


class OutlierDetectorTestMixin:
    def prepare_data(self):
        X, y              = make_blobs(
            centers       = 1,
            contamination = 0.1,
            n_features    = 2,
            n_samples     = 100,
            random_state  = 0
        )

        return train_test_split(X, y, random_state=0)

    @unittest.skip('this test fail in scikit-larn 0.19.1')
    def test_check_estimator(self):
        self.assertIsNone(check_estimator(self.sut))

    def test_fit(self):
        self.assertIsInstance(self.sut.fit(self.X_train), BaseEstimator)

    def test_fit_predict(self):
        y_pred = self.sut.fit_predict(self.X_train)

        self.assertEqual(y_pred.shape, self.y_train.shape)

    def test_predict(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        y_pred = self.sut.predict(self.X_test)

        self.assertEqual(y_pred.shape, self.y_test.shape)

    def test_predict_proba(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        n_samples, _ = self.X_test.shape
        n_classes    = 2
        y_score      = self.sut.predict_proba(self.X_test)

        self.assertEqual(y_score.shape, (n_samples, n_classes))

    def test_decision_function(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        y_score = self.sut.decision_function(self.X_test)

        self.assertEqual(y_score.shape, self.y_test.shape)

    def test_score_samples(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        score_samples = self.sut.score_samples(self.X_test)

        self.assertEqual(score_samples.shape, self.y_test.shape)
        self.assertLessEqual(np.max(score_samples), 0.)

    def test_anomaly_score(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        anomaly_score = self.sut.anomaly_score(self.X_test)

        self.assertEqual(anomaly_score.shape, self.y_test.shape)
        self.assertGreaterEqual(np.min(anomaly_score), 0.)

    def test_roc_auc_score(self):
        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        score_samples = self.sut.score_samples(self.X_test)
        score         = roc_auc_score(self.y_test, score_samples)

        self.assertGreaterEqual(score, 0.5)

    @if_matplotlib
    def test_plot_anomaly_score(self):
        import matplotlib.pyplot as plt

        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        ax = self.sut.plot_anomaly_score(self.X_test)

        plt.close('all')

        self.assertTrue(ax.has_data())

    @if_matplotlib
    def test_plot_roc_curve(self):
        import matplotlib.pyplot as plt

        if hasattr(self.sut, 'novelty'):
            self.sut.set_params(novelty=True)

        self.sut.fit(self.X_train)

        ax = self.sut.plot_roc_curve(self.X_test, self.y_test)

        plt.close('all')

        self.assertTrue(ax.has_data())

    def test_predict_notffied(self):
        self.assertRaises(NotFittedError, self.sut.predict, self.X_test)

    def test_predict_proba_notffied(self):
        self.assertRaises(NotFittedError, self.sut.predict_proba, self.X_test)

    def test_decision_function_notffied(self):
        self.assertRaises(
            NotFittedError, self.sut.decision_function, self.X_test
        )

    def test_score_samples_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.score_samples, self.X_test)

    def test_anomaly_score_notfitted(self):
        self.assertRaises(NotFittedError, self.sut.anomaly_score, self.X_test)

    @if_matplotlib
    def test_plot_anomaly_score_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.plot_anomaly_score, self.X_test
        )

    @if_matplotlib
    def test_plot_roc_curve_notfitted(self):
        self.assertRaises(
            NotFittedError, self.sut.plot_roc_curve, self.X_test, self.y_test
        )
