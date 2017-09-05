from unittest import TestCase

import numpy as np
from sklearn.model_selection import ParameterGrid

from kenchi import GaussianDetector, GGMDetector


class GaussianDetectorTest(TestCase):
    def test_score(self):
        train_size  = 1000
        test_size   = 100
        n_outliers  = 10
        n_features  = 10

        rnd         = np.random.RandomState(0)

        mean        = np.zeros(n_features)
        cov         = np.eye(n_features)

        X_train     = rnd.multivariate_normal(mean, cov, train_size)

        X_test      = np.concatenate((
            rnd.multivariate_normal(mean, cov, test_size - n_outliers),
            rnd.uniform(-10.0, 10.0, size=(n_outliers, n_features))
        ))

        y_test      = np.concatenate((
            np.zeros(test_size - n_outliers, dtype=np.int32),
            np.ones(n_outliers, dtype=np.int32),
        ))

        param_grid  = {'use_method_of_moments': [False, True]}

        for params in ParameterGrid(param_grid):
            with self.subTest(**params):
                det = GaussianDetector().set_params(**params)

                self.assertIsInstance(det.fit(X_train), GaussianDetector)
                self.assertGreater(det.score(X_test, y_test), 0.5)


class GGMDetectorTest(TestCase):
    def test_score(self):
        train_size  = 1000
        test_size   = 100
        n_outliers  = 10
        n_features  = 10

        rnd         = np.random.RandomState(0)

        mean        = np.zeros(n_features)
        cov         = np.eye(n_features)

        X_train     = rnd.multivariate_normal(mean, cov, train_size)

        X_test      = np.concatenate((
            rnd.multivariate_normal(mean, cov, test_size - n_outliers),
            rnd.uniform(-10.0, 10.0, size=(n_outliers, n_features))
        ))

        y_test      = np.concatenate((
            np.zeros(test_size - n_outliers, dtype=np.int32),
            np.ones(n_outliers, dtype=np.int32),
        ))

        param_grid  = {'q': [99.9]}

        for params in ParameterGrid(param_grid):
            with self.subTest(**params):
                det = GGMDetector().set_params(**params)

                self.assertIsInstance(det.fit(X_train), GGMDetector)
                self.assertGreater(det.score(X_test, y_test), 0.5)
