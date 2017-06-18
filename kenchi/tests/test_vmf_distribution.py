from unittest import TestCase

import numpy as np
from kenchi import VMFDetector


class VMFDetectorTest(TestCase):
    def test_score(self):
        train_size = 1000
        test_size  = 100
        n_outliers = 10
        n_features = 10

        rnd        = np.random.RandomState(0)

        X_train    = rnd.normal(size=(train_size, n_features))

        X_test     = np.concatenate(
            (
                rnd.normal(size=(test_size - n_outliers, n_features)),
                rnd.uniform(-10.0, 10.0, size=(n_outliers, n_features))
            )
        )
        y_test     = np.concatenate(
            (
                np.zeros(test_size - n_outliers, dtype=np.int32),
                np.ones(n_outliers, dtype=np.int32),
            )
        )

        det        = VMFDetector(fpr=0.1)

        self.assertIsInstance(det.fit(X_train), VMFDetector)
        self.assertGreater(det.score(X_test, y_test), 0.0)
