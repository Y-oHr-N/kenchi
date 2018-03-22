from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class CountEncoder(BaseEstimator, TransformerMixin):
    """Count encoder."""

    def __init__(self, categorical_features='all', copy=True):
        self.categorical_features = categorical_features
        self.copy                 = copy

    def _check_array(self, X, check_shape=False):
        X             = check_array(X, copy=self.copy, estimator=self)
        _, n_features = X.shape

        if check_shape and n_features != self.counters_.size:
            raise ValueError()

        return X

    def fit(self, X, y=None):
        X                        = self._check_array(X)
        _, n_features            = X.shape
        self.counters_           = np.empty(n_features, dtype=object)

        if self.categorical_features == 'all':
            categorical_features = np.arange(n_features)
        else:
            categorical_features = self.categorical_features

        for j in categorical_features:
            self.counters_[j]    = Counter(X[:, j])

        return self

    def transform(self, X):
        check_is_fitted(self, 'counters_')

        X                        = self._check_array(X, check_shape=True)
        _, n_features            = X.shape

        if self.categorical_features == 'all':
            categorical_features = np.arange(n_features)
        else:
            categorical_features = self.categorical_features

        for j in categorical_features:
            vect                 = np.vectorize(
                lambda xj: self.counters_[j][xj]
            )
            X[:, j]              = vect(X[:, j])

        return X
