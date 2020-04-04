# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.cluster import KMeans


class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.kmeans = KMeans(n_clusters = 10)

    def fit(self, X, y=None):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        X_1 = self.kmeans.transform(X)
        X_new = np.concatenate((X, X_1), axis=1)
        return X_new