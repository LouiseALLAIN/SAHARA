# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import pickle
from os.path import isfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
import numpy as np
from sklearn.model_selection import ShuffleSplit

'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from preprocessing import Preprocessing

class ModelPreprocessed(BaseEstimator, ClassifierMixin):
    def __init__(self):
        '''
        Best model for classification in the preprocessed challenge
         args: None
        '''
        self.preproc = Preprocessing()
        self.clf = RandomForestClassifier(max_depth=100, n_estimators=100, random_state=30)

    def fit(self, X, y):
        '''
        Training the model.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        '''
        X = self.preproc.fit_transform(X)
        self.clf.fit(X, y)
        return self 
   
    def predict_proba(self, X):
        '''
        Compute probabilities to belong to given classes.
        '''        
        X = self.preproc.transform(X)
        y_proba = self.clf.predict_proba(X)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))
        
    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
        print("Model reloaded from: " + modelfile)
        return self

# Adpat to Codalab interface
# ModelPreprocessed for preprocessed challenge
# ModelRaw for raw challenge
model = ModelPreprocessed
