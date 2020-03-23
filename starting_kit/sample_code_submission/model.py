'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import ShuffleSplit

class model (BaseEstimator):
   def __init__(self):
    self.classifier = RandomForestClassifier(max_depth=100, n_estimators=100, random_state=30)
        
        
   def fit(self, X, y):   
         self.classifier.fit(X, np.ravel(y))
         return self
   
   def predict(self, X):
        return self.classifier.predict(X)

   def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

   def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
        print("Model reloaded from: " + modelfile)
        return self
