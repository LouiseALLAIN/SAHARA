# coding: utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

from model import ModelPreprocessed
from visual import plot_ROC

DIRECTORY = './figs/'

"""
Main script to test our model.

run with
python main.py

"""


def main():
    print("Hello world !")
    X, y = make_classification(n_samples=2000, n_features=19, n_classes=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = ModelPreprocessed()
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print("accuracy =", accuracy)
    y_proba = clf.predict_proba(X_test)
    y_decision = y_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_decision, pos_label=1)
    plot_ROC(fpr, tpr, directory=DIRECTORY, title="dummy ROC curve")


if __name__ == '__main__':
    main()