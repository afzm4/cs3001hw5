# -*- coding: utf-8 -*-
"""
Andrew Floyd
11/4/2018
CS3001 - HW5 (kNN)
Dr. Fu
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dtc = DecisionTreeClassifier(random_state=0)
iris = load_iris()
cvs = cross_val_score(dtc, iris.data, iris.target, cv=5)
X = iris.data
y = iris.target
kf = KFold(5, True, 1)
print(kf)
total_accu = 0
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print(accu)
    total_accu = total_accu + accu
    #print(y_pred)
    #print(y_test)

aver_accu = total_accu/5
print("Average Performance Accuracy: ", aver_accu)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20)

for x in range(1, 60):
    classifier = KNeighborsClassifier(n_neighbors=x)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accu = accuracy_score(y_test, y_pred)
    print(x, ":", accu)