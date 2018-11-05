# -*- coding: utf-8 -*-
"""
Andrew Floyd
11/4/2018
CS3001 - HW5 (kNN)
Dr. Fu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def createDots(resultsDF2):
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('K vs. Accuracy of Prediction')
    plt.plot(resultsDF2, 'ro')

dtc = DecisionTreeClassifier(random_state=0)
iris = load_iris()
X = iris.data
y = iris.target
kf = KFold(5, True, 1)
total_accuK = 0
total_accuDT = 0
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print(accu)
    total_accuK = total_accuK + accu
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accu = accuracy_score(y_test, y_pred)
    print(accu)
    total_accuDT = total_accuDT + accu

aver_accu2 = total_accuK/5
print("Average Performance Accuracy (K): ", aver_accu2)
aver_accu1 = total_accuDT/5
print("Average Performance Accuracy (DT): ", aver_accu1)

averages = []
averages.append(aver_accu2)
averages.append(aver_accu1)

index = ['kNN', 'DecisionTree']
plt.figure(0)
plt.xlabel('Method')
plt.ylabel('Average Accuracy')
plt.title('kNN v. Decision Tree')
plt.bar(index, averages)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20)

results = []
for x in range(1, 60):
    classifier = KNeighborsClassifier(n_neighbors=x)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accu = accuracy_score(y_test, y_pred)
    results.append(accu)
    print(x, ":", accu)
    
resultsDF = { i : results[i] for i in range(0, len(results)) }
resultsDF2 = pd.DataFrame({'accu':results})

plt.figure(1)
#createDots(resultsDF2)

plt.figure(2)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('K vs. Accuracy of Prediction')
#plt.bar(list(resultsDF.keys()), resultsDF.values())

def createDots(resultsDF2):
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('K vs. Accuracy of Prediction')
    plt.plot(resultsDF2, 'ro')