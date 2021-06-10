# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 22:21:50 2021

@author: Jamie Stephens
"""
from math import sqrt


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import featureselection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

df = featureselection.df
features = featureselection.features

df = df[features]

print(features)

data = df.values
X, y = data[:, :-1], data[:, -1]

#sns.set_style("whitegrid");
#sns.pairplot(df, hue=64, height=3);
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(data, y,test_size=0.4, random_state = 42)
# knn_clf=KNeighborsClassifier(n_neighbors=8)
# knn_clf.fit(X_train,y_train)
# ypred=knn_clf.predict(X_test)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy: ",metrics.accuracy_score(y_test, ypred))

range_k = range(1,15)
scores = {}
scores_list = []
for k in range_k:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
print(scores_list)
result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = metrics.classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)