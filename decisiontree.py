# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:59:01 2021

@author: Jamie Stephens
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import featureselection

df = featureselection.df

df = df[['33','32','26','45','34','64']]

feature_cols = ['33','32','26','45','34']
X = df[feature_cols] # Features
y = df['64']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))