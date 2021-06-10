# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:54:02 2021

@author: Administrator
"""
from numpy import mean
from numpy import std
import pandas as pd
import eda
import featureselection
import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

df = featureselection.df
data = df.values
X, y = data[:, :-1], data[:, -1]
print("got here okay")
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


model = GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model on the dataset
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))