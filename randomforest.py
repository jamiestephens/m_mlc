# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:37:34 2021

@author: Jamie Stephens
"""
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


df = pd.read_csv('test2.csv')


df = df[['33','32','26','45','34','64']]
feature_cols = ['33','32','26','45','34']
X = df[feature_cols] # Features
y = df['64']

model = RandomForestClassifier()

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

row = [[-8.52381793,5.24451077,-12.14967704,-2.92949242,0.99314133,0.67326595,-0.38657932,1.27955683,-0.60712621,3.20807316,0.60504151,-1.38706415,8.92444588,-7.43027595,-2.33653219,1.10358169,0.21547782,1.05057966,0.6975331,0.26076035]]
yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])

print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))