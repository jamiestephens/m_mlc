# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:24:05 2021

@author: Jamie Stephens
"""
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
import preprocessing


df = pd.read_csv('test2.csv')

print(df.describe())

# define dataset
data = df.values
X, y = data[:, :-1], data[:, -1]
# define the model
model = DecisionTreeRegressor()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.xlabel("Feature No.")
pyplot.ylabel("Importance")
pyplot.show()

