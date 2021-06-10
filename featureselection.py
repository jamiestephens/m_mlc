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
import smote

df = smote.dfsmoted
data = df.values
X, y = data[:, :-1], data[:, -1]

model = DecisionTreeRegressor()
model.fit(X, y)
importance = model.feature_importances_

b = []

for i,v in enumerate(importance):
    a = [i,v]
    b.append(a)
    
sorter = lambda x: (x[1], x[0])
sorted_l = sorted(b, key=sorter,reverse=True)

# number of variables
n = 6

i = 2
features = []

for row in sorted_l[:n]:
    for elem in row:
        if (i % 2) == 0:
            features.append(elem)
        i = i + 1

last = df.columns.values[-1:]
lastone = last[0]
features.append(lastone)
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.xlabel("Feature No.")
pyplot.ylabel("Importance")
pyplot.show()