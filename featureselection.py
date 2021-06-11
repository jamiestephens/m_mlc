# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:24:05 2021

@author: Jamie Stephens
"""
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import matplotlib
import preprocessing

matplotlib.rc('axes',edgecolor='white')
df = preprocessing.idf
data = df.values
X, y = data[:, :-1], data[:, -1]

model = RandomForestClassifier()
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

datalabels = pd.read_csv('datalabels.csv')


for i in features:
    print(datalabels.loc[datalabels['df_label'] == i, 'label_name'])


plt.figure(facecolor="#2b2b2b")
plt.title("Feature Importances",color='w')
plt.bar([x for x in range(len(importance))], importance,color='w')
plt.xlabel("Feature",color='w')
plt.ylabel("Relative Importance",color='w')
ax = plt.axes()
ax.set_facecolor("#2b2b2b")
ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
ax.tick_params(axis='y', colors='white')

plt.grid(linestyle='-', linewidth=0.5)
plt.show()

