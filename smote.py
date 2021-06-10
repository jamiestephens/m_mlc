# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 03:32:03 2021

@author: Jamie Stephens
"""

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
import preprocessing
import pandas as pd

df = preprocessing.df_scaled

data = df.values
X, y = data[:, :-1], data[:, -1]
last = df.columns.values[-1:]
lastone = last[0]

counter = Counter(y)
# define pipeline
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
# transform the dataset
X, y = pipeline.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()

dfsmoted = pd.DataFrame(data=X)

dfsmoted[lastone] = y.tolist()
