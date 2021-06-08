# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:11:01 2021

@author: Jamie Stephens
"""
import pandas as pd
import numpy as np
from scipy.io import arff
from numpy import genfromtxt
from pymfe.mfe import MFE
import csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import std
import eda
from sklearn.impute import SimpleImputer

df = eda.df
df = df.apply(pd.to_numeric)

def minmax():    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)  
    df_scaled = pd.DataFrame(data=scaled)
    df_scaled.replace('',np.NaN,inplace=True)
    imp=SimpleImputer(missing_values=np.NaN)
    idf=pd.DataFrame(imp.fit_transform(df_scaled))
    idf.columns=df_scaled.columns
    idf.index=df_scaled.index
    df_scaled.to_csv('test2.csv')
    return df_scaled
    
def standardscaler():
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    standardscaler.df_scaled = pd.DataFrame(data=scaled)
    return standardscaler.df_scaled

def comparemethods():
    dataframe = standardscaler.df_scaled
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1]
    X = X.astype('float')
    y = LabelEncoder().fit_transform(y.astype('str'))
    # define the modeling pipeline
    model = LogisticRegression(solver='liblinear')
    scaler = MinMaxScaler()
    pipeline = Pipeline([('s',scaler),('m',model)])
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model
    m_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # summarize the result
    print('Accuracy: %.3f (%.3f)' % (mean(m_scores), std(m_scores)))
        
if __name__ == "__main__":
    minmax()