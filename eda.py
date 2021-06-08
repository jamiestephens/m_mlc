# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:23:03 2021

@author: Administrator
"""
import pandas as pd
from scipy.io import arff


data = arff.loadarff('1year.arff')
df = pd.DataFrame(data[0])
    
print(df['class'].unique().tolist())
    
print(df.isna().sum())
    
print(df.isin(['?']).sum(axis=0))
