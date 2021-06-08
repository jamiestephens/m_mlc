# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 23:23:03 2021

@author: Jamie Stephens
"""
import pandas as pd
from scipy.io import arff
import glob


path = r'data/'
all_files = glob.glob(path + "*")

i = 0

for filename in all_files:
    data = arff.loadarff(filename)
    if i == 0:
        df = pd.DataFrame(data[0])
    else:
        df1 = pd.DataFrame(data[0])
        df = pd.concat([df, df1], axis=0)
    i = i + 1
s


## Confirm that 'class' is binary:
#print("Range of responses for class: ", df['class'].unique().tolist())
    
## Check number of ? or non-integer responses 
#print("Number of invalid entries: \n", df.isna().sum())

#print(df.describe())

#print(df.shape[0])