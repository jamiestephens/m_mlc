# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:11:01 2021

@author: Jamie Stephens
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import eda
from sklearn.impute import SimpleImputer

df = eda.df
df = df.apply(pd.to_numeric)

df.replace('',np.NaN,inplace=True)
imp=SimpleImputer(missing_values=np.NaN)
idf=pd.DataFrame(imp.fit_transform(df))
idf.columns=df.columns
idf.index=df.index


#scaler = MinMaxScaler()
#scaled = scaler.fit_transform(idf)  
#df_scaled = pd.DataFrame(data=scaled)    

#df_nona = df.dropna()


#print("Number of invalid entries: \n", idf.isna().sum())

