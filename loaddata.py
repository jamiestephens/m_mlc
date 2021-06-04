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


def loaddata():
    data = arff.loadarff('1year.arff')
    df = pd.DataFrame(data[0])
    
    print(df['class'].unique().tolist())

def logisticeregnoscaling():
    data = arff.loadarff('1year.arff')
    df = pd.DataFrame(data[0])
    
    df.hist()
    pyplot.show()
    
if __name__ == "__main__":
    loaddata()