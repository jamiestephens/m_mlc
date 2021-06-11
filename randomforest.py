# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:54:02 2021

@author: Jamie Stephens
"""
from numpy import mean
from numpy import std
import pandas as pd
import preprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier



def randomforest(weight):
    df = preprocessing.idf
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    
    # Hyperparameter tuning here:
    model = RandomForestClassifier(class_weight=weight)
    cv = StratifiedKFold(n_splits=5)
    # evaluate the model on the dataset
    n_scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, verbose=1)
    
    # report performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
    
    cnf_matrix = confusion_matrix(y_test, y_train)

if __name__ == "__main__":
    #gbb("balanced",-1)
    randomforest({0:1,1:10})